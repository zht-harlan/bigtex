import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from offline_dataset_utils import (
    load_cached_text_embeddings,
    load_dataset_with_texts,
    normalize_dataset_name,
)
from purified_graph_models import QuantizedPurifiedGraphEncoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_loaders(data, batch_size, num_layers):
    num_neighbors = [-1] * max(num_layers, 1)
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_idx,
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.valid_idx,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.test_idx,
    )
    return train_loader, valid_loader, test_loader


def compute_acc_and_f1(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    acc = float((y_true_np == y_pred_np).mean())
    f1_macro = f1_score(y_true_np, y_pred_np, average="macro")
    return acc, f1_macro


def evaluate(model, loader, device, split_name, quantization_weight):
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_quant_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            seed_logits = outputs["logits"][: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size].view(-1).long()

            task_loss = F.cross_entropy(seed_logits, seed_labels)
            quant_loss = outputs["quantization_loss"]
            loss = task_loss + quantization_weight * quant_loss

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_quant_loss += quant_loss.item()
            y_true_all.append(seed_labels.cpu())
            y_pred_all.append(seed_logits.argmax(dim=-1).cpu())

    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    acc, f1_macro = compute_acc_and_f1(y_true, y_pred)
    avg_denom = max(len(loader), 1)
    return {
        "loss": total_loss / avg_denom,
        "task_loss": total_task_loss / avg_denom,
        "quant_loss": total_quant_loss / avg_denom,
        "acc": acc,
        "f1_macro": f1_macro,
    }


def train_one_run(
    model,
    train_loader,
    valid_loader,
    test_loader,
    device,
    epochs,
    lr,
    weight_decay,
    quantization_weight,
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = float("-inf")
    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_quant_loss = 0.0
        train_true = []
        train_pred = []
        first_batch_logged = False

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]"):
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch.x, batch.edge_index)
            seed_logits = outputs["logits"][: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size].view(-1).long()

            task_loss = F.cross_entropy(seed_logits, seed_labels)
            quant_loss = outputs["quantization_loss"]
            loss = task_loss + quantization_weight * quant_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_quant_loss += quant_loss.item()
            train_true.append(seed_labels.detach().cpu())
            train_pred.append(seed_logits.argmax(dim=-1).detach().cpu())

            if not first_batch_logged:
                print(
                    "Sanity check batch shapes: "
                    f"Ze={tuple(outputs['ze'][: batch.batch_size].shape)}, "
                    f"codes={tuple(outputs['code_indices'][: batch.batch_size].shape)}, "
                    f"Zq={tuple(outputs['zq'][: batch.batch_size].shape)}, "
                    f"logits={tuple(seed_logits.shape)}"
                )
                first_batch_logged = True

        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        train_acc, train_f1 = compute_acc_and_f1(train_true, train_pred)
        train_loss = total_loss / max(len(train_loader), 1)
        train_task_loss = total_task_loss / max(len(train_loader), 1)
        train_quant_loss = total_quant_loss / max(len(train_loader), 1)

        val_metrics = evaluate(model, valid_loader, device, "val", quantization_weight)
        test_metrics = evaluate(model, test_loader, device, "test", quantization_weight)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_task={train_task_loss:.4f}, train_quant={train_quant_loss:.4f}, "
            f"train_acc={train_acc:.4f}, train_f1={train_f1:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}, val_f1={val_metrics['f1_macro']:.4f}, "
            f"test_acc={test_metrics['acc']:.4f}, test_f1={test_metrics['f1_macro']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "train_loss": train_loss,
                "train_task_loss": train_task_loss,
                "train_quant_loss": train_quant_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_metrics["loss"],
                "val_task_loss": val_metrics["task_loss"],
                "val_quant_loss": val_metrics["quant_loss"],
                "val_acc": val_metrics["acc"],
                "val_f1_macro": val_metrics["f1_macro"],
                "test_loss": test_metrics["loss"],
                "test_task_loss": test_metrics["task_loss"],
                "test_quant_loss": test_metrics["quant_loss"],
                "test_acc": test_metrics["acc"],
                "test_f1_macro": test_metrics["f1_macro"],
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return best_metrics


def build_summary(run_results, dataset_name, gnn_type):
    runs_df = pd.DataFrame(run_results)
    summary = {
        "dataset": dataset_name,
        "model": "QuantizedPurifiedGraphEncoder",
        "gnn_type": gnn_type,
        "num_runs": len(run_results),
        "train_acc_mean": runs_df["train_acc"].mean(),
        "train_acc_std": runs_df["train_acc"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "train_f1_macro_mean": runs_df["train_f1_macro"].mean(),
        "train_f1_macro_std": runs_df["train_f1_macro"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "val_acc_mean": runs_df["val_acc"].mean(),
        "val_acc_std": runs_df["val_acc"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "val_f1_macro_mean": runs_df["val_f1_macro"].mean(),
        "val_f1_macro_std": runs_df["val_f1_macro"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "test_acc_mean": runs_df["test_acc"].mean(),
        "test_acc_std": runs_df["test_acc"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "test_f1_macro_mean": runs_df["test_f1_macro"].mean(),
        "test_f1_macro_std": runs_df["test_f1_macro"].std(ddof=1) if len(run_results) > 1 else 0.0,
        "test_quant_loss_mean": runs_df["test_quant_loss"].mean(),
        "test_quant_loss_std": runs_df["test_quant_loss"].std(ddof=1) if len(run_results) > 1 else 0.0,
    }
    return runs_df, pd.DataFrame([summary])


def save_final_artifacts(model, data, device, output_dir, dataset_name):
    model.eval()
    with torch.no_grad():
        outputs = model(data.x.to(device), data.edge_index.to(device))
        ze = outputs["ze"].cpu()
        zq = outputs["zq"].cpu()
        code_indices = outputs["code_indices"].cpu()
    ze_path = os.path.join(output_dir, f"{dataset_name}_continuous_ze.pt")
    zq_path = os.path.join(output_dir, f"{dataset_name}_quantized_zq.pt")
    code_path = os.path.join(output_dir, f"{dataset_name}_code_indices.pt")
    torch.save(ze, ze_path)
    torch.save(zq, zq_path)
    torch.save(code_indices, code_path)
    print(f"Saved Ze tensor to: {ze_path}")
    print(f"Saved quantized Zq tensor to: {zq_path}")
    print(f"Saved code indices to: {code_path}")


def load_stage3_checkpoint_into_stage4_encoder(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage-3 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.encoder.load_state_dict(
        state_dict, strict=False
    )
    print(f"Loaded stage-3 encoder weights from: {checkpoint_path}")
    if missing_keys:
        print(f"Missing keys when loading stage-3 weights: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys when loading stage-3 weights: {len(unexpected_keys)}")


def main():
    parser = argparse.ArgumentParser(description="Train stage-4 quantized graph encoder on cached text embeddings")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument("--artifact_root", default="offline_artifacts", help="cached embedding root")
    parser.add_argument("--embedding_filename", default="text_embeddings.pt", help="embedding tensor filename")
    parser.add_argument("--results_dir", default="quantized_graph_results", help="output directory")
    parser.add_argument("--hidden_dim", default=256, type=int, help="continuous Ze dimension")
    parser.add_argument("--num_layers", default=2, type=int, help="number of GNN layers")
    parser.add_argument("--gnn_type", default="sage", choices=["sage", "gcn", "gat"], help="GNN backbone")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout rate")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--epochs", default=30, type=int, help="training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--num_runs", default=1, type=int, help="number of repeated runs")
    parser.add_argument("--seed_base", default=42, type=int, help="base random seed")
    parser.add_argument("--codebook_size", default=128, type=int, help="codebook size per quantizer")
    parser.add_argument("--num_quantizers", default=3, type=int, help="number of residual quantizers")
    parser.add_argument("--quantizer_dim", default=0, type=int, help="internal quantizer dim, 0 means hidden_dim")
    parser.add_argument("--quantization_weight", default=1.0, type=float, help="weight for quantization loss")
    parser.add_argument("--commitment_weight", default=0.25, type=float, help="commitment loss weight")
    parser.add_argument("--disable_straight_through", action="store_true", help="disable straight-through estimator")
    parser.add_argument("--disable_commitment_loss", action="store_true", help="disable commitment/codebook losses")
    parser.add_argument("--save_quantized_artifacts", action="store_true", help="save Ze/Zq/code indices")
    parser.add_argument(
        "--stage3_checkpoint",
        default="",
        help="optional stage-3 checkpoint path to initialize graph encoder",
    )
    args = parser.parse_args()

    start_time = time.time()
    dataset_name = normalize_dataset_name(args.dataset_name)
    dataset_results_dir = os.path.join(args.results_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed_base)
    _, data, _ = load_dataset_with_texts(dataset_name)
    cached_embeddings, embedding_path = load_cached_text_embeddings(
        dataset_name=dataset_name,
        artifact_root=args.artifact_root,
        filename=args.embedding_filename,
    )
    if cached_embeddings.size(0) != data.num_nodes:
        raise ValueError(
            f"Embedding row count {cached_embeddings.size(0)} does not match num_nodes {data.num_nodes}."
        )

    data.x = cached_embeddings
    input_dim = data.x.size(1)
    num_classes = int(torch.unique(data.y).numel())
    quantizer_dim = args.quantizer_dim if args.quantizer_dim > 0 else args.hidden_dim

    print(f"Loaded cached embeddings from: {embedding_path}")
    print(f"cached_text_embedding shape: {tuple(data.x.shape)}")
    print(
        "Configured quantizer: "
        f"num_quantizers={args.num_quantizers}, codebook_size={args.codebook_size}, quantizer_dim={quantizer_dim}"
    )

    run_results = []
    best_model_state = None
    best_model_seed = None
    best_val_acc = float("-inf")

    for run_idx in range(args.num_runs):
        current_seed = args.seed_base + run_idx
        set_seed(current_seed)
        print(f"\nRUN {run_idx + 1}/{args.num_runs}")
        print("=" * 60)

        train_loader, valid_loader, test_loader = prepare_loaders(
            data, args.batch_size, args.num_layers
        )
        model = QuantizedPurifiedGraphEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            gnn_type=args.gnn_type,
            num_layers=args.num_layers,
            dropout=args.dropout,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            quantizer_dim=quantizer_dim,
            use_straight_through=not args.disable_straight_through,
            use_commitment_loss=not args.disable_commitment_loss,
            commitment_weight=args.commitment_weight,
            debug_shapes=True,
        ).to(device)
        load_stage3_checkpoint_into_stage4_encoder(
            model, args.stage3_checkpoint, device
        )

        metrics = train_one_run(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            quantization_weight=args.quantization_weight,
        )
        metrics.update(
            {
                "dataset": dataset_name,
                "run": run_idx + 1,
                "seed": current_seed,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_quantizers": args.num_quantizers,
                "codebook_size": args.codebook_size,
                "quantizer_dim": quantizer_dim,
            }
        )
        run_results.append(metrics)

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_model_seed = current_seed

    runs_df, summary_df = build_summary(run_results, dataset_name, args.gnn_type)
    runs_csv_path = os.path.join(dataset_results_dir, f"{dataset_name}_quantized_graph_runs.csv")
    summary_csv_path = os.path.join(dataset_results_dir, f"{dataset_name}_quantized_graph_summary.csv")
    checkpoint_path = os.path.join(dataset_results_dir, f"{dataset_name}_quantized_graph_encoder.pt")

    runs_df.to_csv(runs_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    torch.save(
        {
            "dataset": dataset_name,
            "seed": best_model_seed,
            "model_state_dict": best_model_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": num_classes,
            "gnn_type": args.gnn_type,
            "num_layers": args.num_layers,
            "num_quantizers": args.num_quantizers,
            "codebook_size": args.codebook_size,
            "quantizer_dim": quantizer_dim,
        },
        checkpoint_path,
    )

    if args.save_quantized_artifacts and best_model_state is not None:
        best_model = QuantizedPurifiedGraphEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            gnn_type=args.gnn_type,
            num_layers=args.num_layers,
            dropout=args.dropout,
            codebook_size=args.codebook_size,
            num_quantizers=args.num_quantizers,
            quantizer_dim=quantizer_dim,
            use_straight_through=not args.disable_straight_through,
            use_commitment_loss=not args.disable_commitment_loss,
            commitment_weight=args.commitment_weight,
            debug_shapes=False,
        ).to(device)
        best_model.load_state_dict(best_model_state)
        save_final_artifacts(best_model, data, device, dataset_results_dir, dataset_name)

    print(f"Saved run metrics to: {runs_csv_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved best checkpoint to: {checkpoint_path}")
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
