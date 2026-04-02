import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn, optim
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from offline_dataset_utils import (
    load_cached_text_embeddings,
    load_dataset_with_texts,
    normalize_dataset_name,
)
from purified_graph_models import PurifiedGraphEncoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_loaders(data, batch_size):
    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=batch_size,
        input_nodes=data.train_idx,
    )
    valid_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=batch_size,
        input_nodes=data.valid_idx,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
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


def evaluate(model, loader, device, split_name):
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            batch = batch.to(device)
            logits, ze = model(batch.x, batch.edge_index)
            seed_logits = logits[: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size].view(-1).long()

            loss = F.cross_entropy(seed_logits, seed_labels)
            total_loss += loss.item()
            y_true_all.append(seed_labels.cpu())
            y_pred_all.append(seed_logits.argmax(dim=-1).cpu())

    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    acc, f1_macro = compute_acc_and_f1(y_true, y_pred)
    avg_loss = total_loss / max(len(loader), 1)
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
    }


def train_one_run(model, train_loader, valid_loader, test_loader, device, epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = float("-inf")
    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_true = []
        train_pred = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]"):
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, ze = model(batch.x, batch.edge_index)
            seed_logits = logits[: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size].view(-1).long()

            loss = F.cross_entropy(seed_logits, seed_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_true.append(seed_labels.detach().cpu())
            train_pred.append(seed_logits.argmax(dim=-1).detach().cpu())

        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        train_acc, train_f1 = compute_acc_and_f1(train_true, train_pred)
        train_loss = total_loss / max(len(train_loader), 1)

        val_metrics = evaluate(model, valid_loader, device, "val")
        test_metrics = evaluate(model, test_loader, device, "test")

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, val_f1={val_metrics['f1_macro']:.4f}, "
            f"test_acc={test_metrics['acc']:.4f}, test_f1={test_metrics['f1_macro']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1_macro": val_metrics["f1_macro"],
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
        "model": "PurifiedGraphEncoder",
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
    }
    return runs_df, pd.DataFrame([summary])


def save_final_node_embeddings(model, data, device, output_path):
    model.eval()
    with torch.no_grad():
        ze = model.encode(data.x.to(device), data.edge_index.to(device)).cpu()
    torch.save(ze, output_path)


def main():
    parser = argparse.ArgumentParser(description="Train stage-3 graph encoder on cached text embeddings")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument("--artifact_root", default="offline_artifacts", help="cached embedding root")
    parser.add_argument("--embedding_filename", default="text_embeddings.pt", help="embedding tensor filename")
    parser.add_argument("--results_dir", default="purified_graph_results", help="output directory")
    parser.add_argument("--hidden_dim", default=256, type=int, help="hidden dimension for Ze")
    parser.add_argument("--num_layers", default=2, type=int, help="number of GNN layers")
    parser.add_argument("--gnn_type", default="sage", choices=["sage", "gcn", "gat"], help="GNN backbone")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout rate")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--epochs", default=30, type=int, help="training epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--num_runs", default=1, type=int, help="number of repeated runs")
    parser.add_argument("--seed_base", default=42, type=int, help="base random seed")
    parser.add_argument("--save_ze", action="store_true", help="save final full-graph Ze")
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
    print(f"Loaded cached embeddings from: {embedding_path}")
    print(f"cached_text_embedding shape: {tuple(data.x.shape)}")
    print(f"Projected Ze shape will be: ({data.num_nodes}, {args.hidden_dim})")
    print(f"Number of classes: {num_classes}")

    run_results = []
    best_model_state = None
    best_model_seed = None
    best_val_acc = float("-inf")

    for run_idx in range(args.num_runs):
        current_seed = args.seed_base + run_idx
        set_seed(current_seed)
        print(f"\nRUN {run_idx + 1}/{args.num_runs}")
        print("=" * 60)

        train_loader, valid_loader, test_loader = prepare_loaders(data, args.batch_size)
        model = PurifiedGraphEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            gnn_type=args.gnn_type,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        metrics = train_one_run(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        metrics.update(
            {
                "dataset": dataset_name,
                "run": run_idx + 1,
                "seed": current_seed,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            }
        )
        run_results.append(metrics)

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_model_seed = current_seed

    runs_df, summary_df = build_summary(run_results, dataset_name, args.gnn_type)
    runs_csv_path = os.path.join(dataset_results_dir, f"{dataset_name}_purified_graph_runs.csv")
    summary_csv_path = os.path.join(dataset_results_dir, f"{dataset_name}_purified_graph_summary.csv")
    checkpoint_path = os.path.join(dataset_results_dir, f"{dataset_name}_purified_graph_encoder.pt")

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
        },
        checkpoint_path,
    )

    if args.save_ze and best_model_state is not None:
        best_model = PurifiedGraphEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            gnn_type=args.gnn_type,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        best_model.load_state_dict(best_model_state)
        ze_path = os.path.join(dataset_results_dir, f"{dataset_name}_ze.pt")
        save_final_node_embeddings(best_model, data, device, ze_path)
        print(f"Saved Ze tensor to: {ze_path}")

    print(f"Saved run metrics to: {runs_csv_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved best checkpoint to: {checkpoint_path}")
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
