import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import Evaluator
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from SaveEmb import SaveEmbeddings
from finetune_bert import finetune_bert_with_soft_prompt
from load_data import *
from models import *


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_loaders(data, batch_size=64):
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


def compute_acc_and_f1(y_true, y_pred, evaluator):
    acc = evaluator.eval(
        {"y_true": y_true.unsqueeze(1), "y_pred": y_pred.unsqueeze(1)}
    )["acc"]
    f1_macro = f1_score(
        y_true.cpu().numpy(),
        y_pred.cpu().numpy(),
        average="macro",
    )
    return acc, f1_macro


def train_model(
    model,
    train_loader,
    valid_loader,
    test_loader,
    dataset_name,
    epochs=10,
    mode="AE",
    num_classes=7,
):
    evaluator = Evaluator(
        name="ogbn-products" if dataset_name == "products" else "ogbn-arxiv"
    )
    device = next(model.parameters()).device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = float("-inf")
    best_model_state = None
    best_run_metrics = None
    log_file = f"result_log_{dataset_name}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,train_acc,train_f1_macro,"
            "val_loss,val_acc,val_f1_macro,test_acc,test_f1_macro\n"
        )

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        y_pred_train = []
        y_true_train = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]"):
            if batch.x.size(0) == 0:
                continue

            batch = batch.to(device)
            optimizer.zero_grad()

            outputs, _, _, _, _ = model(
                batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size
            )
            seed_outputs = outputs[: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size].squeeze().long()
            if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                continue

            one_hot_labels = F.one_hot(seed_labels, num_classes=num_classes).float()
            loss_task = criterion(seed_outputs, one_hot_labels)

            if mode == "AE" and hasattr(model, "last_ae_recon"):
                loss_recon = F.mse_loss(model.last_ae_recon, model.last_ae_input)
                loss = loss_task + 0.1 * loss_recon
            else:
                loss = loss_task

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            y_pred_train.append(seed_outputs.argmax(dim=1).cpu())
            y_true_train.append(seed_labels.cpu())

        y_pred_train = torch.cat(y_pred_train, dim=0)
        y_true_train = torch.cat(y_true_train, dim=0)
        train_acc, train_f1_macro = compute_acc_and_f1(
            y_true_train, y_pred_train, evaluator
        )

        model.eval()
        total_val_loss = 0.0
        y_pred_val = []
        y_true_val = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]"):
                batch = batch.to(device)
                outputs, _, _, _, _ = model(
                    batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size
                )
                seed_outputs = outputs[: batch.batch_size]
                seed_labels = batch.y[: batch.batch_size].squeeze().long()
                if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                    continue

                one_hot_labels = F.one_hot(seed_labels, num_classes=num_classes).float()
                total_val_loss += criterion(seed_outputs, one_hot_labels).item()
                y_pred_val.append(seed_outputs.argmax(dim=1).cpu())
                y_true_val.append(seed_labels.cpu())

        y_pred_val = torch.cat(y_pred_val, dim=0)
        y_true_val = torch.cat(y_true_val, dim=0).squeeze()
        val_acc, val_f1_macro = compute_acc_and_f1(y_true_val, y_pred_val, evaluator)

        y_pred_test = []
        y_true_test = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [test]"):
                batch = batch.to(device)
                outputs, _, _, _, _ = model(
                    batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size
                )
                seed_outputs = outputs[: batch.batch_size]
                seed_labels = batch.y[: batch.batch_size].squeeze().long()
                if seed_outputs.dim() == 0 or seed_labels.dim() == 0:
                    continue

                y_pred_test.append(seed_outputs.argmax(dim=1).cpu())
                y_true_test.append(seed_labels.cpu())

        y_pred_test = torch.cat(y_pred_test, dim=0)
        y_true_test = torch.cat(y_true_test, dim=0).squeeze()
        test_acc, test_f1_macro = compute_acc_and_f1(
            y_true_test, y_pred_test, evaluator
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_run_metrics = {
                "train_acc": train_acc,
                "train_f1_macro": train_f1_macro,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "test_acc": test_acc,
                "test_f1_macro": test_f1_macro,
            }

        scheduler.step(val_acc)

        train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss = total_val_loss / max(len(valid_loader), 1)
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1_macro:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1_macro:.4f}, "
            f"test_acc={test_acc:.4f}, test_f1={test_f1_macro:.4f}"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},{train_loss:.4f},{train_acc:.4f},{train_f1_macro:.4f},"
                f"{val_loss:.4f},{val_acc:.4f},{val_f1_macro:.4f},"
                f"{test_acc:.4f},{test_f1_macro:.4f}\n"
            )

    if best_model_state is None or best_run_metrics is None:
        raise RuntimeError("No valid metrics were collected during training.")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"{dataset_name}_model.pt")
    return best_run_metrics


def remap_labels(data):
    unique_labels = torch.unique(data.y)
    label_map = {
        old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)
    }
    data.y = torch.tensor([label_map[label.item()] for label in data.y])
    return data, len(unique_labels)


def get_predictions_fast(model, data, device, batch_size=1024):
    model.eval()
    num_nodes = data.x.size(0)
    num_classes = len(torch.unique(data.y))
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=batch_size,
        shuffle=False,
    )
    all_preds = torch.zeros(num_nodes, num_classes, device=device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions"):
            batch = batch.to(device)
            outputs, _, _, _, _ = model(
                batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size
            )
            seed_outputs = outputs[: batch.batch_size]
            seed_ids = batch.n_id[: batch.batch_size]
            all_preds[seed_ids] = F.softmax(seed_outputs, dim=-1)

    return all_preds


def label_propagation_fast(y_soft, edge_index, num_nodes, mask, alpha=0.5, num_iters=50):
    from torch_sparse import SparseTensor

    device = y_soft.device
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    adj = SparseTensor(
        row=row,
        col=col,
        value=deg_inv_sqrt[row] * deg_inv_sqrt[col],
        sparse_sizes=(num_nodes, num_nodes),
    ).to(device)

    y_init = y_soft.clone()
    y_prop = y_soft.clone()
    for _ in range(num_iters):
        y_new = adj @ y_prop
        y_prop = alpha * y_new + (1 - alpha) * y_soft
        y_prop[mask] = y_init[mask]
    return y_prop


def correct_and_smooth(
    model,
    data,
    device,
    correction_alpha=0.8,
    correction_iters=50,
    smoothing_alpha=0.8,
    smoothing_iters=50,
):
    evaluator = Evaluator(name="ogbn-arxiv")
    num_nodes = data.x.size(0)
    num_classes = len(torch.unique(data.y))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[data.train_idx] = True

    y_soft = get_predictions_fast(model, data, device)
    y_pred_base = y_soft.argmax(dim=-1)

    train_labels = data.y[train_mask].squeeze()
    train_onehot = F.one_hot(train_labels, num_classes).float()

    errors = torch.zeros(num_nodes, num_classes, device=device)
    errors[train_mask] = train_onehot - y_soft[train_mask]
    errors_prop = label_propagation_fast(
        errors,
        data.edge_index,
        num_nodes,
        train_mask,
        alpha=correction_alpha,
        num_iters=correction_iters,
    )

    y_corrected = y_soft + errors_prop
    y_corrected = y_corrected.clamp(min=0)
    y_corrected = y_corrected / y_corrected.sum(dim=1, keepdim=True)

    y_smooth = torch.zeros(num_nodes, num_classes, device=device)
    y_smooth[train_mask] = train_onehot
    y_smooth = label_propagation_fast(
        y_smooth,
        data.edge_index,
        num_nodes,
        train_mask,
        alpha=smoothing_alpha,
        num_iters=smoothing_iters,
    )

    y_final = y_corrected + y_smooth
    y_final = y_final / y_final.sum(dim=1, keepdim=True)
    y_pred_final = y_final.argmax(dim=-1)

    test_idx = data.test_idx
    y_true = data.y[test_idx].cpu()
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)

    base_acc = evaluator.eval(
        {"y_true": y_true, "y_pred": y_pred_base[test_idx].cpu().unsqueeze(1)}
    )["acc"]
    cs_acc = evaluator.eval(
        {"y_true": y_true, "y_pred": y_pred_final[test_idx].cpu().unsqueeze(1)}
    )["acc"]

    return y_pred_final, {
        "base_acc": base_acc,
        "cs_acc": cs_acc,
        "improvement": cs_acc - base_acc,
    }


def load_dataset(dataset_name):
    if dataset_name == "products-subset":
        data, texts = load_products_subset()
        data, _ = remap_labels(data)
    elif dataset_name == "products":
        data, texts = load_ogb_products()
    elif dataset_name == "citeseer":
        data, texts = load_citeseer()
    elif dataset_name == "photo":
        data, texts = load_photo()
    elif dataset_name in ["arxiv", "arxiv_sim"]:
        data, texts = load_arxiv(dataset_name=dataset_name)
    elif dataset_name == "cora":
        data, texts = load_cora()
    elif dataset_name == "pubmed":
        data, texts = load_pubmed()
    elif dataset_name == "arxiv_2023":
        data, texts = load_arxiv_2023()
        data, _ = remap_labels(data)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return data, texts


def build_summary(run_results, dataset_name, model_name, cs_results=None):
    runs_df = pd.DataFrame(run_results)
    summary_row = {
        "dataset": dataset_name,
        "model": model_name,
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
    if cs_results is not None:
        summary_row.update(
            {
                "cs_base_acc": cs_results["base_acc"],
                "cs_acc": cs_results["cs_acc"],
                "cs_improvement": cs_results["improvement"],
            }
        )
    return pd.DataFrame(run_results), pd.DataFrame([summary_row])


def main():
    parser = argparse.ArgumentParser(description="BiGTex node classification")
    parser.add_argument("dataset_name", default="cora", help="dataset name")
    parser.add_argument("model_name", default="BiGTex", help="GCN, GAT, SAGE, BiGTex")
    parser.add_argument("--batch_size", default=64, type=int, help="training batch size")
    parser.add_argument("--epochs", default=30, type=int, help="training epochs")
    parser.add_argument("--num_layers", default=2, type=int, help="number of GNN layers")
    parser.add_argument("--embedding_dim", default=768, type=int, help="embedding size")
    parser.add_argument("--num_iterate", default=5, type=int, help="number of runs")
    parser.add_argument(
        "--language_model_name",
        default="SCIBERT",
        help="BERT, GPT, SCIBERT, DeBERTA",
    )
    parser.add_argument("--soft_prompting", default="True", help="True or False")
    parser.add_argument("--Lora", default="True", help="True or False")
    parser.add_argument("--GNN", default="sage", help="gcn, gat, sage")
    parser.add_argument("--finetune_epochs", default=10, type=int, help="PLM finetune epochs")
    parser.add_argument(
        "--finetune_batch_size", default=16, type=int, help="PLM finetune batch size"
    )
    parser.add_argument("--mode", default="MLP", help="AE, MLP, cross")
    parser.add_argument("--use_adaptive", default="True", help="True or False")
    parser.add_argument("--results_dir", default="results", help="directory for csv outputs")
    parser.add_argument("--save_embeddings", action="store_true", help="save embeddings csv")
    parser.add_argument("--run_cs", action="store_true", help="run Correct&Smooth")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, texts = load_dataset(args.dataset_name)
    data = data.to(device)
    num_classes = len(torch.unique(data.y))
    print(f"Number of classes: {num_classes}")

    pretrained_lm_model = None
    use_pretrained_lm = False
    if args.model_name == "BiGTex":
        print(f"\n{'#' * 80}")
        print("# Stage1: fine-tuning the PLM")
        print(f"{'#' * 80}\n")
        pretrained_lm_model = finetune_bert_with_soft_prompt(
            data=data,
            texts=texts,
            feature_dim=data.x.shape[1],
            num_classes=num_classes,
            dataset_name=args.dataset_name,
            LM=args.language_model_name,
            model_save_dir="finetuned_models",
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            device=device,
        )
        for param in pretrained_lm_model.parameters():
            param.requires_grad = False
        pretrained_lm_model.eval()
        use_pretrained_lm = True
        print(f"\n{'#' * 80}")
        print("# Stage2: main training with fine-tuned PLM")
        print(f"{'#' * 80}\n")

    train_loader, valid_loader, test_loader = prepare_loaders(data, args.batch_size)
    run_results = []
    final_model = None

    for i in range(args.num_iterate):
        current_seed = 42 + i
        set_seed(current_seed)
        print(f"\nITERATION {i + 1}/{args.num_iterate}")
        print("=" * 50)

        model = AdaptiveGraphTextModel(
            feature_dim=data.x.shape[1],
            text_embedding_dim=768,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            texts=texts,
            num_gcn_layers=args.num_layers,
            Lora=str2bool(args.Lora),
            soft=str2bool(args.soft_prompting),
            LM=args.language_model_name,
            GNN=args.GNN,
            use_pretrained_lm=use_pretrained_lm,
            pretrained_lm_model=pretrained_lm_model,
            mode=args.mode,
            use_adaptive=str2bool(args.use_adaptive),
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.model_name}")
        print(f"Mode_Fusion: {args.mode}")
        print(f"Language Model: {args.language_model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%\n")

        run_metrics = train_model(
            model,
            train_loader,
            valid_loader,
            test_loader,
            args.dataset_name,
            epochs=args.epochs,
            mode=args.mode,
            num_classes=num_classes,
        )
        run_metrics.update(
            {
                "dataset": args.dataset_name,
                "model": args.model_name,
                "run": i + 1,
                "seed": current_seed,
            }
        )
        run_results.append(run_metrics)
        final_model = model

    cs_results = None
    best_model_for_post = AdaptiveGraphTextModel(
        feature_dim=data.x.shape[1],
        text_embedding_dim=768,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        texts=texts,
        num_gcn_layers=args.num_layers,
        Lora=str2bool(args.Lora),
        soft=str2bool(args.soft_prompting),
        LM=args.language_model_name,
        GNN=args.GNN,
        use_pretrained_lm=use_pretrained_lm,
        pretrained_lm_model=pretrained_lm_model,
        mode=args.mode,
        use_adaptive=str2bool(args.use_adaptive),
    ).to(device)
    best_model_for_post.load_state_dict(
        torch.load(f"{args.dataset_name}_model.pt", map_location=device)
    )
    best_model_for_post.eval()

    if args.run_cs:
        _, cs_results = correct_and_smooth(best_model_for_post, data, device)

    runs_df, summary_df = build_summary(
        run_results, args.dataset_name, args.model_name, cs_results
    )
    runs_csv_path = os.path.join(args.results_dir, f"{args.dataset_name}_runs.csv")
    summary_csv_path = os.path.join(args.results_dir, f"{args.dataset_name}_summary.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    if args.save_embeddings and final_model is not None:
        SaveEmbeddings(
            best_model_for_post,
            train_loader,
            valid_loader,
            test_loader,
            args.dataset_name,
            args.model_name,
        )

    summary = summary_df.iloc[0]
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean Train Acc: {summary['train_acc_mean']:.4f} ± {summary['train_acc_std']:.4f}")
    print(
        f"Mean Train F1-macro: {summary['train_f1_macro_mean']:.4f} ± "
        f"{summary['train_f1_macro_std']:.4f}"
    )
    print(f"Mean Val Acc: {summary['val_acc_mean']:.4f} ± {summary['val_acc_std']:.4f}")
    print(
        f"Mean Val F1-macro: {summary['val_f1_macro_mean']:.4f} ± "
        f"{summary['val_f1_macro_std']:.4f}"
    )
    print(f"Mean Test Acc: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(
        f"Mean Test F1-macro: {summary['test_f1_macro_mean']:.4f} ± "
        f"{summary['test_f1_macro_std']:.4f}"
    )
    print(f"Saved run metrics to: {runs_csv_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
