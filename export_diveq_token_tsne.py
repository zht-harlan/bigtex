import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from fusion_models_diveq import CrossModalFusionDiVeQPLM
from offline_dataset_utils import (
    load_cached_text_embeddings,
    load_dataset_with_texts,
    load_refined_texts,
    normalize_dataset_name,
)


def get_seed_texts(refined_texts, batch):
    seed_ids = batch.n_id[: batch.batch_size].detach().cpu().tolist()
    return [refined_texts[node_id] for node_id in seed_ids]


def prepare_all_loader(data, batch_size, num_layers):
    num_neighbors = [-1] * max(num_layers, 1)
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=torch.arange(data.num_nodes),
        shuffle=False,
    )


def sample_per_class(embeddings, labels, codes, max_points_per_class):
    labels_np = np.asarray(labels)
    embeddings_np = np.asarray(embeddings)
    codes_np = np.asarray(codes)
    keep_indices = []

    for label in np.unique(labels_np):
        label_indices = np.where(labels_np == label)[0]
        if len(label_indices) > max_points_per_class:
            selected = np.random.choice(label_indices, max_points_per_class, replace=False)
        else:
            selected = label_indices
        keep_indices.extend(selected.tolist())

    keep_indices = np.array(sorted(keep_indices))
    return embeddings_np[keep_indices], labels_np[keep_indices], codes_np[keep_indices]


def main():
    parser = argparse.ArgumentParser(description="Export DiVeQ struct-token embeddings and create a t-SNE plot.")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument("--checkpoint_path", required=True, help="path to *_diveq_joint_model.pt")
    parser.add_argument("--data_root", default="datasets", help="root directory containing local dataset folders")
    parser.add_argument("--artifact_root", default="offline_artifacts", help="offline artifact root")
    parser.add_argument("--embedding_filename", default="text_embeddings.pt", help="cached embedding filename")
    parser.add_argument("--refined_texts_filename", default="refined_texts.jsonl", help="refined texts filename")
    parser.add_argument("--batch_size", default=256, type=int, help="inference batch size")
    parser.add_argument("--max_points_per_class", default=500, type=int, help="max t-SNE points per class")
    parser.add_argument("--output_dir", default="tsne_outputs_diveq", help="output directory")
    args = parser.parse_args()

    dataset_name = normalize_dataset_name(args.dataset_name)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    _, data, _ = load_dataset_with_texts(dataset_name, data_root=args.data_root)
    cached_embeddings, _ = load_cached_text_embeddings(
        dataset_name=dataset_name,
        artifact_root=args.artifact_root,
        filename=args.embedding_filename,
    )
    refined_texts, _ = load_refined_texts(
        dataset_name=dataset_name,
        artifact_root=args.artifact_root,
        filename=args.refined_texts_filename,
    )
    data.x = cached_embeddings

    input_dim = data.x.size(1)
    num_classes = int(torch.unique(data.y).numel())
    quantizer_dim = checkpoint.get("quantizer_dim", checkpoint["hidden_dim"])

    model = CrossModalFusionDiVeQPLM(
        input_dim=input_dim,
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=num_classes,
        gnn_type=checkpoint["gnn_type"],
        num_layers=checkpoint["num_layers"],
        graph_dropout=0.2,
        codebook_size=checkpoint["codebook_size"],
        quantizer_dim=quantizer_dim,
        backbone_name=checkpoint["backbone_name"],
        max_text_length=256,
        use_lora=True,
        lora_r=checkpoint.get("lora_r", 8),
        lora_alpha=checkpoint.get("lora_alpha", 32),
        lora_dropout=checkpoint.get("lora_dropout", 0.1),
        freeze_plm_embeddings=False,
        enable_vq_aux_head=checkpoint.get("enable_vq_aux_head", False),
        debug_shapes=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = prepare_all_loader(data, args.batch_size, checkpoint["num_layers"])
    struct_embeddings = []
    labels = []
    codes = []
    node_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting DiVeQ token embeddings"):
            seed_texts = get_seed_texts(refined_texts, batch)
            outputs = model(batch.x, batch.edge_index, seed_texts, batch_size=batch.batch_size)
            struct_token = model.struct_token_projection(outputs["zq"]).cpu().numpy()
            struct_embeddings.append(struct_token)
            labels.append(batch.y[: batch.batch_size].view(-1).cpu().numpy())
            codes.append(outputs["code_indices"].view(-1).cpu().numpy())
            node_ids.append(batch.n_id[: batch.batch_size].cpu().numpy())

    struct_embeddings = np.concatenate(struct_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    codes = np.concatenate(codes, axis=0)
    node_ids = np.concatenate(node_ids, axis=0)

    export_csv_path = os.path.join(output_dir, f"{dataset_name}_struct_token_embeddings.csv")
    pd.DataFrame(
        {
            "node_id": node_ids,
            "label": labels,
            "code": codes,
            "embedding": [",".join(map(str, row.tolist())) for row in struct_embeddings],
        }
    ).to_csv(export_csv_path, index=False)

    sampled_embeddings, sampled_labels, sampled_codes = sample_per_class(
        struct_embeddings, labels, codes, args.max_points_per_class
    )
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(sampled_labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = sampled_labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=12,
            alpha=0.7,
            color=cmap(idx),
            label=str(label),
        )
    plt.title(f"DiVeQ struct-token t-SNE ({dataset_name})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(title="Label", fontsize="small", loc="best")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{dataset_name}_struct_token_tsne.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved struct-token embeddings to: {export_csv_path}")
    print(f"Saved t-SNE plot to: {plot_path}")


if __name__ == "__main__":
    main()
