import argparse
import json
import os

import numpy as np
import torch

from offline_dataset_utils import dataset_artifact_dir, normalize_dataset_name, save_json
from offline_text_encoder import OfflineTextEncoder


def load_refined_texts(refined_texts_path):
    refined_texts = []
    node_ids = []

    with open(refined_texts_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            node_ids.append(int(row["node_id"]))
            refined_texts.append(row["refined_text"])

    return node_ids, refined_texts


def save_embedding_artifacts(
    dataset_dir,
    dataset_name,
    encoder_name,
    pooling,
    node_ids,
    embeddings,
):
    pt_path = os.path.join(dataset_dir, "text_embeddings.pt")
    npy_path = os.path.join(dataset_dir, "text_embeddings.npy")
    node_ids_path = os.path.join(dataset_dir, "node_ids.pt")
    mapping_path = os.path.join(dataset_dir, "node_id_to_row.json")
    manifest_path = os.path.join(dataset_dir, "embedding_manifest.json")

    torch.save(embeddings, pt_path)
    np.save(npy_path, embeddings.numpy())
    torch.save(torch.tensor(node_ids, dtype=torch.long), node_ids_path)

    mapping = {str(node_id): row_idx for row_idx, node_id in enumerate(node_ids)}
    save_json(mapping_path, mapping)
    save_json(
        manifest_path,
        {
            "dataset": dataset_name,
            "encoder_name": encoder_name,
            "pooling": pooling,
            "num_nodes": len(node_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "pt_path": pt_path,
            "npy_path": npy_path,
            "node_ids_path": node_ids_path,
            "mapping_path": mapping_path,
        },
    )

    return pt_path, npy_path, node_ids_path, mapping_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Offline text embedding extraction")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument(
        "--input_root",
        default="offline_artifacts",
        help="root directory containing refined texts",
    )
    parser.add_argument(
        "--refined_texts_path",
        default="",
        help="optional direct path to refined_texts.jsonl",
    )
    parser.add_argument(
        "--encoder_name",
        default="scibert",
        help="encoder backbone, e.g. roberta, bert, scibert, microsoft/deberta-v3-base",
    )
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"], help="pooling")
    parser.add_argument("--batch_size", default=32, type=int, help="encoding batch size")
    parser.add_argument("--max_length", default=256, type=int, help="max sequence length")
    parser.add_argument("--normalize", action="store_true", help="l2 normalize embeddings")
    args = parser.parse_args()

    dataset_name = normalize_dataset_name(args.dataset_name)
    dataset_dir = dataset_artifact_dir(args.input_root, dataset_name)
    refined_texts_path = args.refined_texts_path or os.path.join(dataset_dir, "refined_texts.jsonl")
    if not os.path.exists(refined_texts_path):
        raise FileNotFoundError(
            f"Refined texts not found: {refined_texts_path}. Run preprocess_refined_texts.py first."
        )

    node_ids, refined_texts = load_refined_texts(refined_texts_path)
    encoder = OfflineTextEncoder(
        model_name=args.encoder_name,
        pooling=args.pooling,
        max_length=args.max_length,
        normalize=args.normalize,
    )
    embeddings = encoder.encode_texts(refined_texts, batch_size=args.batch_size)

    pt_path, npy_path, node_ids_path, mapping_path, manifest_path = save_embedding_artifacts(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        encoder_name=encoder.model_name,
        pooling=args.pooling,
        node_ids=node_ids,
        embeddings=embeddings,
    )

    print(f"Saved embedding tensor to: {pt_path}")
    print(f"Saved embedding numpy array to: {npy_path}")
    print(f"Saved node ids to: {node_ids_path}")
    print(f"Saved node id mapping to: {mapping_path}")
    print(f"Saved embedding manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
