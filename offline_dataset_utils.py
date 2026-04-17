import json
import os

import torch

from load_data import (
    load_arxiv,
    load_arxiv_2023,
    load_cstag_csv_dataset,
    load_citeseer,
    load_cora,
    load_ogb_products,
    load_photo,
    load_products_subset,
    load_pubmed,
)


DATASET_ALIASES = {
    "ogbn-arxiv": "arxiv",
    "ogbn_arxiv": "arxiv",
    "amazon-photo": "photo",
    "amazon_photo": "photo",
    "amazonphoto": "photo",
    "children": "children",
    "history": "history",
}


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name.lower(), dataset_name.lower())


def load_dataset_with_texts(dataset_name, data_root="datasets"):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name in ["children", "history"]:
        data, texts = load_cstag_csv_dataset(dataset_name=dataset_name, data_root=data_root)
    elif dataset_name == "products-subset":
        data, texts = load_products_subset()
    elif dataset_name == "products":
        data, texts = load_ogb_products()
    elif dataset_name == "citeseer":
        data, texts = load_citeseer()
    elif dataset_name == "photo":
        photo_dir = os.path.join(data_root, "Photo")
        photo_pt = os.path.join(photo_dir, "Photo.csv")
        if os.path.exists(photo_pt):
            data, texts = load_cstag_csv_dataset(dataset_name=dataset_name, data_root=data_root)
        else:
            data, texts = load_photo()
    elif dataset_name in ["arxiv", "arxiv_sim"]:
        data, texts = load_arxiv(dataset_name=dataset_name)
    elif dataset_name == "cora":
        data, texts = load_cora()
    elif dataset_name == "pubmed":
        data, texts = load_pubmed()
    elif dataset_name == "arxiv_2023":
        data, texts = load_arxiv_2023()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_name, data, texts


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dataset_artifact_dir(output_root, dataset_name):
    normalized_name = normalize_dataset_name(dataset_name)
    path = os.path.join(output_root, normalized_name)
    ensure_dir(path)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_cached_text_embeddings(dataset_name, artifact_root="offline_artifacts", filename="text_embeddings.pt"):
    dataset_dir = dataset_artifact_dir(artifact_root, dataset_name)
    embedding_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"Cached text embeddings not found: {embedding_path}. "
            "Run precompute_text_embeddings.py first."
        )

    embeddings = torch.load(embedding_path, map_location="cpu")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    return embeddings.float(), embedding_path


def load_refined_texts(dataset_name, artifact_root="offline_artifacts", filename="refined_texts.jsonl"):
    dataset_dir = dataset_artifact_dir(artifact_root, dataset_name)
    refined_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(refined_path):
        raise FileNotFoundError(
            f"Refined texts not found: {refined_path}. "
            "Run preprocess_refined_texts.py first."
        )

    refined_texts = []
    with open(refined_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            refined_texts.append(row["refined_text"])
    return refined_texts, refined_path
