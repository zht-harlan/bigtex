import json
import os
import urllib.request
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import degree, to_edge_index

try:
    from torch.serialization import add_safe_globals
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except Exception:
    pass


ARXIV_TEXT_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
PUBMED_GDRIVE_FILE_ID = "1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W"
CSTAG_DIR_NAMES = {
    "children": "Children",
    "history": "History",
    "photo": "Photo",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _candidate_cstag_csv_paths(data_root, dataset_name):
    canonical_name = CSTAG_DIR_NAMES[dataset_name]
    lower_name = dataset_name.lower()
    title_name = dataset_name.title()
    upper_name = dataset_name.upper()

    directory_names = []
    for name in [canonical_name, lower_name, title_name, upper_name]:
        if name not in directory_names:
            directory_names.append(name)

    file_names = []
    for name in [
        f"{canonical_name}.csv",
        f"{lower_name}.csv",
        f"{title_name}.csv",
        f"{upper_name}.csv",
        "data.csv",
    ]:
        if name not in file_names:
            file_names.append(name)

    root_prefixes = [data_root, os.path.join(data_root, "CSTAG")]

    candidate_paths = []
    for root_prefix in root_prefixes:
        for directory_name in directory_names:
            for file_name in file_names:
                candidate_paths.append(os.path.join(root_prefix, directory_name, file_name))
            candidate_paths.append(os.path.join(root_prefix, directory_name))

        for file_name in file_names:
            candidate_paths.append(os.path.join(root_prefix, file_name))

    deduped_paths = []
    for path in candidate_paths:
        if path not in deduped_paths:
            deduped_paths.append(path)
    return deduped_paths


def resolve_cstag_csv_path(data_root, dataset_name):
    candidate_paths = _candidate_cstag_csv_paths(data_root, dataset_name)

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    missing_preview = "\n".join(candidate_paths[:8])
    raise FileNotFoundError(
        f"CSTAG CSV not found for dataset '{dataset_name}' under data_root '{data_root}'. "
        f"Tried paths including:\n{missing_preview}"
    )


def download_with_urllib(url, output_path):
    ensure_dir(os.path.dirname(output_path))
    print(f"Downloading {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)


def download_from_gdrive(file_id, output_path):
    try:
        import gdown
    except ImportError:
        print("gdown is not installed; skip Google Drive download.")
        return False

    ensure_dir(os.path.dirname(output_path))
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
    return os.path.exists(output_path)


def ensure_arxiv_text_file():
    output_path = os.path.join("datasets", "arxiv", "titleabs.tsv.gz")
    if not os.path.exists(output_path):
        download_with_urllib(ARXIV_TEXT_URL, output_path)
    return output_path


def ensure_pubmed_json():
    candidate_paths = [
        os.path.join("datasets", "PubMed", "pubmed.json"),
        os.path.join("datasets", "pubmed", "pubmed.json"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    output_path = candidate_paths[1]
    success = download_from_gdrive(PUBMED_GDRIVE_FILE_ID, output_path)
    return output_path if success else None


def load_json_with_fallback_encodings(path, encodings=None):
    encodings = encodings or ["utf-8", "utf-8-sig", "gbk", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Failed to decode JSON file: {path}. Last error: {last_error}")


def build_feature_texts(x, prefix, topk=32):
    x_cpu = x.detach().cpu()
    texts = []
    for idx, row in enumerate(x_cpu):
        row = row.flatten()
        nonzero_idx = torch.nonzero(row, as_tuple=False).view(-1)
        if nonzero_idx.numel() == 0:
            top_idx = torch.topk(row.abs(), k=min(topk, row.numel())).indices.tolist()
        else:
            top_idx = nonzero_idx[:topk].tolist()
        tokens = [f"feature_{i}" for i in top_idx]
        if not tokens:
            tokens = ["no_feature"]
        texts.append(f"[sep] {prefix} node {idx} [sep] " + " ".join(tokens))
    return texts


def _make_random_split_data(x, edge_index, y, seed=42):
    num_nodes = y.numel()
    rng = np.random.default_rng(seed)
    node_id = np.arange(num_nodes)
    rng.shuffle(node_id)

    data = Data(
        n_id=torch.arange(num_nodes),
        x=x.contiguous(),
        edge_index=edge_index.contiguous(),
        y=y.contiguous(),
    )
    data.train_idx = torch.from_numpy(np.sort(node_id[: int(num_nodes * 0.6)])).long().contiguous()
    data.valid_idx = torch.from_numpy(
        np.sort(node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)])
    ).long().contiguous()
    data.test_idx = torch.from_numpy(np.sort(node_id[int(num_nodes * 0.8) :])).long().contiguous()
    return data


def _build_edge_index_from_neighbors(neighbor_series):
    row = []
    col = []

    for source_id, raw_neighbors in enumerate(neighbor_series.tolist()):
        if pd.isna(raw_neighbors):
            continue
        neighbors = literal_eval(raw_neighbors) if isinstance(raw_neighbors, str) else raw_neighbors
        for target_id in neighbors:
            row.append(source_id)
            col.append(int(target_id))

    if not row:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index.contiguous()


def load_cstag_csv_dataset(dataset_name, data_root="datasets", seed=42):
    normalized_name = dataset_name.lower()
    if normalized_name not in CSTAG_DIR_NAMES:
        raise ValueError(f"Unsupported CSTAG dataset: {dataset_name}")

    csv_path = resolve_cstag_csv_path(data_root, normalized_name)
    dataset_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    required_columns = {"text", "label", "node_id", "neighbour"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"CSTAG CSV missing columns {sorted(missing_columns)}: {csv_path}")

    df = df.sort_values("node_id").reset_index(drop=True)
    expected_node_ids = np.arange(len(df))
    actual_node_ids = df["node_id"].to_numpy()
    if not np.array_equal(actual_node_ids, expected_node_ids):
        raise ValueError(
            f"CSTAG node_id column must be contiguous from 0..N-1 in {csv_path}; "
            f"got first values {actual_node_ids[:5].tolist()}"
        )

    edge_index = _build_edge_index_from_neighbors(df["neighbour"])
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    x = torch.ones((len(df), 1), dtype=torch.float32).contiguous()
    data = _make_random_split_data(x=x, edge_index=edge_index, y=y, seed=seed)

    texts = [f"[sep] {text}" for text in df["text"].fillna("").astype(str).tolist()]
    degrees = degree(edge_index.reshape(-1), num_nodes=data.num_nodes) if edge_index.numel() else torch.zeros(data.num_nodes)
    print(f"Loaded CSTAG CSV dataset from: {csv_path}")
    print(f"Average node degree (directed edge list): {degrees.mean().item():.4f}")
    return data, texts


def load_cora():
    planetoid = Planetoid("dataset", "cora", transform=T.NormalizeFeatures())
    base_data = planetoid[0]

    local_path = os.path.join("datasets", "cora", "cora.pt")
    if os.path.exists(local_path):
        cora_data = torch.load(local_path, weights_only=False)
        texts = [
            f"[sep] {text.split(':', 1)[0].strip()} [sep] {text.split(':', 1)[1].strip()}"
            if ":" in text
            else f"[sep] {text}"
            for text in cora_data.raw_texts
        ]
    else:
        print("datasets/cora/cora.pt not found; using synthetic texts from node features.")
        texts = build_feature_texts(base_data.x, "cora")

    data = Data(
        n_id=torch.arange(base_data.x.shape[0]),
        x=planetoid.x,
        edge_index=base_data.edge_index,
        y=planetoid.y,
    )

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    data.train_idx = np.sort(node_id[: int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8) :])

    degrees = degree(data.edge_index.reshape(-1), num_nodes=data.num_nodes)
    print(f"Average node degree (undirected): {degrees.mean().item():.4f}")
    return data, texts


def load_arxiv(dataset_name):
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    base_data = dataset[0]
    idx_splits = dataset.get_idx_split()

    x = (
        torch.load("datasets/arxiv_sim/x_embs.pt")
        if dataset_name == "arxiv_sim"
        else base_data.x
    )

    data = Data(
        n_id=torch.arange(base_data.num_nodes),
        x=x,
        edge_index=base_data.edge_index,
        y=dataset.y,
        train_idx=idx_splits["train"],
        valid_idx=idx_splits["valid"],
        test_idx=idx_splits["test"],
    )

    nodeidx2paperid = pd.read_csv("datasets/arxiv/nodeidx2paperid.csv.gz", compression="gzip")
    raw_text_path = ensure_arxiv_text_file()
    raw_text = pd.read_csv(
        raw_text_path,
        sep="\t",
        header=None,
        names=["paper id", "title", "abs"],
    )

    nodeidx2paperid["paper id"] = nodeidx2paperid["paper id"].astype(str)
    raw_text["paper id"] = raw_text["paper id"].astype(str)
    df = pd.merge(nodeidx2paperid, raw_text, on="paper id")
    texts = [f"[sep] {ti} [sep] {ab}" for ti, ab in zip(df["title"], df["abs"])]
    return data, texts


def load_ogb_products():
    dataset = PygNodePropPredDataset(name="ogbn-products")
    data = dataset[0]
    idx_splits = dataset.get_idx_split()

    data = Data(
        n_id=torch.arange(data.num_nodes),
        x=data.x,
        edge_index=data.edge_index,
        y=dataset.y,
        train_idx=idx_splits["train"],
        valid_idx=idx_splits["valid"],
        test_idx=idx_splits["test"],
    )

    data_root = "datasets/products"
    raw_text_path = "datasets/products/products_text"

    if not os.path.exists(f"{data_root}/product3.csv"):
        i = 1
        for root, _, files in os.walk(os.path.join(raw_text_path, "")):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8-sig") as file_in:
                    title = []
                    for line in file_in.readlines():
                        dic = json.loads(line)
                        dic["title"] = dic["title"].strip("\"\n")
                        title.append(dic)
                    writercsv = pd.DataFrame(columns=["uid", "title", "content"], data=title)
                    writercsv.to_csv(
                        os.path.join(data_root, f"product{i}.csv"),
                        index=False,
                        encoding="utf_8_sig",
                    )
                    i += 1

        pro1 = pd.read_csv(data_root + "/product1.csv")
        pro2 = pd.read_csv(data_root + "/product2.csv")
        file = pd.concat([pro1, pro2])
        file.drop_duplicates()
        file.to_csv(os.path.join(data_root, "product3.csv"), index=False, sep=" ")
    else:
        file = pd.read_csv(data_root + "/product3.csv", sep=" ")

    category_path_csv = r"dataset\ogbn_products\mapping/labelidx2productcategory.csv.gz"
    products_asin_path_csv = r"dataset\ogbn_products\mapping/nodeidx2asin.csv.gz"
    products_ids = pd.read_csv(products_asin_path_csv)
    categories = pd.read_csv(category_path_csv)

    products_ids.columns = ["ID", "asin"]
    categories.columns = ["label_idx", "category"]
    file.columns = ["asin", "title", "content"]
    products_ids["label_idx"] = data.y
    data1 = pd.merge(products_ids, file, how="left", on="asin")
    data1 = pd.merge(data1, categories, how="left", on="label_idx")
    texts = ("[sep] " + data1["title"].fillna("") + "[sep] " + data1["content"].fillna("")).tolist()
    return data, texts


def load_products_subset(seed=42):
    raw_data = torch.load("datasets/products/ogbn-products_subset.pt", weights_only=False)
    node_desc = pd.read_csv("datasets/products/ogbn-products_subset.csv")
    num_nodes = raw_data.num_nodes

    texts = []
    for i in range(num_nodes):
        node_title = node_desc.iloc[i, 2] if pd.notna(node_desc.iloc[i, 2]) else "missing"
        node_content = node_desc.iloc[i, 3] if pd.notna(node_desc.iloc[i, 3]) else "missing"
        texts.append("[sep] " + str(node_title) + " [sep] " + str(node_content))

    edge_index = raw_data.adj_t.to_symmetric()
    edge_index = to_edge_index(edge_index)[0]

    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)

    data = Data(
        n_id=torch.arange(num_nodes),
        x=raw_data.x,
        edge_index=edge_index,
        y=raw_data.y,
        train_idx=perm[:n_train],
        valid_idx=perm[n_train : n_train + n_val],
        test_idx=perm[n_train + n_val :],
    )
    return data, texts


def load_pubmed():
    planetoid = Planetoid("dataset", "PubMed", transform=T.NormalizeFeatures())
    base_data = planetoid[0]
    data = Data(
        n_id=torch.arange(base_data.x.shape[0]),
        x=planetoid.x,
        edge_index=base_data.edge_index,
        y=planetoid.y,
    )

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    data.train_idx = np.sort(node_id[: int(data.num_nodes * 0.6)])
    data.valid_idx = np.sort(node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(data.num_nodes * 0.8) :])

    pubmed_json_path = ensure_pubmed_json()
    if pubmed_json_path and os.path.exists(pubmed_json_path):
        try:
            pubmed = load_json_with_fallback_encodings(pubmed_json_path)
            df_pubmed = pd.DataFrame.from_dict(pubmed)
            ab = df_pubmed["AB"].fillna("")
            ti = df_pubmed["TI"].fillna("")
            texts = [f"[sep] {title} [sep] {abstract}" for title, abstract in zip(ti, ab)]
        except (ValueError, KeyError, TypeError) as exc:
            print(
                f"Failed to parse pubmed.json at {pubmed_json_path}; "
                f"falling back to synthetic texts. Error: {exc}"
            )
            texts = build_feature_texts(base_data.x, "pubmed")
    else:
        print("pubmed.json not available; using synthetic texts from node features.")
        texts = build_feature_texts(base_data.x, "pubmed")

    return data, texts


def load_arxiv_2023():
    data = torch.load("datasets/arxiv_2023/graph.pt")
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
    )
    data.train_idx = np.sort(node_id[: int(num_nodes * 0.6)])
    data.valid_idx = np.sort(node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8) :])

    degrees = degree(data.edge_index.reshape(-1), num_nodes=num_nodes)
    print(f"Average node degree (undirected): {degrees.mean().item():.4f}")

    df = pd.read_csv("datasets/arxiv_2023/paper_info.csv")
    texts = [f"[sep] {ti} [sep] {ab}" for ti, ab in zip(df["title"], df["abstract"])]
    return data, texts


def load_photo():
    local_path = os.path.join("datasets", "photo", "photo.pt")
    if os.path.exists(local_path):
        data = torch.load(local_path, weights_only=False)
        data.y = data.label
        data.x = data.x.float()
        texts = [f"[sep] {ti}" for ti in data.raw_texts]
        edge_index = data.edge_index
    else:
        print("datasets/photo/photo.pt not found; downloading official PyG Amazon Photo and using synthetic texts.")
        dataset = Amazon(root="dataset", name="Photo", transform=T.NormalizeFeatures())
        pyg_data = dataset[0]
        data = Data(
            n_id=torch.arange(pyg_data.x.shape[0]),
            x=pyg_data.x.float(),
            edge_index=pyg_data.edge_index,
            y=pyg_data.y,
        )
        texts = build_feature_texts(data.x, "amazon_photo")
        edge_index = data.edge_index

    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,
        edge_index=edge_index,
        y=data.y,
    )
    data.train_idx = np.sort(node_id[: int(num_nodes * 0.6)])
    data.valid_idx = np.sort(node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8) :])

    degrees = degree(data.edge_index.reshape(-1), num_nodes=num_nodes)
    print(f"Average node degree (undirected): {degrees.mean().item():.4f}")
    return data, texts


def load_citeseer():
    data = torch.load("datasets/citeseer/citeseer_random_sbert.pt", weights_only=False)
    data.x = data.x.float()
    texts = [f"[sep] {ti}" for ti in data.raw_texts]

    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)
    data = Data(
        n_id=torch.arange(data.x.shape[0]),
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
    )
    data.train_idx = np.sort(node_id[: int(num_nodes * 0.6)])
    data.valid_idx = np.sort(node_id[int(num_nodes * 0.6) : int(num_nodes * 0.8)])
    data.test_idx = np.sort(node_id[int(num_nodes * 0.8) :])

    degrees = degree(data.edge_index.reshape(-1), num_nodes=num_nodes)
    print(f"Average node degree (undirected): {degrees.mean().item():.4f}")
    return data, texts
