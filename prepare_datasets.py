import argparse

from load_data import load_arxiv, load_cora, load_photo, load_pubmed


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for BiGTex")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arxiv", "cora", "pubmed", "photo"],
        help="datasets to prepare",
    )
    args = parser.parse_args()

    loaders = {
        "arxiv": lambda: load_arxiv("arxiv"),
        "cora": load_cora,
        "pubmed": load_pubmed,
        "photo": load_photo,
    }

    for dataset in args.datasets:
        if dataset not in loaders:
            raise ValueError(f"Unsupported dataset: {dataset}")
        print(f"Preparing dataset: {dataset}")
        data, texts = loaders[dataset]()
        print(f"Prepared {dataset}: num_nodes={data.num_nodes}, num_texts={len(texts)}")


if __name__ == "__main__":
    main()
