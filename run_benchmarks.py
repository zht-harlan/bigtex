import argparse
import os
import subprocess
import sys

import pandas as pd


DATASET_ALIASES = {
    "ogbn-arxiv": "arxiv",
    "ogbn_arxiv": "arxiv",
    "amazon-photo": "photo",
    "amazon_photo": "photo",
    "amazonphoto": "photo",
}


def normalize_dataset_name(dataset_name):
    lowered = dataset_name.lower()
    return DATASET_ALIASES.get(lowered, lowered)


def main():
    parser = argparse.ArgumentParser(description="Run BiGTex benchmarks on multiple datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ogbn-arxiv", "cora", "pubmed", "amazon-photo"],
        help="datasets to run",
    )
    parser.add_argument("--model_name", default="BiGTex", help="model name passed to main.py")
    parser.add_argument("--num_iterate", default=5, type=int, help="number of runs per dataset")
    parser.add_argument("--epochs", default=30, type=int, help="epochs for main training")
    parser.add_argument(
        "--results_root",
        default="benchmark_results",
        help="root directory for per-dataset and merged csv files",
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--finetune_epochs", default=10, type=int, help="PLM finetune epochs")
    parser.add_argument(
        "--finetune_batch_size", default=16, type=int, help="PLM finetune batch size"
    )
    parser.add_argument("--language_model_name", default="SCIBERT", help="language model")
    parser.add_argument("--mode", default="MLP", help="fusion mode")
    parser.add_argument("--GNN", default="sage", help="gnn backbone")
    parser.add_argument("--seed_base", default=42, type=int, help="base random seed")
    parser.add_argument("--run_cs", action="store_true", help="run Correct&Smooth")
    parser.add_argument("--save_embeddings", action="store_true", help="save embeddings")
    args = parser.parse_args()

    os.makedirs(args.results_root, exist_ok=True)
    summary_frames = []

    for dataset in args.datasets:
        normalized_dataset = normalize_dataset_name(dataset)
        dataset_dir = os.path.join(args.results_root, normalized_dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "main.py",
            normalized_dataset,
            args.model_name,
            "--num_iterate",
            str(args.num_iterate),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--finetune_epochs",
            str(args.finetune_epochs),
            "--finetune_batch_size",
            str(args.finetune_batch_size),
            "--language_model_name",
            args.language_model_name,
            "--mode",
            args.mode,
            "--GNN",
            args.GNN,
            "--seed_base",
            str(args.seed_base),
            "--results_dir",
            dataset_dir,
        ]
        if args.run_cs:
            cmd.append("--run_cs")
        if args.save_embeddings:
            cmd.append("--save_embeddings")

        print(f"Running dataset: {dataset} -> {normalized_dataset}")
        subprocess.run(cmd, check=True)

        summary_path = os.path.join(dataset_dir, f"{normalized_dataset}_summary.csv")
        summary_frames.append(pd.read_csv(summary_path))

    merged_summary = pd.concat(summary_frames, ignore_index=True)
    merged_summary_path = os.path.join(args.results_root, "benchmark_summary.csv")
    merged_summary.to_csv(merged_summary_path, index=False)
    print(f"Merged summary saved to: {merged_summary_path}")


if __name__ == "__main__":
    main()
