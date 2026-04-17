import argparse
import os
import subprocess
import sys

from offline_dataset_utils import normalize_dataset_name


def main():
    parser = argparse.ArgumentParser(description="Run offline text purification and embedding extraction")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument("--output_root", default="offline_artifacts", help="artifact root")
    parser.add_argument(
        "--purifier_mode",
        default="mock",
        choices=["mock", "local", "api"],
        help="purification backend",
    )
    parser.add_argument(
        "--purifier_model_name",
        default="",
        help="local purification model name or API model id",
    )
    parser.add_argument("--prompt_path", default="", help="optional prompt template path")
    parser.add_argument("--api_url", default="", help="generic API endpoint for api mode")
    parser.add_argument("--api_key", default="", help="generic API key for api mode")
    parser.add_argument("--data_root", default="datasets", help="root directory containing local dataset folders")
    parser.add_argument("--encoder_name", default="scibert", help="encoder backbone")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"], help="pooling")
    parser.add_argument("--batch_size", default=32, type=int, help="encoding batch size")
    parser.add_argument("--max_length", default=256, type=int, help="max text length")
    parser.add_argument("--normalize", action="store_true", help="l2 normalize embeddings")
    args = parser.parse_args()

    preprocess_cmd = [
        sys.executable,
        "preprocess_refined_texts.py",
        args.dataset_name,
        "--output_root",
        args.output_root,
        "--data_root",
        args.data_root,
        "--mode",
        args.purifier_mode,
    ]
    if args.purifier_model_name:
        preprocess_cmd.extend(["--model_name", args.purifier_model_name])
    if args.prompt_path:
        preprocess_cmd.extend(["--prompt_path", args.prompt_path])
    if args.api_url:
        preprocess_cmd.extend(["--api_url", args.api_url])
    if args.api_key:
        preprocess_cmd.extend(["--api_key", args.api_key])

    embed_cmd = [
        sys.executable,
        "precompute_text_embeddings.py",
        args.dataset_name,
        "--input_root",
        args.output_root,
        "--encoder_name",
        args.encoder_name,
        "--pooling",
        args.pooling,
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
    ]
    if args.normalize:
        embed_cmd.append("--normalize")

    print("Running text purification...")
    subprocess.run(preprocess_cmd, check=True)
    print("Running offline text encoding...")
    subprocess.run(embed_cmd, check=True)

    dataset_dir = os.path.join(args.output_root, normalize_dataset_name(args.dataset_name))
    print(f"Offline artifacts ready under: {dataset_dir}")


if __name__ == "__main__":
    main()
