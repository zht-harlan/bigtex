import argparse
import csv
import json
import os

from offline_dataset_utils import (
    dataset_artifact_dir,
    load_dataset_with_texts,
    normalize_dataset_name,
    save_json,
)
from text_purifier import DEFAULT_PURIFICATION_PROMPT, build_text_purifier


def load_prompt_template(prompt_path):
    if not prompt_path:
        return DEFAULT_PURIFICATION_PROMPT
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def save_refined_outputs(dataset_dir, dataset_name, original_texts, refined_texts, mode, model_name):
    jsonl_path = os.path.join(dataset_dir, "refined_texts.jsonl")
    csv_path = os.path.join(dataset_dir, "refined_texts.csv")
    manifest_path = os.path.join(dataset_dir, "refined_texts_manifest.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for node_id, (original_text, refined_text) in enumerate(zip(original_texts, refined_texts)):
            row = {
                "node_id": node_id,
                "dataset": dataset_name,
                "original_text": original_text,
                "refined_text": refined_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["node_id", "dataset", "original_text", "refined_text"],
        )
        writer.writeheader()
        for node_id, (original_text, refined_text) in enumerate(zip(original_texts, refined_texts)):
            writer.writerow(
                {
                    "node_id": node_id,
                    "dataset": dataset_name,
                    "original_text": original_text,
                    "refined_text": refined_text,
                }
            )

    save_json(
        manifest_path,
        {
            "dataset": dataset_name,
            "num_nodes": len(refined_texts),
            "purifier_mode": mode,
            "purifier_model_name": model_name,
            "jsonl_path": jsonl_path,
            "csv_path": csv_path,
        },
    )

    return jsonl_path, csv_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Offline text purification for graph datasets")
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument(
        "--output_root",
        default="offline_artifacts",
        help="root directory for refined texts and embeddings",
    )
    parser.add_argument(
        "--mode",
        default="mock",
        choices=["mock", "local", "api"],
        help="text purification backend",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="local model name or api model identifier",
    )
    parser.add_argument(
        "--prompt_path",
        default="",
        help="optional prompt template file",
    )
    parser.add_argument("--api_url", default="", help="generic API endpoint for api mode")
    parser.add_argument("--api_key", default="", help="generic API key for api mode")
    args = parser.parse_args()

    dataset_name = normalize_dataset_name(args.dataset_name)
    _, _, original_texts = load_dataset_with_texts(dataset_name)
    dataset_dir = dataset_artifact_dir(args.output_root, dataset_name)

    prompt_template = load_prompt_template(args.prompt_path)
    purifier = build_text_purifier(
        mode=args.mode,
        prompt_template=prompt_template,
        model_name=args.model_name,
        api_url=args.api_url,
        api_key=args.api_key,
    )

    refined_texts = purifier.purify_texts(original_texts)
    jsonl_path, csv_path, manifest_path = save_refined_outputs(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        original_texts=original_texts,
        refined_texts=refined_texts,
        mode=args.mode,
        model_name=args.model_name,
    )

    print(f"Saved refined texts JSONL to: {jsonl_path}")
    print(f"Saved refined texts CSV to: {csv_path}")
    print(f"Saved refined text manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
