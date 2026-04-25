import argparse
import os
import re

import pandas as pd


SUMMARY_SUFFIX = "_summary.csv"


def infer_value_from_path(path, pattern, cast_func):
    match = re.search(pattern, path)
    if not match:
        return None
    try:
        return cast_func(match.group(1))
    except Exception:
        return None


def collect_summary_files(results_root):
    summary_files = []
    for root, _, files in os.walk(results_root):
        for file_name in files:
            if file_name.endswith(SUMMARY_SUFFIX):
                summary_files.append(os.path.join(root, file_name))
    summary_files.sort()
    return summary_files


def build_merged_rows(summary_files, results_root):
    rows = []

    for summary_path in summary_files:
        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        record = df.iloc[0].to_dict()
        normalized_path = summary_path.replace("\\", "/")
        relative_path = os.path.relpath(summary_path, results_root).replace("\\", "/")

        record["summary_path"] = summary_path
        record["relative_summary_path"] = relative_path
        record["analysis_group"] = relative_path.split("/")[0] if "/" in relative_path else ""

        inferred_dataset = infer_value_from_path(normalized_path, r"/([^/]+)_diveq_joint_summary\.csv$", str)
        if inferred_dataset and not record.get("dataset"):
            record["dataset"] = inferred_dataset

        codebook_size = infer_value_from_path(normalized_path, r"/codebook_(\d+)(?:/|$)", int)
        if codebook_size is not None:
            record["sweep_codebook_size"] = codebook_size

        lora_r = infer_value_from_path(normalized_path, r"/lora_r_(\d+)(?:/|$)", int)
        if lora_r is not None:
            record["sweep_lora_r"] = lora_r

        lr_value = infer_value_from_path(normalized_path, r"/lr_([0-9pmem]+)__aux_", str)
        if lr_value is not None:
            lr_text = lr_value.replace("p", ".").replace("m", "-")
            record["sweep_lr"] = lr_text

        aux_value = infer_value_from_path(normalized_path, r"__aux_([0-9p]+)(?:/|$)", str)
        if aux_value is not None:
            record["sweep_vq_aux_weight"] = aux_value.replace("p", ".")

        rows.append(record)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge parameter analysis summary CSV files.")
    parser.add_argument("results_root", help="root directory containing summary CSV files")
    parser.add_argument(
        "--output",
        default="",
        help="optional output CSV path; defaults to <results_root>/merged_summary.csv",
    )
    args = parser.parse_args()

    summary_files = collect_summary_files(args.results_root)
    if not summary_files:
        raise FileNotFoundError(f"No summary CSV files found under: {args.results_root}")

    merged_rows = build_merged_rows(summary_files, args.results_root)
    if not merged_rows:
        raise RuntimeError(f"No non-empty summary CSV files found under: {args.results_root}")

    merged_df = pd.DataFrame(merged_rows)
    sort_columns = [col for col in ["analysis_group", "dataset", "sweep_codebook_size", "sweep_lora_r", "sweep_lr", "sweep_vq_aux_weight"] if col in merged_df.columns]
    if sort_columns:
        merged_df = merged_df.sort_values(sort_columns).reset_index(drop=True)

    output_path = args.output or os.path.join(args.results_root, "merged_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Found {len(summary_files)} summary files.")
    print(f"Saved merged summary to: {output_path}")


if __name__ == "__main__":
    main()
