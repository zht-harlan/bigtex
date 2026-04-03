import argparse
import os
import subprocess
import sys

from offline_dataset_utils import normalize_dataset_name


def run_stage(cmd, stage_name):
    print(f"\n{'=' * 90}")
    print(f"Running {stage_name}")
    print(f"{'=' * 90}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_offline_artifacts_cmd(args, dataset_name):
    cmd = [
        sys.executable,
        "build_offline_text_artifacts.py",
        dataset_name,
        "--output_root",
        args.artifact_root,
        "--purifier_mode",
        args.purifier_mode,
        "--encoder_name",
        args.backbone_name,
        "--pooling",
        args.pooling,
        "--batch_size",
        str(args.text_batch_size),
        "--max_length",
        str(args.text_max_length),
    ]
    if args.normalize_embeddings:
        cmd.append("--normalize")
    if args.purifier_model_name:
        cmd.extend(["--purifier_model_name", args.purifier_model_name])
    if args.prompt_path:
        cmd.extend(["--prompt_path", args.prompt_path])
    if args.api_url:
        cmd.extend(["--api_url", args.api_url])
    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    return cmd


def build_joint_training_cmd(args, dataset_name):
    dataset_results_dir = os.path.join(args.results_dir, dataset_name)
    node_codes_path = os.path.join(dataset_results_dir, f"{dataset_name}_node_codes.csv")
    cmd = [
        sys.executable,
        "train_joint_graph_text_classifier.py",
        dataset_name,
        "--artifact_root",
        args.artifact_root,
        "--results_dir",
        args.results_dir,
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_layers",
        str(args.num_layers),
        "--gnn_type",
        args.gnn_type,
        "--graph_dropout",
        str(args.graph_dropout),
        "--batch_size",
        str(args.joint_batch_size),
        "--epochs",
        str(args.joint_epochs),
        "--lr",
        str(args.joint_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--num_runs",
        str(args.num_runs),
        "--seed_base",
        str(args.seed_base),
        "--codebook_size",
        str(args.codebook_size),
        "--num_quantizers",
        str(args.num_quantizers),
        "--backbone_name",
        args.backbone_name,
        "--max_text_length",
        str(args.max_text_length),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--node_codes_path",
        node_codes_path,
    ]
    if args.quantizer_dim > 0:
        cmd.extend(["--quantizer_dim", str(args.quantizer_dim)])
    if args.disable_lora:
        cmd.append("--disable_lora")
    if args.freeze_plm_embeddings:
        cmd.append("--freeze_plm_embeddings")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run the full 5-stage BiGTex pipeline end-to-end."
    )
    parser.add_argument("dataset_name", help="dataset name")
    parser.add_argument(
        "--artifact_root", default="offline_artifacts", help="offline artifact root"
    )
    parser.add_argument(
        "--results_dir", default="joint_fusion_results", help="joint stage-3/4/5 results root"
    )

    parser.add_argument(
        "--purifier_mode",
        default="mock",
        choices=["mock", "local", "api"],
        help="text purification backend",
    )
    parser.add_argument(
        "--purifier_model_name",
        default="",
        help="local purifier model name or api model id",
    )
    parser.add_argument("--prompt_path", default="", help="optional prompt template path")
    parser.add_argument("--api_url", default="", help="generic API endpoint")
    parser.add_argument("--api_key", default="", help="generic API key")

    parser.add_argument("--backbone_name", default="scibert", help="unified PLM backbone")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"], help="embedding pooling")
    parser.add_argument(
        "--text_batch_size", default=32, type=int, help="offline text encoding batch size"
    )
    parser.add_argument(
        "--text_max_length", default=256, type=int, help="offline text encoding max length"
    )
    parser.add_argument(
        "--normalize_embeddings", action="store_true", help="l2 normalize offline embeddings"
    )

    parser.add_argument("--hidden_dim", default=256, type=int, help="Ze hidden dimension")
    parser.add_argument("--num_layers", default=2, type=int, help="number of GNN layers")
    parser.add_argument(
        "--gnn_type", default="sage", choices=["sage", "gcn", "gat"], help="graph backbone"
    )
    parser.add_argument("--graph_dropout", default=0.2, type=float, help="graph dropout")
    parser.add_argument("--joint_batch_size", default=64, type=int, help="joint stage-3/4/5 batch size")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--num_runs", default=1, type=int, help="repeated joint runs")
    parser.add_argument("--seed_base", default=42, type=int, help="base random seed")
    parser.add_argument("--joint_epochs", default=20, type=int, help="joint training epochs")
    parser.add_argument("--joint_lr", default=2e-4, type=float, help="joint training learning rate")
    parser.add_argument("--codebook_size", default=128, type=int, help="RQ-VAE codebook size")
    parser.add_argument("--num_quantizers", default=3, type=int, help="number of residual quantizers")
    parser.add_argument("--quantizer_dim", default=0, type=int, help="quantizer dim, 0 means hidden_dim")
    parser.add_argument("--max_text_length", default=256, type=int, help="joint refined text max length")
    parser.add_argument("--disable_lora", action="store_true", help="disable LoRA in stage 5")
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")
    parser.add_argument(
        "--freeze_plm_embeddings",
        action="store_true",
        help="freeze token embedding matrix in stage 5",
    )

    args = parser.parse_args()
    dataset_name = normalize_dataset_name(args.dataset_name)

    run_stage(
        build_offline_artifacts_cmd(args, dataset_name),
        "Stage 1-2: text purification and frozen PLM embedding extraction",
    )
    run_stage(
        build_joint_training_cmd(args, dataset_name),
        "Stage 3-5: joint MLP + GNN + RQ + PLM(LoRA) optimization",
    )

    final_summary = os.path.join(
        args.results_dir, dataset_name, f"{dataset_name}_joint_summary.csv"
    )
    final_checkpoint = os.path.join(
        args.results_dir, dataset_name, f"{dataset_name}_joint_model.pt"
    )

    print(f"\n{'=' * 90}")
    print("Full pipeline completed")
    print(f"{'=' * 90}")
    print(f"Final summary: {final_summary}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
