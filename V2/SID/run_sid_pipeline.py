import argparse
import os
import subprocess
import sys


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SID model and export SID codes")
    parser.add_argument("--data_path", required=True, help="Path to poi_Emb_dict.pkl")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint output root directory")
    parser.add_argument("--output_csv", required=True, help="Exported SID csv path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda device")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--learner", default="AdamW")
    parser.add_argument("--lr_scheduler_type", default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--bn", type=str2bool, default=True)
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--kmeans_init", type=str2bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--use_sk", type=str2bool, default=False)
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.1, 0.1, 0.1])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--use-linear", type=int, default=1)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--e_dim", type=int, default=64)
    parser.add_argument("--quant_loss_weight", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--checkpoint_name", default="best_loss_model.pth")
    return parser.parse_args()


def build_common_args(args):
    common_args = [
        "--data_path", args.data_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--dropout_prob", str(args.dropout_prob),
        "--bn", str(args.bn),
        "--loss_type", args.loss_type,
        "--kmeans_init", str(args.kmeans_init),
        "--kmeans_iters", str(args.kmeans_iters),
        "--use_sk", str(args.use_sk),
        "--sk_iters", str(args.sk_iters),
        "--use-linear", str(args.use_linear),
        "--device", args.device,
        "--e_dim", str(args.e_dim),
        "--quant_loss_weight", str(args.quant_loss_weight),
        "--beta", str(args.beta),
    ]
    common_args.extend(["--sk_epsilons", *[str(x) for x in args.sk_epsilons]])
    common_args.extend(["--num_emb_list", *[str(x) for x in args.num_emb_list]])
    common_args.extend(["--layers", *[str(x) for x in args.layers]])
    return common_args


def build_train_only_args(args):
    return ["--save_limit", str(args.save_limit)]


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    train_cmd = [
        python_exe,
        os.path.join(script_dir, "train_SID.py"),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--eval_step", str(args.eval_step),
        "--learner", args.learner,
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--warmup_epochs", str(args.warmup_epochs),
        "--weight_decay", str(args.weight_decay),
        "--ckpt_dir", args.ckpt_dir,
    ]
    train_cmd.extend(build_common_args(args))
    train_cmd.extend(build_train_only_args(args))

    export_cmd = [
        python_exe,
        os.path.join(script_dir, "get_SID.py"),
        "--ckpt_dir", args.ckpt_dir,
        "--output_csv", args.output_csv,
        "--checkpoint_name", args.checkpoint_name,
    ]
    export_cmd.extend(build_common_args(args))

    print("Running training command:")
    print(" ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    print("Running export command:")
    print(" ".join(export_cmd))
    subprocess.run(export_cmd, check=True)


if __name__ == "__main__":
    main()
