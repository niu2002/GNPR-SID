import argparse
import csv
import datetime
import logging
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CRQVAE.crqvae import CRQVAE
from POIdatasets import EmbDataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except ValueError:
        index = len(color_set) - 1
    prev_log = "\033["
    prev_log += "1;3" if highlight else "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def get_local_time():
    cur = datetime.datetime.now()
    return cur.strftime("%b-%d-%Y_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Export SID codes from a trained CRQVAE checkpoint")

    parser.add_argument("--data_path", type=str, required=True, help="Path to POI embedding dict (.pkl)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint root or timestamped checkpoint directory")
    parser.add_argument("--checkpoint_name", type=str, default="best_loss_model.pth", help="Checkpoint filename to load")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=str2bool, default=True, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=str2bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument("--use_sk", type=str2bool, default=False, help="use sinkhorn or not")
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.1, 0.1, 0.1], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use-linear", type=int, default=1, help="use-linear")
    parser.add_argument("--device", type=str, default="cpu", help="gpu or cpu")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 64, 64], help="emb num of every vq")
    parser.add_argument("--e_dim", type=int, default=64, help="vq codebook embedding size")
    parser.add_argument("--quant_loss_weight", type=float, default=0.5, help="vq quantion loss weight")
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128], help="hidden sizes of every layer")

    return parser.parse_args()


def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint_path(ckpt_dir, checkpoint_name):
    direct_file = os.path.join(ckpt_dir, checkpoint_name)
    if os.path.isfile(direct_file):
        return direct_file

    subdirs = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, name))
    ]
    subdirs.sort(key=os.path.getmtime, reverse=True)
    for subdir in subdirs:
        candidate = os.path.join(subdir, checkpoint_name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot find checkpoint '{checkpoint_name}' under '{ckpt_dir}'."
    )


def build_model(args, data_dim):
    return CRQVAE(
        in_dim=data_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        use_linear=args.use_linear,
    )


def export_sid_codes():
    set_seed(2024)
    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)
    data = EmbDataset(args.data_path)
    model = build_model(args, data.dim)

    pin_memory = str(args.device).startswith("cuda")
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    checkpoint_path = resolve_checkpoint_path(args.ckpt_dir, args.checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()

    sids = {}
    vectors = {}
    iter_data = tqdm(
        data_loader,
        total=len(data_loader),
        ncols=100,
        desc=set_color("Generate codebooks ", "pink"),
    )

    for _, batch in enumerate(iter_data):
        pids, features = batch[0], batch[1]
        pids = pids.tolist()
        features = features.to(args.device)
        vector, indices = model.get_indices(features)
        for idx, poi in enumerate(pids):
            sids[poi] = indices[idx].tolist()
            vectors[poi] = vector[idx].tolist()

    value_counts = Counter(tuple(value) for value in sids.values())
    seen_values = {}
    updated_dict = {}
    for key in sorted(sids.keys()):
        value = sids[key]
        value_tuple = tuple(value)
        if value_counts[value_tuple] > 1:
            if value_tuple not in seen_values:
                seen_values[value_tuple] = 0
            else:
                seen_values[value_tuple] += 1
            updated_dict[key] = value + [seen_values[value_tuple]]
        else:
            updated_dict[key] = value

    ensure_dir(os.path.dirname(args.output_csv))
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pid", "sid", "vector"])
        for key, value in updated_dict.items():
            writer.writerow([key, value, vectors[key]])

    print(f"Exported {len(updated_dict)} SID rows to {args.output_csv}")


if __name__ == "__main__":
    export_sid_codes()
