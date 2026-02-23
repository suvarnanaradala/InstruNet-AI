import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_balance(metadata_path: Path):
    df = pd.read_csv(metadata_path)
    if "label" not in df.columns:
        raise SystemExit(f"metadata CSV {metadata_path} has no 'label' column")

    counts = df["label"].value_counts().sort_index()
    total = counts.sum()

    stats = {
        "num_classes": int(counts.size),
        "total_examples": int(total),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "mean_count": float(counts.mean()),
        "median_count": float(counts.median()),
        "std_count": float(counts.std()),
        "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float("inf"),
    }

    return counts, stats


def print_report(counts, stats):
    print("Class distribution:")
    print(counts.to_string())
    print()
    print("Summary:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


def save_counts_csv(counts, out_path: Path):
    counts.rename("count").to_frame().to_csv(out_path)


def plot_counts(counts, out_path: Path | None = None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    counts_sorted = counts.sort_values(ascending=False)
    ax.bar(counts_sorted.index.astype(str), counts_sorted.values, color="#4C72B0")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        plt.close(fig)
        return str(out_path)
    else:
        plt.show()
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Check class balance from processed/metadata.csv")
    p.add_argument("--metadata", default="processed/metadata.csv", help="Path to metadata CSV")
    p.add_argument("--save_csv", default=None, help="Optional path to write class counts CSV")
    p.add_argument("--plot", action="store_true", help="Show or save a bar plot of class counts")
    p.add_argument("--plot_out", default="class_counts.png", help="If --plot, optional output path to save plot")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise SystemExit(f"metadata file not found: {meta_path}")

    counts, stats = check_balance(meta_path)
    print_report(counts, stats)

    if args.save_csv:
        out_path = Path(args.save_csv)
        save_counts_csv(counts, out_path)
        print(f"Wrote counts CSV: {out_path}")

    if args.plot:
        plot_out = Path(args.plot_out) if args.plot_out else None
        saved = plot_counts(counts, out_path=plot_out)
        if saved:
            print(f"Wrote plot: {saved}")
