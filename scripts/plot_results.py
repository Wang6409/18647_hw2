#!/usr/bin/env python3
"""Summarize reorder_bench CSV output and create bandwidth/roofline plots."""

import argparse
import csv
import os
import statistics


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No measurements found in {path}")
    return rows


def write_table(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="reorder_bench CSV file")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: "
            "python3 -m pip install matplotlib"
        ) from error

    rows = read_rows(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    grouped = {}
    metadata = {}
    copy_rates = {}
    for row in rows:
        rank = int(row["rank"])
        threads = int(row["threads"])
        seconds = float(row["seconds"])
        key = (rank, threads)
        grouped.setdefault(key, []).append(seconds)
        metadata[rank] = (row["dimensions"], int(row["elements"]),
                          int(row["destination_inner_stride_bytes"]))
        copy_rates.setdefault(threads, []).append(
            float(row["copy_bandwidth_GB_s"])
        )

    ranks = sorted(metadata)
    core_counts = sorted({threads for _, threads in grouped})
    mean_times = {
        key: statistics.mean(samples) for key, samples in grouped.items()
    }
    bandwidth = {
        key: (2 * metadata[key[0]][1] / mean_times[key] / 1.0e9)
        for key in mean_times
    }
    speedup = {
        (rank, threads): mean_times[(rank, 1)] / mean_times[(rank, threads)]
        for rank in ranks
        for threads in core_counts
    }

    bandwidth_headers = ["rank", "dimensions"] + [
        f"cores_{threads}_GB_s" for threads in core_counts
    ]
    bandwidth_rows = [
        [rank, metadata[rank][0]]
        + [bandwidth[(rank, threads)] for threads in core_counts]
        for rank in ranks
    ]
    write_table(
        os.path.join(args.output_dir, "bandwidth_vs_cores.csv"),
        bandwidth_headers,
        bandwidth_rows,
    )

    speedup_headers = ["rank", "dimensions"] + [
        f"cores_{threads}_speedup" for threads in core_counts
    ]
    speedup_rows = [
        [rank, metadata[rank][0]]
        + [speedup[(rank, threads)] for threads in core_counts]
        for rank in ranks
    ]
    write_table(
        os.path.join(args.output_dir, "speedup_vs_cores.csv"),
        speedup_headers,
        speedup_rows,
    )

    traffic_headers = [
        "rank",
        "dimensions",
        "elements",
        "logical_read_plus_write_bytes",
        "ideal_write_allocate_bytes",
        "destination_inner_stride_bytes",
    ]
    traffic_rows = [
        [
            rank,
            metadata[rank][0],
            metadata[rank][1],
            2 * metadata[rank][1],
            3 * metadata[rank][1],
            metadata[rank][2],
        ]
        for rank in ranks
    ]
    write_table(
        os.path.join(args.output_dir, "traffic_model.csv"),
        traffic_headers,
        traffic_rows,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for rank in ranks:
        axis.plot(
            core_counts,
            [bandwidth[(rank, threads)] for threads in core_counts],
            marker="o",
            label=f"rank {rank}",
        )
    axis.set(
        xlabel="OpenMP threads (cores)",
        ylabel="Effective bandwidth (GB/s, decimal)",
        title="Tensor reorder bandwidth by rank and core count",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    figure.savefig(
        os.path.join(args.output_dir, "bandwidth_vs_cores.png"), dpi=160
    )
    figure.savefig(os.path.join(args.output_dir, "bandwidth_vs_cores.pdf"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for rank in ranks:
        axis.plot(
            core_counts,
            [bandwidth[(rank, threads)] for threads in core_counts],
            marker="o",
            label=f"rank {rank} reorder",
        )
    axis.plot(
        core_counts,
        [statistics.mean(copy_rates[threads]) for threads in core_counts],
        color="black",
        linestyle="--",
        marker="s",
        label="parallel contiguous-copy bandwidth ceiling",
    )
    axis.set(
        xlabel="OpenMP threads (cores)",
        ylabel="Effective bandwidth (GB/s, decimal)",
        title="Memory-bandwidth roofline proxy",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    figure.savefig(os.path.join(args.output_dir, "roofline.png"), dpi=160)
    figure.savefig(os.path.join(args.output_dir, "roofline.pdf"))
    plt.close(figure)

    print(f"Wrote plots and summary CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
