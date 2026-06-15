"""
exp_plot.py — turn results.csv from exp_preempt.py into the two report figures.

  Plot 1: preemptions vs N        (shows the saturation knee)
  Plot 2: throughput vs N         (shows it plateaus, not cliffs)

Run on the login node (has matplotlib) or locally after scp'ing results.csv:
  python exp_plot.py --csv results.csv
Writes preempt_vs_n.png and throughput_vs_n.png next to the csv.
"""

import argparse
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results.csv")
    args = p.parse_args()

    N, preempt, tput, recov = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            N.append(int(row["N"]))
            preempt.append(int(row["num_preemptions"]))
            tput.append(float(row["throughput_tok_s"]))
            recov.append(int(row["cached_blocks_on_realloc"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: preemptions (and recovered blocks) vs N
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N, preempt, "o-", label="preemptions")
    ax.plot(N, recov, "s--", label="blocks recovered (prefix cache)")
    ax.set_xlabel("N (concurrent requests)")
    ax.set_ylabel("count")
    ax.set_title("Preemptions vs load — recovery stays ~0")
    ax.legend()
    fig.tight_layout()
    fig.savefig("preempt_vs_n.png", dpi=150)
    print("wrote preempt_vs_n.png")

    # Plot 2: throughput vs N
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N, tput, "o-", color="tab:green")
    ax.set_xlabel("N (concurrent requests)")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Throughput plateaus (no cliff) once preemption begins")
    fig.tight_layout()
    fig.savefig("throughput_vs_n.png", dpi=150)
    print("wrote throughput_vs_n.png")


if __name__ == "__main__":
    main()
