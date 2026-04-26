"""Window-size sensitivity sweep for the eigengap signal.

Re-runs the eigengap calculation post-hoc on saved log files for different
window sizes W ∈ {10, 20, 50, 100}.

PROBLEM: the original tian_eigengap.py only saves σ₁..σ₅ for the *fixed*
window it was run with, NOT the raw ΔW snapshots. So we cannot truly
re-window from existing logs. Instead this script does fresh runs for each W
on a single seed and saves them as new tags.
"""

import subprocess
import sys
from pathlib import Path

WINDOWS = [10, 20, 50, 100]
SEEDS = [0, 1, 2]
ETAS = ["0.0002", "0"]

HERE = Path(__file__).parent

def main():
    for W in WINDOWS:
        for seed in SEEDS:
            for eta in ETAS:
                tag = f"win_W{W}_eta{eta}_seed{seed}"
                out = HERE / "runs" / tag / "log.jsonl"
                if out.exists() and sum(1 for _ in open(out)) >= 401:
                    print(f"[skip] {tag}")
                    continue
                cmd = [
                    sys.executable, "-u", "tian_eigengap.py",
                    "--M", "71", "--K", "2048", "--n-train", "2016",
                    "--eta", eta, "--lr", "1e-3",
                    "--epochs", "400", "--eval-every", "200",
                    "--window", str(W), "--seed", str(seed),
                    "--tag", tag,
                ]
                print(f"[run] {tag}")
                subprocess.run(cmd, cwd=HERE, check=True)
    print("done")


if __name__ == "__main__":
    main()
