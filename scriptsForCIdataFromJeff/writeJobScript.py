#!/usr/bin/env python3
"""
Write a SLURM job script with the provided contents.

Usage:
  python write_job_script.py                 # writes ./job.slurm
  python write_job_script.py /path/to/dir    # writes /path/to/dir/job.slurm
  python write_job_script.py -o dataTrans.slurm
"""

from __future__ import annotations

import argparse
from pathlib import Path


JOB_TEXT = """#!/bin/bash
#SBATCH -A m2360
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name invDFT
#SBATCH -t 4:00:00
#SBATCH -n 512
#SBATCH --ntasks-per-node=64
#SBATCH -c 4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=vishalsu@umich.edu
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
srun -n 512 invDFT_exe gsParams.prm invParams.prm > output_invDFT
"""


def main():
    ap = argparse.ArgumentParser(description="Create a SLURM job script.")
    ap.add_argument("path", nargs="?", default=".", help="Output directory (default: .)")
    ap.add_argument("-o", "--output", default="job.slurm", help="Output filename (default: job.slurm)")
    args = ap.parse_args()

    out_dir = Path(args.path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.output
    out_path.write_text(JOB_TEXT, encoding="utf-8")

    # Optional: make it executable on Unix-like systems
    try:
        out_path.chmod(out_path.stat().st_mode | 0o111)
    except Exception:
        pass


if __name__ == "__main__":
    main()
