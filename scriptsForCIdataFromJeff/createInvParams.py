#!/usr/bin/env python3
"""
Write invParams.prm with fixed contents.

Usage:
  python make_invParams.py            # writes ./invParams.prm
  python make_invParams.py /path/dir  # writes /path/dir/invParams.prm
"""

from __future__ import annotations

import argparse
from pathlib import Path


INV_PARAMS_TEXT = """set SOLVER MODE = INVERSE
subsection Inverse DFT parameters
 set TOL FOR BFGS = 1e-12
 set BFGS LINE SEARCH = 1
 set TOL FOR BFGS LINE SEARCH = 1e-06
 set BFGS HISTORY = 100
 set BFGS MAX ITERATIONS = 10000
 set READ FE DENSITY DATA = false
 set READ VXC DATA = false
 set POSTFIX TO THE FILENAME FOR READING VXC DATA = vxcData_mesh0p06_ball6p0_temp10_1_20
 set WRITE VXC DATA = true
 set FOLDER FOR VXC DATA = vxcDataOut
 set POSTFIX TO THE FILENAME FOR WRITING VXC DATA = vxcData_1
 set FREQUENCY FOR WRITING VXC = 30
 set RHO TOL FOR CONSTRAINTS = 5e-06
 set VXC MESH DOMAIN SIZE = 6.0
 set VXC MESH SIZE NEAR ATOM = 0.06
 set BLOCK SIZE OF INTERPOLATE = 14
 set INITIAL TOL FOR CHEBYSHEV FILTERING = 1e-08
 set MAX ITERATIONS FOR ADJOINT PROBLEM            = 10000
 set ALPHA1 FOR WEIGHTS FOR LOSS FUNCTION = 0.0
 set ALPHA2 FOR WEIGHTS FOR LOSS FUNCTION = 0.0
 set TAU FOR WEIGHTS FOR LOSS FUNCTION = 1e-05
 set TAU FOR WEIGHTS FOR SETTING VX BC = 1e-08
 set TAU FOR WEIGHTS FOR SETTING FABC = 0.001
 set TOL FOR FRACTIONAL OCCUPANCY = 1e-08
 set TOL FOR DEGENERACY = 0.005
 set FACTOR FOR LDA VXC = 0.0
 set READ GAUSSIAN DATA AS INPUT = true
 set READ SLATER DATA AS INPUT = false
 set SET FERMIAMALDI IN THE FAR FIELD AS INPUT = true
 set USE DELTA RHO CORRECTION = true
 set GAUSSIAN DENSITY FOR PRIMARY RHO SPIN UP = DensityMatrix_HBCI
 set GAUSSIAN DENSITY FOR DFT RHO SPIN UP = DensityMatrix_LDA
 set ATOMIC ORBITAL ATOMIC COORD FILE = AtomicCoords
 set GAUSSIAN S MATRIX FILE = SMatrix
end
subsection POST PROCESS
 set WRITE VTU FILE = true
 set INTERPOLATE TO POINTS = true
 set READ POINTS FROM FILE = false
 set FILENAME FOR OUTPUT = vxcFinalpp
 set STARTING X = -15.0
 set STARTING Y = -15.0
 set STARTING Z = -15.0
 set ENDING X = 15.0
 set ENDING Y = 15.0
 set ENDING Z = 15.0
 set NUMBER OF POINTS ALONG X DIRECTION = 500
 set NUMBER OF POINTS ALONG Y DIRECTION = 500
 set NUMBER OF POINTS ALONG Z DIRECTION = 500
end
"""


def main():
    ap = argparse.ArgumentParser(description="Create invParams.prm")
    ap.add_argument("path", nargs="?", default=".", help="Output directory (default: .)")
    args = ap.parse_args()

    out_dir = Path(args.path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "invParams.prm"
    out_path.write_text(INV_PARAMS_TEXT, encoding="utf-8")


if __name__ == "__main__":
    main()
