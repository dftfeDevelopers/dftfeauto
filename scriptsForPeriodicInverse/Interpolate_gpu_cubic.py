#!/usr/bin/env python3
import torch
import numpy as np
import argparse
import json
import os

# Select GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANGS_TO_BOHR = 1.8897259886

def parse_args():
    parser = argparse.ArgumentParser(description="Tricubic interpolation of density on GPU/CPU")
    parser.add_argument('--rho', type=str, default="rhoSymm_amd_double_coord", help="Path to density file (flat list)")
    parser.add_argument('--quad-in', type=str, default="densityQuadData.txt", help="Input quadrature file")
    parser.add_argument('--out', type=str, default="dftfeOut_symm.txt", help="Output file")
    parser.add_argument('--qbatch', type=int, default=50000, help="Batch size for interpolation")
    parser.add_argument('--n1d', type=int, default=None, help="Grid points per axis (without +1)")
    parser.add_argument('--a', type=float,default= None, help="Cell length a in Angstroms")
    parser.add_argument('--offset', type=float, default=None, help="Grid offset in Angstroms")
    parser.add_argument('--nelec', type=float, default=None, help="Reference electron count for normalization")
    parser.add_argument('--plot-y', type=str, default='rho_y', help="Output file for y-line interpolation plot data")
    parser.add_argument('--meta', type=str, default=None, help="Path to metadata JSON exported from XSF parser")
    parser.add_argument('--density', type=str, default=None, help="Path to density file exported from XSF parser; overrides --rho if given")
    return parser.parse_args()

# -------------- Cubic Kernel Functions -------------- #

def cubic_weight(x):
    absx = torch.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    # Catmull-Rom spline: a = -0.5
    w1 = 1.5 * absx3 - 2.5 * absx2 + 1
    w2 = -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    return torch.where(absx < 1, w1, torch.where(absx < 2, w2, torch.zeros_like(x)))

def cubic_indices_and_weights(x, ngrid):
    """
    For points x in [0, ngrid-1], get indices (N, 4) and weights (N, 4) for cubic interpolation.
    """
    x0 = torch.floor(x) - 1
    idxs = torch.stack([x0, x0+1, x0+2, x0+3], dim=1).long()
    idxs = idxs % ngrid  # wrap for periodic boundary
    dx = x - (x0 + 1)  # offset from 2nd of the group
    ws = torch.stack([
        cubic_weight(dx + 1),
        cubic_weight(dx),
        cubic_weight(dx - 1),
        cubic_weight(dx - 2),
    ], dim=1)
    return idxs, ws

def tricubic_interp(grid_x, density, points):
    """
    density: (ngrid, ngrid, ngrid) - Z, Y, X order
    points: (N, 3), coordinates in grid units
    grid_x: (ngrid,), 1D grid positions
    Returns: (N,) interpolated values
    """
    ngrid = grid_x.shape[0]
    # Remap coordinates to [0, ngrid-1]
    coords = (points - grid_x[0]) / (grid_x[1] - grid_x[0])
    idxs_x, wx = cubic_indices_and_weights(coords[:, 0], ngrid)
    idxs_y, wy = cubic_indices_and_weights(coords[:, 1], ngrid)
    idxs_z, wz = cubic_indices_and_weights(coords[:, 2], ngrid)

    N = points.shape[0]
    out = torch.zeros(N, dtype=density.dtype, device=density.device)
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                vals = density[idxs_z[:,iz], idxs_y[:,iy], idxs_x[:,ix]]  # Z,Y,X
                weight = wx[:, ix] * wy[:, iy] * wz[:, iz]
                out += vals * weight
    return out

def main():
    args = parse_args()

    if args.meta is not None:
        if os.path.exists(args.meta):
            with open(args.meta, 'r') as jf:
                meta = json.load(jf)
            # Override n1d from grid_shape if present
            if 'grid_shape' in meta and isinstance(meta['grid_shape'], list) and len(meta['grid_shape']) > 0:
                n1d_meta = meta['grid_shape'][0] - 1
                print(f"Metadata override: n1d set to {n1d_meta} from grid_shape")
                args.n1d = n1d_meta
            # Override a from lattice if cubic lattice is detected
            if 'lattice_vectors' in meta and isinstance(meta['lattice_vectors'], list) and len(meta['lattice_vectors']) == 3:
                lattice = meta['lattice_vectors']
                # Check if lattice is cubic (diagonal vectors equal length and off-diagonal zero)
                try:
                    v0 = np.array(lattice[0])
                    v1 = np.array(lattice[1])
                    v2 = np.array(lattice[2])
                    off_diag_zero = (
                        np.allclose([v0[1], v0[2], v1[0], v1[2], v2[0], v2[1]], 0.0, atol=1e-8)
                    )
                    lengths = np.array([np.linalg.norm(v0), np.linalg.norm(v1), np.linalg.norm(v2)])
                    if off_diag_zero and np.allclose(lengths, lengths[0], atol=1e-6):
                        args.a = float(lengths[0])
                        print(f"Metadata override: a set to {args.a} Angstrom from cubic lattice vectors")
                except Exception:
                    pass

            # Override offset from origin if present
            if 'origin' in meta and isinstance(meta['origin'], list) and len(meta['origin']) > 0:
                origin_x = float(meta['origin'][0])
                if args.a > 0:
                    offset_ang = origin_x
                    args.offset = offset_ang
                    print(f"Metadata override: offset set to {args.offset} Angstrom from origin")

    # Select density file based on args.density if given
    if args.density is not None:
        rhoFname = args.density
        print(f"Using density file from --density: {rhoFname}")
    else:
        rhoFname = args.rho
        print(f"Using density file from --rho: {rhoFname}")

    print('Using device:', device)
    print('Quadrature input file:', args.quad_in)
    print('Output file:', args.out)

    dftfeInFname = args.quad_in
    dftfeOutFname = args.out
    qbatch = args.qbatch
    n1d = args.n1d
    aAngs = args.a
    offsetAngs = args.offset
    nelectronsRef = args.nelec

    a = aAngs * ANGS_TO_BOHR
    offset = offsetAngs * ANGS_TO_BOHR
    n1dUsed = n1d + 1
    h = a / n1d
    grid_np = np.linspace(0, a, n1dUsed) + offset
    grid_x = torch.from_numpy(grid_np).float().to(device)  # For interpolation

    # ---------- Read and Prepare Density ---------- #
    rho_ = []
    with open(rhoFname) as f:
        for line in f:
            vals = line.strip().split()
            if vals:
                rho_.extend(map(float, vals))
    rho_np = np.array(rho_, dtype=np.float32)
    assert rho_np.size == n1dUsed ** 3, "Density size mismatch."
    rho_np = rho_np.reshape((n1dUsed, n1dUsed, n1dUsed))  # Z, Y, X
    rho = torch.from_numpy(rho_np).to(device)

    # Electron normalization
    nelectrons = rho[:n1d, :n1d, :n1d].sum().item()
    rho *= (nelectronsRef / nelectrons)
    w = (a ** 3.0) / (n1d ** 3)
    rho /= w

    # ----------- Load quad points & weights ------------ #
    dftfeIn = np.loadtxt(dftfeInFname, dtype=np.float32)
    dftfeOut = np.zeros((dftfeIn.shape[0], dftfeIn.shape[1] + 1), dtype=np.float32)
    dftfeOut[:, :6] = dftfeIn

    qpts_np = dftfeIn[:, 1:4]
    qwts_np = dftfeIn[:, 4]
    nquad = qwts_np.shape[0]

    # Remap quad points for periodicity
    qpts_np = qpts_np - np.floor((qpts_np - offset) / a) * a
    points = torch.from_numpy(qpts_np).float().to(device)

    qwts = torch.from_numpy(qwts_np).float().to(device)

    # ----------- Do interpolation in batches ------------ #
    rhoOut = torch.zeros(nquad, dtype=torch.float32, device=device)
    for start in range(0, nquad, qbatch):
        end = min(nquad, start + qbatch)
        batch_pts = points[start:end]
        rhoOut[start:end] = tricubic_interp(grid_x, rho, batch_pts)
        print(f"Cubic batch {start}-{end} done")
    nelectronsEval = (rhoOut * qwts).sum().item()
    print("No. of electrons (cubic):", nelectronsEval)

    rhoOut *= nelectronsRef / nelectronsEval
    dftfeOut[:, 5] = rhoOut.cpu().numpy()
    np.savetxt(dftfeOutFname, dftfeOut)

    # ----------- Interpolate along y-axis for plotting ------------ #
    npts = 200
    y = np.linspace(0, a, npts)
    pts_np = np.zeros((npts, 3), dtype=np.float32)
    pts_np[:, 1] = y
    pts_np = pts_np - np.floor((pts_np - offset) / a) * a
    pts = torch.from_numpy(pts_np).float().to(device)
    rho_y = tricubic_interp(grid_x, rho, pts).cpu().numpy()
    data = np.column_stack((y, rho_y))
    np.savetxt(args.plot_y, data)

    print("Done. Results written. (Cubic interpolation, PyTorch, device:", device, ")")

if __name__ == '__main__':
    main()


