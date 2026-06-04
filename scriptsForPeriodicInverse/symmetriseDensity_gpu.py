#!/usr/bin/env python3
import numpy as np
import torch
import time
import argparse
import json
import os
import pandas as pd

# ---- CPU SECTION ----
import spglib
from pymatgen.core import Structure, Lattice, Species, Element


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symmetrise density on GPU using spglib/pymatgen symmetries"
    )
    parser.add_argument(
        "--coords-file",
        type=str,
        default=None,
        help="Path to text file with fractional coordinates (Nx3). If omitted, use built-in coords",
    )
    parser.add_argument(
        "--coords-ang",
        type=bool,
        default = None,
        help="Interpret input coordinates (from --coords-file) as Cartesian Angstroms; convert to fractional before building Structure",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="rhoSymm_amd_double_coord",
        help="Output file for symmetrized density",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to metadata JSON exported from XSF parser",
    )
    parser.add_argument(
        "--density",
        type=str,
        default=None,
        help="Path to density file exported from XSF parser; overrides --rho if given",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override from metadata JSON if provided
    if args.meta is not None:
        if not os.path.isfile(args.meta):
            raise FileNotFoundError(f"Metadata JSON file '{args.meta}' not found")
        with open(args.meta, "r") as fmeta:
            meta = json.load(fmeta)
        # Override lattice parameter a if cubic lattice is detected
        grid_shape = meta.get("grid_shape", None)
        if len(grid_shape) !=3:
            raise ValueError("grid_shape is not proper in meta data")
        if grid_shape[0]!=grid_shape[1] or grid_shape[0]!= grid_shape[2]:
            raise ValueError("grid_shape is not cubic")

        n1d = grid_shape[0]-1
        lattice_vectors = meta.get("lattice_vectors", None)
        if lattice_vectors is not None and len(lattice_vectors) == 3:
            # Check if cubic: all vectors same norm within tolerance
            vecs = np.array(lattice_vectors)
            norms = np.linalg.norm(vecs, axis=1)
            if np.allclose(norms, norms[0], atol=1e-5):
                aAngs = float(norms[0])
                print(f"Metadata overrides lattice parameter a: {aAngs} Å (cubic detected)")
        # Override offset if origin present
        origin = meta.get("origin", None)
        if origin is not None:
            # Accept scalar or 3-vector
            if isinstance(origin, (list, tuple)) and len(origin) >= 1:
                origin_x = float(origin[0])
            else:
                origin_x = float(origin)
            offset = origin_x/aAngs 
            # If clearly fractional, use directly; else convert from Angstroms
            #if 0.0 <= origin_x <= 1.0:
            #    offset = origin_x
            #    print(f"Metadata overrides offset: {offset} (fractional)")
            #else:
            #    offset = origin_x / aAngs
            #    print(f"Metadata overrides offset: {origin_x} Å converted to fractional {offset}")

    lattice = Lattice.cubic(aAngs)

    if args.coords_file is not None:
        coords_array = pd.read_csv(args.coords_file)
        if coords_array.ndim != 2 or coords_array.shape[1] != 4:
            raise ValueError("Coordinates file must have Nx4 shape")

        if args.coords_ang:
            # Convert Cartesian Angstroms to fractional coordinates
            frac_coords = lattice.get_fractional_coords(coords_array[["x", "y", "z"]].to_numpy())
            coords = frac_coords.tolist()
            print(f"Converted input coordinates from Cartesian Angstroms to fractional.")
        else:
            # Already fractional
            coords = coords_array[["x", "y", "z"]].to_numpy().tolist()

        species = coords_array["symbol"].tolist()
        coords_source = f"from file '{args.coords_file}'"
        if args.coords_ang:
            coords_source += " (converted from Cartesian Angstroms)"

    # Create the Structure object
    structure = Structure(lattice, species, coords)
    print(structure)

    # Get spglib input format from pymatgen Structure
    lattice_np = structure.lattice.matrix
    numbers = [site.specie.number for site in structure]

    cell = (lattice_np, coords, numbers)

    symmetry_data = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    rotations = symmetry_data["rotations"]
    translations = symmetry_data["translations"]

    # Read density as before
    rhoFname = args.density if args.density else args.rho

    print(f"Using density file: '{rhoFname}'")

    n1dUsed = n1d + 1

    print(f"Lattice parameter a used: {aAngs}")
    print(f"Grid offset used: {offset}")

    x_np = np.linspace(0, 1, n1dUsed) + offset
    step = x_np[1] - x_np[0]

    rho_ = np.zeros(n1dUsed * n1dUsed * n1dUsed)
    count = 0
    with open(rhoFname) as f:
        lines = f.readlines()
        for line in lines:
            line_ = line.strip()
            if line_:
                vals = [float(x) for x in line_.split()]
                start = count
                end = count + len(vals)
                rho_[start:end] = vals
                count += len(vals)

    density_grid_np = np.zeros((n1dUsed, n1dUsed, n1dUsed))
    count = 0
    for i in range(n1dUsed):
        for j in range(n1dUsed):
            for k in range(n1dUsed):
                density_grid_np[i, j, k] = rho_[count]
                count += 1

    # ---- GPU SECTION ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # Convert numpy arrays to torch tensors and move to GPU
    density_grid = torch.from_numpy(density_grid_np).to(device)
    x = torch.from_numpy(x_np).to(device)

    # Fractional coordinate grid
    ix, iy, iz = torch.meshgrid(
        torch.arange(n1dUsed, device=device),
        torch.arange(n1dUsed, device=device),
        torch.arange(n1dUsed, device=device),
        indexing="ij",
    )
    coords_grid = torch.stack([x[ix], x[iy], x[iz]], dim=-1)  # shape (n1dUsed, n1dUsed, n1dUsed, 3)

    symmetrized_grid = torch.zeros_like(density_grid)

    print(f"Number of grid points n1d = {n1d}")
    print(f"Density input file = '{rhoFname}'")
    print(f"Coordinates source = {coords_source}")
    print(f"Output file = '{args.out}'")

    loop_start = time.perf_counter()

    i = 0
    for R_np, t_np in zip(rotations, translations):
        i = i + 1
        # R and t as torch GPU tensors
        R = torch.from_numpy(R_np.T).double().to(device)  # .T for correct application
        t = torch.from_numpy(t_np).double().to(device)

        # Reshape coordinate grid for matmul
        coords_flat = coords_grid.reshape(-1, 3)
        tcoords = coords_flat @ R  # matmul
        tcoords += t

        tcoords = tcoords.reshape(n1dUsed, n1dUsed, n1dUsed, 3)
        # Wrap into unit cell
        tcoords = torch.remainder(tcoords - x[0], 1.0) + x[0]

        # Map to index
        idxs = ((tcoords - x[0]) / step).round().long()
        i_new = torch.remainder(idxs[..., 0], n1dUsed)
        j_new = torch.remainder(idxs[..., 1], n1dUsed)
        k_new = torch.remainder(idxs[..., 2], n1dUsed)
        # Fast gather, using advanced indexing
        transformed_density = density_grid[i_new, j_new, k_new]
        symmetrized_grid += transformed_density

    print(f"performed {i} symmetry operations")
    loop_end = time.perf_counter()
    print(f"looptime: {(loop_end - loop_start)} seconds")
    symmetrized_grid /= len(rotations)

    np.savetxt(args.out, symmetrized_grid.flatten().cpu().numpy())


if __name__ == "__main__":
    main()


