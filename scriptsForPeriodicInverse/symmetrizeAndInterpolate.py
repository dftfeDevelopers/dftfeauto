#!/usr/bin/env python3
"""
This script reads a file specified as an argument and prints summary information 
or the full contents of the file.
"""

import argparse
import os
import sys
import subprocess
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
import csv
import numpy as np


@dataclass
class XSFData:
    lattice: List[Tuple[float, float, float]] = field(default_factory=list)
    atoms: List[Tuple[str, float, float, float]] = field(default_factory=list)
    grid_shape: Optional[Tuple[int, int, int]] = None
    origin: Optional[Tuple[float, float, float]] = None
    density: Optional[List[float]] = None

def parse_floats(line: str) -> List[float]:
    return list(map(float, line.split()))

def parse_xsf(contents: str) -> XSFData:
    """
    Parse a typical XSF file content extracting lattice vectors, atoms, and a single 3D data grid.
    This parser assumes a common structure but XSF variants may exist.
    """
    lines = contents.splitlines()
    data = XSFData()
    i = 0
    nlines = len(lines)

    # Mapping from atomic number to symbol for common elements
    atomic_num_to_symbol = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        14: "Si"
    }

    while i < nlines:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        low = line.lower()

        # Parse lattice vectors: PRIMVEC or CELLVEC sections (3 lines of vectors)
        if low.startswith("primvec") or low.startswith("cellvec"):
            lattice = []
            i += 1
            for _ in range(3):
                if i >= nlines:
                    break
                lattice_line = lines[i].strip()
                lattice.append(tuple(parse_floats(lattice_line)))
                i += 1
            data.lattice = lattice
            continue

        # Parse atoms: PRIMCOORD or ATOMS sections
        if low.startswith("primcoord") or low.startswith("atoms"):
            # Following line or trailing count on same line
            count = None
            tokens = line.split()
            if len(tokens) > 1:
                try:
                    count = int(tokens[1])
                except ValueError:
                    count = None
            if count is None:
                i += 1
                if i >= nlines:
                    break
                try:
                    count = int(lines[i].strip().split()[0])
                    i += 1
                except (ValueError, IndexError):
                    # Unable to get count, abort atom parsing
                    continue
            else:
                i += 1

            atoms = []
            for _ in range(count):
                if i >= nlines:
                    break
                atom_line = lines[i].strip()
                parts = atom_line.split()
                if len(parts) >= 4:
                    # Determine if first token is atomic number or symbol
                    try:
                        z = int(parts[0])
                        symbol = atomic_num_to_symbol.get(z, f"Z{z}")
                        coords = tuple(map(float, parts[1:4]))
                        atoms.append((symbol, *coords))
                    except ValueError:
                        # Not an integer, treat first token as symbol
                        symbol = parts[0]
                        try:
                            coords = tuple(map(float, parts[1:4]))
                            atoms.append((symbol, *coords))
                        except ValueError:
                            pass
                i += 1
            data.atoms = atoms
            continue

        # Parse 3D grid data: BEGIN_DATAGRID_3D or BEGIN_BLOCK_DATAGRID_3D (case-insensitive)
        # Also handle block/named variants with suffixes
        if low.startswith("begin_block_datagrid_3d"):
            i += 1
            if i >= nlines:
                break
            # Next line: block label, ignore
            block_label_line = lines[i].strip()
            i += 1
            if i >= nlines:
                break
            line2 = lines[i].strip()
            low2 = line2.lower()
            # Expect line starting with begin_datagrid_3d
            if not low2.startswith("begin_datagrid_3d"):
                # Not valid format, skip
                continue
            # Set end tag prefix for matching end line
            end_tag_prefix = "end_datagrid_3d"
            i += 1
            if i >= nlines:
                break

        elif low.startswith("begin_datagrid_3d"):
            end_tag_prefix = "end_datagrid_3d"
            i += 1
            if i >= nlines:
                break
        else:
            i += 1
            continue

        # Now parse grid shape line
        shape_line = lines[i].strip()
        try:
            Nx, Ny, Nz = map(int, shape_line.split())
            data.grid_shape = (Nx, Ny, Nz)
        except ValueError:
            # Cannot parse grid shape, abort grid reading
            continue
        i += 1
        if i >= nlines:
            break

        # Origin vector line (3 floats)
        origin_line = lines[i].strip()
        try:
            origin_vals = parse_floats(origin_line)
            if len(origin_vals) == 3:
                data.origin = tuple(origin_vals)
            else:
                data.origin = None
        except ValueError:
            data.origin = None
        i += 1
        if i >= nlines:
            break

        # Next 3 lines: lattice/domain vectors
        lattice = []
        for _ in range(3):
            if i >= nlines:
                break
            lattice_line = lines[i].strip()
            try:
                lattice.append(tuple(parse_floats(lattice_line)))
            except ValueError:
                pass
            i += 1
        # Only set lattice if not already set
        if not data.lattice and len(lattice) == 3:
            data.lattice = lattice

        # Read Nx*Ny*Nz floats (density values)
        count_vals = Nx * Ny * Nz
        density_vals = []
        while i < nlines:
            vals_line = lines[i].strip()
            low_line = vals_line.lower()
            # End of data grid detection: any line starting with end_tag_prefix
            if low_line.startswith(end_tag_prefix):
                i += 1
                break
            vals = vals_line.split()
            try:
                floats = list(map(float, vals))
                density_vals.extend(floats)
            except ValueError:
                # Ignore malformed line
                pass
            i += 1
            if len(density_vals) >= count_vals:
                density_vals = density_vals[:count_vals]
                # Still need to skip until end tag
                while i < nlines:
                    end_line = lines[i].strip().lower()
                    if end_line.startswith(end_tag_prefix):
                        i += 1
                        break
                    i += 1
                break
        data.density = density_vals[:count_vals]

    return data

def export_xsf(data: XSFData, base_path: str) -> dict:
    """
    Export XSFData to interoperable files:
      - density data as whitespace-separated floats, 6 per line
      - metadata JSON with grid_shape, origin, lattice_vectors, atoms_file
      - optionally atoms CSV file with symbol,x,y,z
    
    Returns dict with paths of created files.
    """

    if data.grid_shape is None:
        raise ValueError("Cannot export: grid_shape is missing in XSFData")
    if data.density is None:
        raise ValueError("Cannot export: density is missing in XSFData")

    # Write density file
    density_path = f"{base_path}_density.txt"
    with open(density_path, "w", encoding="utf-8") as f:
        vals_per_line = 6
        vals = data.density
        for i in range(0, len(vals), vals_per_line):
            line_vals = vals[i:i+vals_per_line]
            line_str = " ".join(f"{v:.10e}" for v in line_vals)
            f.write(line_str + "\n")

    # Write atoms file if present
    atoms_path = None
    if data.atoms:
        atoms_path = f"{base_path}_atoms.csv"
        with open(atoms_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["symbol", "x", "y", "z"])
            for atom in data.atoms:
                writer.writerow(atom)

    # Prepare metadata
    origin_export = None
    if data.origin is not None:
        origin_arr = np.array(data.origin)
        origin_export = list(origin_arr)
    meta = {
        "grid_shape": list(data.grid_shape),
        "origin": origin_export if origin_export is not None else None,
        "lattice_vectors": [list(vec) for vec in data.lattice] if data.lattice else [],
        "atoms_file": atoms_path if atoms_path is not None else None
    }

    meta_path = f"{base_path}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    paths = {
        "density": density_path,
        "meta": meta_path,
    }
    if atoms_path is not None:
        paths["atoms"] = atoms_path

    return paths

def print_parsed_summary(data: XSFData):
    print("Parsed XSF Data Summary:")

    if data.lattice:
        print("Lattice vectors:")
        for vec in data.lattice:
            print(f"  {vec[0]: .6f} {vec[1]: .6f} {vec[2]: .6f}")
    else:
        print("Lattice vectors: None")

    if data.atoms:
        print(f"Atoms (count: {len(data.atoms)}):")
        for atom in data.atoms[:10]:
            symbol, x, y, z = atom
            print(f"  {symbol} {x: .6f} {y: .6f} {z: .6f}")
        if len(data.atoms) > 10:
            print("  ... (truncated)")
    else:
        print("Atoms: None")

    if data.grid_shape:
        print(f"Grid shape: {data.grid_shape}")
    else:
        print("Grid shape: None")

    if data.origin:
        print(f"Origin: {data.origin[0]: .6f} {data.origin[1]: .6f} {data.origin[2]: .6f}")
    else:
        print("Origin: None")

    if data.density:
        print(f"Density values: {len(data.density)} entries")
        preview_count = min(len(data.density), 10)
        preview = " ".join(f"{v:.6e}" for v in data.density[:preview_count])
        print(f"  Preview: {preview}{' ...' if len(data.density) > preview_count else ''}")
    else:
        print("Density values: None")

def read_file(path: str) -> str:
    # Open and read the entire file, decoding as UTF-8, replacing errors
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def print_summary(contents: str, path: str, max_lines: int = 20):
    lines = contents.splitlines()
    # Print file information
    print(f"File: {path}")
    print(f"Total bytes: {len(contents.encode('utf-8', 'replace'))}")
    print(f"Total lines: {len(lines)}")
    print(f"--- Begin Preview (up to {max_lines} lines) ---")
    # Print up to max_lines lines
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print("... (truncated)")
    print("--- End Preview ---")


import subprocess
import sys
import os

def run_full_density_pipeline(
    xsf_path: str,
    export_base: str = "xsf_out",
    symm_script: str = "symmetriseDensity_gpu.py",
    interp_script: str = "Interpolate_gpu_cubic.py",
    symm_out: str | None = None,
    interp_out: str | None = None,
    quad_in: str = "densityQuadData.txt",
    nelec_ref: float = 864.0,
    plot_y: str = "rho_y",
    extra_symm_args: list[str] | None = None,
    extra_interp_args: list[str] | None = None,
) -> dict:
    """
    Runs the full pipeline:
    1) Parse XSF and export metadata + density via symmetrizeAndInterpolate.py
    2) Symmetrise density via symmetriseDensity_gpu.py
    3) Interpolate via Interpolate_gpu_cubic.py

    Returns a dict with paths to the files produced.
    """
    # Resolve outputs
    meta_path = f"{export_base}_meta.json"
    density_path = f"{export_base}_density.txt"
    coords_path = f"{export_base}_atoms.csv"
    symm_out = symm_out or f"{export_base}_symm.txt"
    interp_out = interp_out or f"{export_base}_interp.txt"

    # Step 1: Export from XSF
    print(f"[1/3] Exporting from XSF: {xsf_path}")
    export_cmd = [
        sys.executable, "symmetrizeAndInterpolate.py",
        xsf_path, "--parse", "--export", export_base
    ]
    subprocess.run(export_cmd, check=True)
    if not (os.path.exists(meta_path) and os.path.exists(density_path)):
        raise RuntimeError("Export step failed: missing meta or density output.")

    # Step 2: Symmetrise
    print(f"[2/3] Symmetrising density -> {symm_out}")
    symm_cmd = [
        sys.executable, symm_script,
        "--meta", meta_path,
        "--density", density_path,
        "--coords-file", coords_path,
        "--coords-ang",'True',
        "--out", symm_out
    ]
    if extra_symm_args:
        symm_cmd.extend(extra_symm_args)
    subprocess.run(symm_cmd, check=True)
    if not os.path.exists(symm_out):
        raise RuntimeError("Symmetrisation step failed: missing symmetrised output.")

    # Step 3: Interpolate
    print(f"[3/3] Interpolating -> {interp_out}")
    interp_cmd = [
        sys.executable, interp_script,
        "--meta", meta_path,
        "--density", symm_out,
        "--quad-in", quad_in,
        "--nelec", str(nelec_ref),
        "--plot-y", plot_y,
        "--out", interp_out
    ]
    if extra_interp_args:
        interp_cmd.extend(extra_interp_args)
    subprocess.run(interp_cmd, check=True)
    if not os.path.exists(interp_out):
        raise RuntimeError("Interpolation step failed: missing interpolation output.")

    print("Pipeline complete.")
    return {
        "meta": meta_path,
        "density_exported": density_path,
        "density_symmetrised": symm_out,
        "interpolation_output": interp_out,
        "plot_y": plot_y,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Read and display the contents of a text file (e.g., XSF density)."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="dmc_density_dt01.s002.SpinDensity_u+d.xsf",
        help="Path to the input file"
    )
    parser.add_argument(
        "-f", "--full",
        action="store_true",
        help="Print the entire file instead of a preview"
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse XSF and print a structured summary"
    )
    parser.add_argument(
        "--export",
        metavar="BASE",
        default=None,
        help="Export parsed XSF to interoperable files with given base path (creates *_density.txt, *_meta.json, *_atoms.csv)"
    )
    parser.add_argument(
        "--run-symmetry",
        action="store_true",
        help="After export, run symmetriseDensity_gpu.py with exported files"
    )
    parser.add_argument(
        "--run-interp",
        action="store_true",
        help="After symmetrization, run Interpolate_gpu_cubic.py with exported files"
    )
    parser.add_argument(
        "--symm-script",
        default="symmetriseDensity_gpu.py",
        help="Path to symmetrization script"
    )
    parser.add_argument(
        "--interp-script",
        default="Interpolate_gpu_cubic.py",
        help="Path to interpolation script"
    )
    parser.add_argument(
        "--symm-out",
        default=None,
        help="Output filename for symmetrized density; defaults to <base>_symm.txt"
    )
    args = parser.parse_args()
    # Validate file existence
    if not os.path.isfile(args.filename):
        print(f"Error: File not found or not a file: {args.filename}", file=sys.stderr)
        sys.exit(1)

    try:
        contents = read_file(args.filename)
    except OSError as e:
        print(f"Error reading file {args.filename}: {e}", file=sys.stderr)
        sys.exit(2)

    data = None
    if args.parse or args.export is not None:
        data = parse_xsf(contents)

    if args.parse:
        print_parsed_summary(data)
        if data.grid_shape:
            print(f"Debug: Detected grid shape: {data.grid_shape}")
        if data.density:
            preview_count = min(len(data.density), 10)
            preview_vals = " ".join(f"{v:.6e}" for v in data.density[:preview_count])
            print(f"Debug: Density preview: {preview_vals}{' ...' if len(data.density) > preview_count else ''}")

    if args.export is not None:
        try:
            paths = export_xsf(data, args.export)
            print("Exported files:")
            print(f"  Density file: {paths['density']}")
            print(f"  Metadata file: {paths['meta']}")
            if "atoms" in paths:
                print(f"  Atoms file: {paths['atoms']}")
        except Exception as e:
            print(f"Error during export: {e}", file=sys.stderr)
            sys.exit(3)

        density_path = paths["density"]
        meta_path = paths["meta"]

        if args.run_symmetry:
            symm_out = args.symm_out if args.symm_out else f"{args.export}_symm.txt"
            cmd = [
                sys.executable,
                args.symm_script,
                "--meta", meta_path,
                "--density", density_path,
                "--coords-file", paths['atoms'],
                "--out", symm_out
            ]
            print(f"Running symmetrization script: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print(f"Symmetrized density written to: {symm_out}")
                density_path = symm_out
            except subprocess.CalledProcessError as e:
                print(f"Symmetrization script failed with error: {e}", file=sys.stderr)
                sys.exit(4)

        if args.run_interp:
            interp_out = f"{args.export}_interp.txt"
            cmd = [
                sys.executable,
                args.interp_script,
                "--meta", meta_path,
                "--density", density_path,
                "--out", interp_out
            ]
            print(f"Running interpolation script: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print(f"Interpolated density written to: {interp_out}")
            except subprocess.CalledProcessError as e:
                print(f"Interpolation script failed with error: {e}", file=sys.stderr)
                sys.exit(5)

    if not args.parse and not args.export:
        if args.full:
            # Print entire file
            print(contents, end='')
        else:
            # Print summary preview
            print_summary(contents, args.filename)

if __name__ == "__main__":
    main()


