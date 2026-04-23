"""
Convert an XYZ file to:
1) coordinates.inp:
   atomic_number atomic_number x_bohr y_bohr z_bohr

2) AtomicCoords:
   Element Id x_ang y_ang z_ang basis_file_name
   where basis_file_name is "{element}_basis.bas" (element is lowercase)
"""

from __future__ import annotations

import argparse
from pathlib import Path

ANGSTROM_TO_BOHR = 1.889726124565062

SYMBOL_TO_Z = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16, "Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,
    "Sc": 21,"Ti": 22,"V": 23, "Cr": 24,"Mn": 25,"Fe": 26,"Co": 27,"Ni": 28,"Cu": 29,"Zn": 30,
    "Ga": 31,"Ge": 32,"As": 33,"Se": 34,"Br": 35,"Kr": 36,
    "Rb": 37,"Sr": 38,
    "Y": 39, "Zr": 40,"Nb": 41,"Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,
    "In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53, "Xe": 54,
    "Cs": 55,"Ba": 56,
    "La": 57,"Ce": 58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,
    "Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,
    "Hf": 72,"Ta": 73,"W": 74, "Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,
    "Tl": 81,"Pb": 82,"Bi": 83,"Po": 84,"At": 85,"Rn": 86,
    "Fr": 87,"Ra": 88,
    "Ac": 89,"Th": 90,"Pa": 91,"U": 92, "Np": 93,"Pu": 94,"Am": 95,"Cm": 96,"Bk": 97,"Cf": 98,
    "Es": 99,"Fm": 100,"Md": 101,"No": 102,"Lr": 103,
    "Rf": 104,"Db": 105,"Sg": 106,"Bh": 107,"Hs": 108,"Mt": 109,"Ds": 110,"Rg": 111,"Cn": 112,
    "Nh": 113,"Fl": 114,"Mc": 115,"Lv": 116,"Ts": 117,"Og": 118,
}

def normalize_symbol(sym: str) -> str:
    sym = sym.strip()
    if not sym:
        return sym
    return sym[0].upper() + sym[1:].lower()

def read_xyz(xyz_path: Path):
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file too short: {xyz_path}")

    natoms = int(lines[0].strip().split()[0])
    atom_lines = [ln for ln in lines[2:] if ln.strip()]
    if len(atom_lines) < natoms:
        raise ValueError(f"Expected {natoms} atom lines but found {len(atom_lines)} in {xyz_path}")

    atoms = []
    for i in range(natoms):
        parts = atom_lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line #{i+1}: '{atom_lines[i]}'")
        sym = normalize_symbol(parts[0])
        x, y, z = map(float, parts[1:4])
        atoms.append((sym, x, y, z))
    return atoms

def main():
    parser = argparse.ArgumentParser(description="Convert molecule.xyz into coordinates.inp and AtomicCoords.")
    parser.add_argument("path", help="Directory containing molecule.xyz")
    args = parser.parse_args()

    folder = Path(args.path).expanduser().resolve()
    xyz_path = folder / "molecule.xyz"
    if not xyz_path.exists():
        raise FileNotFoundError(f"Expected file not found: {xyz_path}")

    atoms = read_xyz(xyz_path)

    coordinates_inp = folder / "coordinates.inp"
    atomic_coords = folder / "AtomicCoords"

    # coordinates.inp (bohrs)
    with coordinates_inp.open("w", encoding="utf-8") as f:
        for sym, x_a, y_a, z_a in atoms:
            Z = SYMBOL_TO_Z.get(sym)
            if Z is None:
                raise KeyError(f"Unknown element symbol '{sym}'. Add it to SYMBOL_TO_Z.")
            x_b = x_a * ANGSTROM_TO_BOHR
            y_b = y_a * ANGSTROM_TO_BOHR
            z_b = z_a * ANGSTROM_TO_BOHR
            f.write(f"{Z:d} {Z:d} {x_b:.10f} {y_b:.10f} {z_b:.10f}\n")

    # AtomicCoords (angstroms) with per-element running Id, but basis file is just "{element}_basis.bas"
    element_counts = {}
    with atomic_coords.open("w", encoding="utf-8") as f:
        for sym, x_a, y_a, z_a in atoms:
            element_counts[sym] = element_counts.get(sym, 0) + 1
            idx = element_counts[sym]
            basis_name = f"{sym.lower()}_basis_invdft.bas"
            f.write(f"{sym} {x_a:.10f} {y_a:.10f} {z_a:.10f} {basis_name}\n")

if __name__ == "__main__":
    main()
