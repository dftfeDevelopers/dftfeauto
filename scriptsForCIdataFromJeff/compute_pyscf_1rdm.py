"""
Read an XYZ file of the form:
  N
  charge=0 multiplicity=1
  H 0. 0. 0.
  F 0.917 0. 0.

Run a PySCF DFT calculation with LDA (VWN) + PW91 correlation (i.e., "lda,pw"),
using per-element NWChem-format basis files named: "{element}_basis_nwchem.bas"
(e.g., h_basis_nwchem.bas, f_basis_nwchem.bas).

Write at the end:
  - overlap matrix S
  - spin-summed 1-RDM (AO basis) for closed-shell: D = 2 * C_occ C_occ^T
    (for open-shell, uses alpha+beta if available)

Outputs: overlap.npy, rdm1.npy (NumPy .npy files)
"""

import re
import numpy as np
from pyscf import gto, dft
from pyscf.gto.basis import parse as nwchem_parse


def read_xyz_with_charge_mult(xyz_path):
    with open(xyz_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    nat = int(lines[0])
    comment = lines[1]

    m = re.search(r"charge\s*=\s*([+-]?\d+)\s+multiplicity\s*=\s*(\d+)", comment, re.I)
    if not m:
        raise ValueError("Line 2 must contain: charge=INT multiplicity=INT")
    charge = int(m.group(1))
    mult = int(m.group(2))

    geom_lines = lines[2:2 + nat]
    if len(geom_lines) != nat:
        raise ValueError("XYZ does not contain the stated number of atoms.")

    atoms = []
    for ln in geom_lines:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Bad geometry line: {ln}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((sym, (x, y, z)))

    return atoms, charge, mult


def load_element_basis_from_files(elements):
    """
    Return a PySCF basis dict: {'H': parsed_basis, 'F': parsed_basis, ...}
    Each element is loaded from: "{element_lower}_basis_nwchem.bas"
    """
    basis = {}
    for el in sorted(set(elements)):
        fn = f"{el.lower()}_basis_nwchem.bas"
        with open(fn, "r") as f:
            txt = f.read()
        # Parse NWChem-style basis text into PySCF internal format
        basis[el] = nwchem_parse(txt)
    return basis


def main():
    xyz_file = "molecule.xyz"

    atoms, charge, mult = read_xyz_with_charge_mult(xyz_file)
    elements = [a[0] for a in atoms]
    basis = load_element_basis_from_files(elements)

    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Angstrom"
    mol.charge = charge
    mol.spin = mult - 1  # PySCF: spin = Nalpha - Nbeta
    mol.basis = basis
    mol.build()

    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    # In PySCF, "lda,pw" corresponds to LDA exchange + PW91 correlation.
    mf.xc = "lda,pw"
    mf.kernel()

    # AO overlap matrix
    S =  mf.get_ovlp()
    # AO density matrix
    dm = mf.make_rdm1(ao_repr=True)

    np.set_printoptions(precision=12, suppress=False, linewidth=200)

    print("\n=== Overlap matrix S (AO) ===")
    np.savetxt("overlapMat_pyscf", S, fmt='%.9f', delimiter=' ')

    print("\n=== Density matrix D (AO) ===")
    np.savetxt("DensityMat_lda_pyscf", dm/2.0, fmt='%.9f', delimiter=' ')

    smat = np.loadtxt("ovlpAO")

    print("\n ===difference in overlap matrices ===\n")
    print(np.max(np.max(np.abs(smat - S))))
    print("\n ======================================\n")
    
    print("SCF energy:", mf.e_tot)
    print("Wrote DensityMat_lda_pyscf and overlapMat_pyscf")


if __name__ == "__main__":
    main()
