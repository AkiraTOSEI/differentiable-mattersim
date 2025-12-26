# -*- coding: utf-8 -*-
"""
Batch topology rebuild test with gradient updates.
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from ase.build import bulk
from ase.io import read as ase_read
from ase.io import write as ase_write

from mattersim.forcefield import MatterSimCalculator
from mattersim.forcefield.differentiable_potential import (
    DifferentiableMatterSimCalculator,
    sizes_to_batch_index,
)


def test_rebuild_topology_every_step_and_serialize_batch():
    os.environ["MATTERSIM_LOGURU_ENQUEUE"] = "0"
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    si = bulk("Si", "diamond", a=5.43)
    fe = bulk("Fe", "fcc", a=3.6)
    atoms_list = [si, fe]
    sizes = torch.tensor([len(si), len(fe)], device=device)
    batch_index = sizes_to_batch_index(sizes, device=device)

    atomic_numbers = torch.cat(
        [
            torch.tensor(si.get_atomic_numbers(), dtype=torch.long, device=device),
            torch.tensor(fe.get_atomic_numbers(), dtype=torch.long, device=device),
        ],
        dim=0,
    )
    atom_logits = torch.nn.Parameter(
        F.one_hot(atomic_numbers, num_classes=95).float()
        + 0.01 * torch.randn(len(atomic_numbers), 95, device=device)
    )
    positions = torch.nn.Parameter(
        torch.tensor(
            si.get_positions().tolist() + fe.get_positions().tolist(),
            dtype=torch.float32,
            device=device,
        )
        + 0.01 * torch.randn(int(sizes.sum().item()), 3, device=device)
    )
    lattice = torch.nn.Parameter(
        torch.stack(
            [
                torch.tensor(si.cell.array, dtype=torch.float32, device=device),
                torch.tensor(fe.cell.array, dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        + 0.01 * torch.randn(2, 3, 3, device=device)
    )

    optimizer = torch.optim.SGD([atom_logits, positions, lattice], lr=1e-3)

    for _ in range(100):
        optimizer.zero_grad()

        atom_types = F.softmax(atom_logits, dim=1)
        atom_numbers = torch.argmax(atom_types, dim=1).detach().cpu().numpy()

        atoms_step = []
        offset = 0
        for i, size in enumerate(sizes.tolist()):
            start = offset
            end = offset + size
            offset = end
            atoms_i = atoms_list[i].copy()
            atoms_i.set_atomic_numbers(atom_numbers[start:end])
            atoms_i.set_positions(positions.detach().cpu().numpy()[start:end])
            atoms_i.set_cell(lattice.detach().cpu().numpy()[i])
            atoms_i.pbc = True
            atoms_step.append(atoms_i)

        calc = DifferentiableMatterSimCalculator(device=device)
        output = calc.forward_batch(
            atoms_step,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            batch_index=batch_index,
            return_forces=False,
            return_stress=False,
            soft_normalize=True,
        )

        loss = output["total_energy"].sum()
        loss.backward()
        optimizer.step()

    atom_numbers = torch.argmax(F.softmax(atom_logits, dim=1), dim=1).detach().cpu().numpy()
    atoms_final = []
    offset = 0
    for i, size in enumerate(sizes.tolist()):
        start = offset
        end = offset + size
        offset = end
        atoms_i = atoms_list[i].copy()
        atoms_i.set_atomic_numbers(atom_numbers[start:end])
        atoms_i.set_positions(positions.detach().cpu().numpy()[start:end])
        atoms_i.set_cell(lattice.detach().cpu().numpy()[i])
        atoms_i.pbc = True
        atoms_final.append(atoms_i)

    matter_calc = MatterSimCalculator(device=device)
    energies_before = []
    forces_before = []
    stresses_before = []
    for atoms in atoms_final:
        atoms.calc = matter_calc
        energies_before.append(atoms.get_potential_energy())
        forces_before.append(atoms.get_forces())
        stresses_before.append(atoms.get_stress(voigt=False))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "final.extxyz")
        ase_write(path, atoms_final, format="extxyz")
        atoms_reload = ase_read(path, index=":")

    energies_after = []
    forces_after = []
    stresses_after = []
    for atoms in atoms_reload:
        atoms.calc = matter_calc
        energies_after.append(atoms.get_potential_energy())
        forces_after.append(atoms.get_forces())
        stresses_after.append(atoms.get_stress(voigt=False))

    energies_before = np.array(energies_before)
    energies_after = np.array(energies_after)

    assert np.isfinite(energies_before).all()
    assert np.isfinite(energies_after).all()
    assert np.max(np.abs(energies_before - energies_after)) < 1e-6

    for f_before, f_after in zip(forces_before, forces_after):
        assert np.max(np.abs(f_before - f_after)) < 1e-6
    for s_before, s_after in zip(stresses_before, stresses_after):
        assert np.max(np.abs(s_before - s_after)) < 1e-6
