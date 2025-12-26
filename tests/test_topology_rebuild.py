# -*- coding: utf-8 -*-
"""
Topology rebuild test with gradient updates.
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
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


def test_rebuild_topology_every_step_and_serialize():
    os.environ["MATTERSIM_LOGURU_ENQUEUE"] = "0"
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    si = bulk("Si", "diamond", a=5.43)
    calc = DifferentiableMatterSimCalculator(device=device)

    atomic_numbers = torch.tensor(si.get_atomic_numbers(), dtype=torch.long, device=device)
    atom_logits = torch.nn.Parameter(
        F.one_hot(atomic_numbers, num_classes=95).float()
        + 0.01 * torch.randn(len(si), 95, device=device)
    )
    positions = torch.nn.Parameter(
        torch.tensor(si.get_positions(), dtype=torch.float32, device=device)
        + 0.01 * torch.randn(len(si), 3, device=device)
    )
    lattice = torch.nn.Parameter(
        torch.tensor(si.cell.array, dtype=torch.float32, device=device)
        + 0.01 * torch.randn(3, 3, device=device)
    )

    optimizer = torch.optim.SGD([atom_logits, positions, lattice], lr=1e-3)

    for _ in range(100):
        optimizer.zero_grad()

        atom_types = F.softmax(atom_logits, dim=1)
        atom_numbers = torch.argmax(atom_types, dim=1).detach().cpu().numpy()
        atoms_step = si.copy()
        atoms_step.set_atomic_numbers(atom_numbers)
        atoms_step.set_positions(positions.detach().cpu().numpy())
        atoms_step.set_cell(lattice.detach().cpu().numpy())
        atoms_step.pbc = True

        output = calc.forward(
            atoms_step,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            return_forces=False,
            return_stress=False,
            soft_normalize=True,
        )

        loss = output["total_energy"].sum()
        loss.backward()
        optimizer.step()

    atom_numbers = torch.argmax(F.softmax(atom_logits, dim=1), dim=1).detach().cpu().numpy()
    atoms_final = si.copy()
    atoms_final.set_atomic_numbers(atom_numbers)
    atoms_final.set_positions(positions.detach().cpu().numpy())
    atoms_final.set_cell(lattice.detach().cpu().numpy())
    atoms_final.pbc = True

    matter_calc = MatterSimCalculator(device=device)
    atoms_final.calc = matter_calc
    energy_before = atoms_final.get_potential_energy()
    forces_before = atoms_final.get_forces()
    stress_before = atoms_final.get_stress(voigt=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "final.extxyz")
        ase_write(path, atoms_final, format="extxyz")
        atoms_reload = ase_read(path)

    atoms_reload.calc = matter_calc
    energy_after = atoms_reload.get_potential_energy()
    forces_after = atoms_reload.get_forces()
    stress_after = atoms_reload.get_stress(voigt=False)

    assert np.isfinite(energy_before)
    assert np.isfinite(energy_after)
    assert abs(energy_before - energy_after) < 1e-6
    assert np.max(np.abs(forces_before - forces_after)) < 1e-6
    assert np.max(np.abs(stress_before - stress_after)) < 1e-6
