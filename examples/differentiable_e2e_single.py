# -*- coding: utf-8 -*-
"""
End-to-end differentiable demo (single structure).

Includes atom_types (soft), lattice, and positions optimization with
forces/stress in the loss (2nd-order gradients).
"""
import json
import os

import torch
import torch.nn.functional as F
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


def main():
    os.environ.setdefault("MATTERSIM_LOGURU_ENQUEUE", "0")
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    si = bulk("Si", "diamond", a=5.43)
    calc = DifferentiableMatterSimCalculator(device=device)

    baseline_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "baseline_outputs.json"
    )
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    target = baseline["diamond_si"]["outputs"]
    target_energy = torch.tensor(target["energy"], device=device)
    target_forces = torch.tensor(target["forces"], device=device)
    target_stress = torch.tensor(target["stress"], device=device)

    atomic_numbers = torch.tensor(si.get_atomic_numbers(), dtype=torch.long, device=device)
    atom_logits = torch.nn.Parameter(
        F.one_hot(atomic_numbers, num_classes=95).float() + 0.05 * torch.randn(len(si), 95, device=device)
    )
    positions_param = torch.nn.Parameter(
        torch.tensor(si.get_positions(), dtype=torch.float32, device=device)
        + 0.02 * torch.randn(len(si), 3, device=device)
    )
    lattice_param = torch.nn.Parameter(
        torch.tensor(si.cell.array, dtype=torch.float32, device=device)
        + 0.01 * torch.randn(3, 3, device=device)
    )

    init_logits = atom_logits.detach().clone()
    init_positions = positions_param.detach().clone()
    init_lattice = lattice_param.detach().clone()

    optimizer = torch.optim.Adam([atom_logits, positions_param, lattice_param], lr=5e-3)

    def report(step, loss_value):
        print(f"step {step} loss {loss_value:.6f}")
        print(
            "grad_norms",
            f"logits={atom_logits.grad.norm():.3e}",
            f"positions={positions_param.grad.norm():.3e}",
            f"lattice={lattice_param.grad.norm():.3e}",
        )
        print(
            "param_change_norms",
            f"logits={(atom_logits - init_logits).norm():.3e}",
            f"positions={(positions_param - init_positions).norm():.3e}",
            f"lattice={(lattice_param - init_lattice).norm():.3e}",
        )

    steps = 5
    for step in range(steps):
        optimizer.zero_grad()

        atom_types = F.softmax(atom_logits, dim=1)
        output = calc.predict_from_tensors(
            atom_types=atom_types,
            lattice=lattice_param,
            positions=positions_param,
            return_forces=True,
            return_stress=True,
            create_graph_forces=True,
            create_graph_stress=True,
            soft_normalize=True,
        )

        energy = output["total_energy"].squeeze()
        forces = output["forces"]
        stress = output["stresses"][0]

        loss = (
            (energy - target_energy).pow(2)
            + (forces - target_forces).pow(2).mean()
            + 0.1 * (stress - target_stress).pow(2).mean()
        )

        loss.backward()
        if step == 0 or step == steps - 1:
            report(step, loss.item())
        optimizer.step()


if __name__ == "__main__":
    main()
