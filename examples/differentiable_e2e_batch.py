# -*- coding: utf-8 -*-
"""
End-to-end differentiable demo (batch).

Batch with variable atom counts, soft atom_types, and forces/stress in the loss.
"""
import json
import os

import torch
import torch.nn.functional as F
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import (
    DifferentiableMatterSimCalculator,
    sizes_to_batch_index,
)


def main():
    os.environ.setdefault("MATTERSIM_LOGURU_ENQUEUE", "0")
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    si = bulk("Si", "diamond", a=5.43)
    fe = bulk("Fe", "fcc", a=3.6)
    atoms_list = [si, fe]
    sizes = torch.tensor([len(si), len(fe)], device=device)
    batch_index = sizes_to_batch_index(sizes, device=device)

    calc = DifferentiableMatterSimCalculator(device=device)

    baseline_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "baseline_outputs.json"
    )
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    target_energy = torch.tensor(
        [
            baseline["diamond_si"]["outputs"]["energy"],
            baseline["fcc_fe"]["outputs"]["energy"],
        ],
        device=device,
    )
    target_forces = torch.tensor(
        baseline["diamond_si"]["outputs"]["forces"] + baseline["fcc_fe"]["outputs"]["forces"],
        device=device,
    )
    target_stress = torch.tensor(
        [
            baseline["diamond_si"]["outputs"]["stress"],
            baseline["fcc_fe"]["outputs"]["stress"],
        ],
        device=device,
    )

    atomic_numbers = torch.cat(
        [
            torch.tensor(si.get_atomic_numbers(), dtype=torch.long, device=device),
            torch.tensor(fe.get_atomic_numbers(), dtype=torch.long, device=device),
        ],
        dim=0,
    )
    atom_logits = torch.nn.Parameter(
        F.one_hot(atomic_numbers, num_classes=95).float()
        + 0.05 * torch.randn(len(atomic_numbers), 95, device=device)
    )

    positions = torch.tensor(
        si.get_positions().tolist() + fe.get_positions().tolist(),
        dtype=torch.float32,
        device=device,
    )
    positions_param = torch.nn.Parameter(
        positions + 0.02 * torch.randn_like(positions)
    )

    lattice = torch.stack(
        [
            torch.tensor(si.cell.array, dtype=torch.float32, device=device),
            torch.tensor(fe.cell.array, dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    lattice_param = torch.nn.Parameter(
        lattice + 0.01 * torch.randn_like(lattice)
    )

    init_logits = atom_logits.detach().clone()
    init_positions = positions_param.detach().clone()
    init_lattice = lattice_param.detach().clone()

    optimizer = torch.optim.Adam([atom_logits, positions_param, lattice_param], lr=5e-3)

    def per_structure_force_loss(forces):
        split = torch.split(forces, sizes.tolist(), dim=0)
        return torch.tensor([f.pow(2).mean() for f in split], device=device)

    def report(step, loss_value, loss_energy, loss_forces, loss_stress):
        print(f"step {step} loss {loss_value:.6f}")
        print(
            "loss_by_structure",
            f"energy={loss_energy.tolist()}",
            f"forces={loss_forces.tolist()}",
            f"stress={loss_stress.tolist()}",
        )
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
        output = calc.predict_from_batch_tensors(
            batch_atom_types=atom_types,
            lattice=lattice_param,
            positions=positions_param,
            batch_index=batch_index,
            return_forces=True,
            return_stress=True,
            create_graph_forces=True,
            create_graph_stress=True,
            soft_normalize=True,
        )

        energy = output["total_energy"]
        forces = output["forces"]
        stress = output["stresses"]

        loss_energy = (energy - target_energy).pow(2)
        loss_forces = per_structure_force_loss(forces - target_forces)
        loss_stress = (stress - target_stress).pow(2).mean(dim=(1, 2))

        loss = loss_energy.mean() + loss_forces.mean() + 0.1 * loss_stress.mean()

        loss.backward()
        if step == 0 or step == steps - 1:
            report(step, loss.item(), loss_energy, loss_forces, loss_stress)
        optimizer.step()


if __name__ == "__main__":
    main()
