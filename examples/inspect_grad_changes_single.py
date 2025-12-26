# -*- coding: utf-8 -*-
"""
Inspect gradient-driven changes for energy/forces/stress losses (single structure).
"""
import os

import torch
import torch.nn.functional as F
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


def run_case(label, calc, si, atom_logits, positions, lattice, loss_type):
    optimizer = torch.optim.SGD([atom_logits, positions, lattice], lr=1e-3)

    atom_types = F.softmax(atom_logits, dim=1)
    output = calc.predict_from_tensors(
        atom_types=atom_types,
        lattice=lattice,
        positions=positions,
        return_forces=(loss_type == "forces"),
        return_stress=(loss_type == "stress"),
        create_graph_forces=(loss_type == "forces"),
        create_graph_stress=(loss_type == "stress"),
        soft_normalize=True,
    )

    if loss_type == "energy":
        loss = output["total_energy"].sum()
    elif loss_type == "forces":
        loss = output["forces"].pow(2).mean()
    elif loss_type == "stress":
        loss = output["stresses"].pow(2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss_before = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    atom_types_after = F.softmax(atom_logits, dim=1)
    output_after = calc.predict_from_tensors(
        atom_types=atom_types_after,
        lattice=lattice,
        positions=positions,
        return_forces=(loss_type == "forces"),
        return_stress=(loss_type == "stress"),
        create_graph_forces=(loss_type == "forces"),
        create_graph_stress=(loss_type == "stress"),
        soft_normalize=True,
    )
    if loss_type == "energy":
        loss_after = output_after["total_energy"].sum().item()
    elif loss_type == "forces":
        loss_after = output_after["forces"].pow(2).mean().item()
    else:
        loss_after = output_after["stresses"].pow(2).mean().item()

    print(f"\n[{label}] loss_type={loss_type} loss_before={loss_before:.6f} loss_after={loss_after:.6f}")
    print(
        "grad_norms",
        f"atom_types={atom_logits.grad.norm():.3e}",
        f"positions={positions.grad.norm():.3e}",
        f"lattice={lattice.grad.norm():.3e}",
    )
    print(
        "param_delta_norms",
        f"atom_types={(atom_logits - run_case.atom_logits_init).norm():.3e}",
        f"positions={(positions - run_case.positions_init).norm():.3e}",
        f"lattice={(lattice - run_case.lattice_init).norm():.3e}",
    )


def init_params(si, device):
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
    return atom_logits, positions, lattice


def main():
    os.environ.setdefault("MATTERSIM_LOGURU_ENQUEUE", "0")
    device = "cpu"

    si = bulk("Si", "diamond", a=5.43)
    calc = DifferentiableMatterSimCalculator(device=device)

    for loss_type in ("energy", "forces", "stress"):
        atom_logits, positions, lattice = init_params(si, device)
        run_case.atom_logits_init = atom_logits.detach().clone()
        run_case.positions_init = positions.detach().clone()
        run_case.lattice_init = lattice.detach().clone()
        run_case("single", calc, si, atom_logits, positions, lattice, loss_type)


if __name__ == "__main__":
    main()
