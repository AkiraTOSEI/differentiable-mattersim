#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微分可能な MatterSim バッチ処理のデモ

このスクリプトは以下を示します：
(D1) atom_types, lattice, positions を同時に最適化して変化することを表示
(D2) atom_types が one-hot でなく連続分布でも更新されることを表示
(D3) 上記を batch（diamond Si + fcc Fe など2結晶）で実行
"""
import torch
import torch.nn.functional as F
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


def demo1_single_joint_optimization():
    """
    (D1-single) atom_types, lattice, positions を同時に最適化
    3つとも値が変化することを print で確認
    """
    print("=" * 70)
    print("デモ1: Single - 3変数同時最適化 (atom_types, lattice, positions)")
    print("=" * 70)

    si = bulk("Si", "diamond", a=5.5)  # 少しずらした格子定数
    calc = DifferentiableMatterSimCalculator(device="cpu")

    # atom_types: one-hot + 小摂動（連続分布化）
    atomic_numbers = torch.tensor(si.get_atomic_numbers(), dtype=torch.long)
    atom_types = F.one_hot(atomic_numbers, num_classes=95).float()
    atom_types = atom_types + torch.randn_like(atom_types) * 0.01
    atom_types = F.softmax(atom_types, dim=1)  # 確率分布に正規化
    atom_types.requires_grad_(True)

    # positions: 摂動
    positions = torch.tensor(si.get_positions(), dtype=torch.float32)
    positions = positions + torch.randn_like(positions) * 0.03
    positions.requires_grad_(True)

    # lattice: 少しずらす
    lattice = torch.tensor(si.cell.array, dtype=torch.float32)
    lattice.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': [atom_types], 'lr': 0.001},
        {'params': [positions], 'lr': 0.02},
        {'params': [lattice], 'lr': 0.005},
    ])

    print("\n初期値:")
    print(f"  atom_types[0, 13:16]: {atom_types[0, 13:16].detach()}")  # Si周辺の確率
    print(f"  positions[0]: {positions[0].detach()}")
    print(f"  lattice[0,0]: {lattice[0, 0].item():.6f}")

    energies = []
    for step in range(20):
        optimizer.zero_grad()
        output = calc.forward(
            si, atom_types=atom_types, positions=positions, lattice=lattice,
            include_forces=False
        )
        energy = output["total_energy"]
        energies.append(energy.item())
        energy.backward()

        # 勾配ノルムを記録
        if step == 0:
            grad_atom = torch.norm(atom_types.grad).item()
            grad_pos = torch.norm(positions.grad).item()
            grad_lat = torch.norm(lattice.grad).item()
            print(f"\nStep 0 勾配ノルム:")
            print(f"  atom_types: {grad_atom:.6e}")
            print(f"  positions:  {grad_pos:.6e}")
            print(f"  lattice:    {grad_lat:.6e}")

        optimizer.step()

    print(f"\n最終値 (step={step}):")
    print(f"  atom_types[0, 13:16]: {atom_types[0, 13:16].detach()}")
    print(f"  positions[0]: {positions[0].detach()}")
    print(f"  lattice[0,0]: {lattice[0, 0].item():.6f}")

    print(f"\nエネルギー変化:")
    print(f"  初期: {energies[0]:.6f} eV")
    print(f"  最終: {energies[-1]:.6f} eV")
    print(f"  変化: {energies[-1] - energies[0]:.6f} eV")
    print("✓ 3変数とも変化を確認\n")


def demo2_single_continuous_atom_types():
    """
    (D2-single) atom_types を連続分布として扱い、勾配更新
    """
    print("=" * 70)
    print("デモ2: Single - 非one-hot atom_types の勾配更新")
    print("=" * 70)

    si = bulk("Si", "diamond", a=5.43)
    calc = DifferentiableMatterSimCalculator(device="cpu")

    # logits をパラメータ化（より自然な連続分布）
    # Si (原子番号14) を中心に分布を作る
    logits = torch.zeros(len(si), 95)
    logits[:, 14] = 5.0  # Si に高い確率
    logits += torch.randn_like(logits) * 0.5  # ノイズ
    logits.requires_grad_(True)

    optimizer = torch.optim.SGD([logits], lr=0.1)

    print("\n初期 logits[0, 13:16]:", logits[0, 13:16].detach())
    print("初期 atom_types[0, 13:16]:", F.softmax(logits, dim=1)[0, 13:16].detach())

    energies = []
    for step in range(10):
        optimizer.zero_grad()
        atom_types = F.softmax(logits, dim=1)  # (N, 95) 確率分布

        output = calc.forward(si, atom_types=atom_types, include_forces=False)
        energy = output["total_energy"]
        energies.append(energy.item())

        # 損失: エネルギー + regularization（Si を保つための項）
        target_si = F.one_hot(torch.tensor([14] * len(si)), num_classes=95).float()
        loss = energy + 0.1 * F.mse_loss(atom_types, target_si)

        loss.backward()
        optimizer.step()

    print("\n最終 logits[0, 13:16]:", logits[0, 13:16].detach())
    print("最終 atom_types[0, 13:16]:", F.softmax(logits, dim=1)[0, 13:16].detach())

    print(f"\nエネルギー変化:")
    print(f"  初期: {energies[0]:.6f} eV")
    print(f"  最終: {energies[-1]:.6f} eV")
    print("✓ 連続分布で勾配更新を確認\n")


def demo3_batch_joint_optimization():
    """
    (D3: D1のbatch版) 2つの結晶（Si, Fe）をバッチ処理で同時最適化
    """
    print("=" * 70)
    print("デモ3: Batch - 2結晶の3変数同時最適化")
    print("=" * 70)

    si = bulk("Si", "diamond", a=5.5)  # 少しずらす
    fe = bulk("Fe", "fcc", a=3.7)  # 少しずらす
    atoms_list = [si, fe]

    calc = DifferentiableMatterSimCalculator(device="cpu")

    # バッチ入力を準備
    sizes = torch.tensor([len(si), len(fe)])

    # atom_types: concatenate
    atom_types_si = F.one_hot(torch.tensor(si.get_atomic_numbers()), 95).float()
    atom_types_fe = F.one_hot(torch.tensor(fe.get_atomic_numbers()), 95).float()
    atom_types = torch.cat([atom_types_si, atom_types_fe], dim=0)
    atom_types = atom_types + torch.randn_like(atom_types) * 0.01
    atom_types = F.softmax(atom_types, dim=1)
    atom_types.requires_grad_(True)

    # positions: concatenate
    positions = torch.cat([
        torch.tensor(si.get_positions(), dtype=torch.float32),
        torch.tensor(fe.get_positions(), dtype=torch.float32)
    ], dim=0)
    positions = positions + torch.randn_like(positions) * 0.02
    positions.requires_grad_(True)

    # lattice: stack
    lattice = torch.stack([
        torch.tensor(si.cell.array, dtype=torch.float32),
        torch.tensor(fe.cell.array, dtype=torch.float32)
    ], dim=0)
    lattice.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': [atom_types], 'lr': 0.001},
        {'params': [positions], 'lr': 0.02},
        {'params': [lattice], 'lr': 0.005},
    ])

    print("\n初期値:")
    print(f"  lattice[0, 0, 0] (Si): {lattice[0, 0, 0].item():.6f}")
    print(f"  lattice[1, 0, 0] (Fe): {lattice[1, 0, 0].item():.6f}")

    batch_energies = []
    for step in range(20):
        optimizer.zero_grad()
        output = calc.forward_batch(
            atoms_list,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            sizes=sizes,
            include_forces=False,
            soft_normalize=True,
        )
        energies = output["total_energy"]  # (2,)
        batch_energies.append(energies.detach().clone())
        loss = energies.sum()
        loss.backward()

        # 勾配ノルムを記録
        if step == 0:
            grad_atom = torch.norm(atom_types.grad).item()
            grad_pos = torch.norm(positions.grad).item()
            grad_lat = torch.norm(lattice.grad).item()
            print(f"\nStep 0 勾配ノルム:")
            print(f"  atom_types: {grad_atom:.6e}")
            print(f"  positions:  {grad_pos:.6e}")
            print(f"  lattice:    {grad_lat:.6e}")

        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step:2d}: energies = [Si: {energies[0].item():.6f}, Fe: {energies[1].item():.6f}] eV")

    print(f"\n最終値:")
    print(f"  lattice[0, 0, 0] (Si): {lattice[0, 0, 0].item():.6f}")
    print(f"  lattice[1, 0, 0] (Fe): {lattice[1, 0, 0].item():.6f}")

    print(f"\nエネルギー変化:")
    print(f"  Si 初期: {batch_energies[0][0].item():.6f} eV, 最終: {batch_energies[-1][0].item():.6f} eV")
    print(f"  Fe 初期: {batch_energies[0][1].item():.6f} eV, 最終: {batch_energies[-1][1].item():.6f} eV")
    print("✓ バッチで3変数が同時更新されることを確認\n")


def demo4_batch_continuous_atom_types():
    """
    (D3: D2のbatch版) バッチ + 連続分布 atom_types
    """
    print("=" * 70)
    print("デモ4: Batch - 非one-hot atom_types の勾配更新")
    print("=" * 70)

    si = bulk("Si", "diamond", a=5.43)
    fe = bulk("Fe", "fcc", a=3.6)
    atoms_list = [si, fe]

    calc = DifferentiableMatterSimCalculator(device="cpu")

    # logits をパラメータ化
    logits_si = torch.zeros(len(si), 95)
    logits_si[:, 14] = 5.0  # Si
    logits_fe = torch.zeros(len(fe), 95)
    logits_fe[:, 26] = 5.0  # Fe
    logits = torch.cat([logits_si, logits_fe], dim=0)
    logits += torch.randn_like(logits) * 0.5
    logits.requires_grad_(True)

    optimizer = torch.optim.SGD([logits], lr=0.1)

    sizes = torch.tensor([len(si), len(fe)])

    print("\n初期 logits[0, 13:16] (Si):", logits[0, 13:16].detach())
    print("初期 logits[2, 25:28] (Fe):", logits[2, 25:28].detach())

    energies = []
    for step in range(10):
        optimizer.zero_grad()
        atom_types = F.softmax(logits, dim=1)

        output = calc.forward_batch(atoms_list, atom_types=atom_types, sizes=sizes,
                                     include_forces=False, soft_normalize=True)
        energy_batch = output["total_energy"]
        energies.append(energy_batch.detach().clone())

        # 損失: エネルギー + regularization
        atomic_numbers = torch.cat([
            torch.tensor(si.get_atomic_numbers()),
            torch.tensor(fe.get_atomic_numbers())
        ], dim=0)
        target = F.one_hot(atomic_numbers, num_classes=95).float()
        loss = energy_batch.sum() + 0.1 * F.mse_loss(atom_types, target)

        loss.backward()
        optimizer.step()

    print("\n最終 logits[0, 13:16] (Si):", logits[0, 13:16].detach())
    print("最終 logits[2, 25:28] (Fe):", logits[2, 25:28].detach())

    print(f"\nエネルギー変化:")
    print(f"  Si 初期: {energies[0][0].item():.6f} eV, 最終: {energies[-1][0].item():.6f} eV")
    print(f"  Fe 初期: {energies[0][1].item():.6f} eV, 最終: {energies[-1][1].item():.6f} eV")
    print("✓ バッチ + 連続分布で勾配更新を確認\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" 微分可能な MatterSim バッチ処理のデモ")
    print("=" * 70 + "\n")

    # デモ1: Single - 3変数同時最適化
    demo1_single_joint_optimization()

    # デモ2: Single - 連続分布 atom_types
    demo2_single_continuous_atom_types()

    # デモ3: Batch - 3変数同時最適化
    demo3_batch_joint_optimization()

    # デモ4: Batch - 連続分布 atom_types
    demo4_batch_continuous_atom_types()

    print("=" * 70)
    print("全てのデモが完了しました！")
    print("=" * 70)
