#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微分可能な MatterSim のデモ

このスクリプトは、MatterSim を用いて構造を最適化する方法を示します。
入力（原子種、格子、座標）に対して勾配を計算し、torch の optimizer で更新できます。
"""
import torch
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


def demo_gradient_flow():
    """勾配が流れることを確認"""
    print("=" * 60)
    print("デモ1: 勾配の流れを確認")
    print("=" * 60)

    # Diamond Si 構造を作成
    si = bulk("Si", "diamond", a=5.43)
    device = "cpu"

    # 微分可能な Calculator を初期化
    calc = DifferentiableMatterSimCalculator(device=device)

    # 座標を tensor として取得し、requires_grad を設定
    positions = torch.tensor(
        si.get_positions(), dtype=torch.float32, device=device, requires_grad=True
    )

    # エネルギーを計算
    output = calc.forward(si, positions=positions, include_forces=False)
    energy = output["total_energy"]

    print(f"エネルギー: {energy.item():.6f} eV")

    # backward で勾配を計算
    energy.backward()

    print(f"座標の勾配形状: {positions.grad.shape}")
    print(f"勾配のノルム: {torch.norm(positions.grad).item():.6e}")
    print("✓ 勾配が正常に計算されました\n")


def demo_structure_optimization():
    """構造最適化のデモ"""
    print("=" * 60)
    print("デモ2: 構造最適化")
    print("=" * 60)

    # Diamond Si 構造を作成し、少しランダムに摂動
    si = bulk("Si", "diamond", a=5.43)
    device = "cpu"

    calc = DifferentiableMatterSimCalculator(device=device)

    # 座標を摂動
    positions = torch.tensor(si.get_positions(), dtype=torch.float32, device=device)
    positions = positions + torch.randn_like(positions) * 0.05  # 0.05 Å の摂動
    positions.requires_grad_(True)

    # Adam オプティマイザを使用
    optimizer = torch.optim.Adam([positions], lr=0.02)

    print(f"初期摂動: ±0.05 Å")
    print(f"学習率: 0.02")
    print(f"最適化ステップ数: 20\n")

    print("ステップ | エネルギー (eV) | 勾配ノルム")
    print("-" * 50)

    for step in range(20):
        optimizer.zero_grad()

        # エネルギーを計算
        output = calc.forward(si, positions=positions, include_forces=False)
        energy = output["total_energy"]

        # backward
        energy.backward()

        # 勾配ノルムを記録
        grad_norm = torch.norm(positions.grad).item()

        # 最適化ステップ
        optimizer.step()

        if step % 5 == 0 or step == 19:
            print(f"{step:7d} | {energy.item():14.6f} | {grad_norm:.6e}")

    print("\n✓ 最適化が完了しました")


def demo_lattice_optimization():
    """格子定数の最適化デモ"""
    print("=" * 60)
    print("デモ3: 格子定数の最適化")
    print("=" * 60)

    # Diamond Si 構造（格子定数を少しずらす）
    si = bulk("Si", "diamond", a=5.5)  # 正しい値は 5.43
    device = "cpu"

    calc = DifferentiableMatterSimCalculator(device=device)

    # 格子を tensor として取得
    lattice = torch.tensor(si.cell.array, dtype=torch.float32, device=device)
    lattice.requires_grad_(True)

    # SGD オプティマイザ
    optimizer = torch.optim.SGD([lattice], lr=0.01)

    print(f"初期格子定数: a={5.5:.3f} Å （正解: 5.43 Å）")
    print(f"学習率: 0.01")
    print(f"最適化ステップ数: 30\n")

    print("ステップ | エネルギー (eV) | 格子定数 a (Å)")
    print("-" * 50)

    for step in range(30):
        optimizer.zero_grad()

        # エネルギーを計算
        output = calc.forward(si, lattice=lattice, include_forces=False)
        energy = output["total_energy"]

        # backward
        energy.backward()

        # 最適化ステップ
        optimizer.step()

        # 現在の格子定数
        current_a = lattice[0, 0].item()

        if step % 5 == 0 or step == 29:
            print(f"{step:7d} | {energy.item():14.6f} | {current_a:.6f}")

    final_a = lattice[0, 0].item()
    print(f"\n最終格子定数: a={final_a:.6f} Å")
    print(f"正解値との差: {abs(final_a - 5.43):.6f} Å")
    print("✓ 格子定数の最適化が完了しました")


if __name__ == "__main__":
    print("\n微分可能な MatterSim のデモ\n")

    # デモ1: 勾配の流れ
    demo_gradient_flow()

    # デモ2: 構造最適化
    demo_structure_optimization()

    # デモ3: 格子定数の最適化
    demo_lattice_optimization()

    print("\n" + "=" * 60)
    print("全てのデモが完了しました！")
    print("=" * 60)
