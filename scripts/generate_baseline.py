#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 生成スクリプト

現状の MatterSim を使って Diamond Si と FCC-Fe の出力を取得し、
テスト用の baseline データとして保存します。
"""
import json
import os

import numpy as np
import torch
from ase.build import bulk

from mattersim.forcefield import MatterSimCalculator


def generate_baseline():
    """Baseline データを生成"""
    os.environ.setdefault("MATTERSIM_LOGURU_ENQUEUE", "0")
    # CPU で実行（再現性のため）
    device = "cpu"
    calc = MatterSimCalculator(device=device)

    baseline_data = {}

    # Diamond Si
    print("Generating baseline for Diamond Si...")
    si = bulk("Si", "diamond", a=5.43)
    si.calc = calc

    energy_si = si.get_potential_energy()
    forces_si = si.get_forces()
    stress_si = si.get_stress(voigt=False)

    baseline_data["diamond_si"] = {
        "structure": {
            "symbols": si.get_chemical_symbols(),
            "positions": si.get_positions().tolist(),
            "cell": si.cell.array.tolist(),
            "pbc": si.pbc.tolist(),
        },
        "outputs": {
            "energy": float(energy_si),
            "forces": forces_si.tolist(),
            "stress": stress_si.tolist(),
        },
    }

    print(f"  Energy: {energy_si:.6f} eV")
    print(f"  Forces shape: {forces_si.shape}")
    print(f"  Stress shape: {stress_si.shape}")

    # FCC-Fe
    print("\nGenerating baseline for FCC-Fe...")
    # FCC-Fe の典型的な格子定数（実験値に近い値）
    fe = bulk("Fe", "fcc", a=3.6)
    fe.calc = calc

    energy_fe = fe.get_potential_energy()
    forces_fe = fe.get_forces()
    stress_fe = fe.get_stress(voigt=False)

    baseline_data["fcc_fe"] = {
        "structure": {
            "symbols": fe.get_chemical_symbols(),
            "positions": fe.get_positions().tolist(),
            "cell": fe.cell.array.tolist(),
            "pbc": fe.pbc.tolist(),
        },
        "outputs": {
            "energy": float(energy_fe),
            "forces": forces_fe.tolist(),
            "stress": stress_fe.tolist(),
        },
    }

    print(f"  Energy: {energy_fe:.6f} eV")
    print(f"  Forces shape: {forces_fe.shape}")
    print(f"  Stress shape: {stress_fe.shape}")

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_outputs.json")

    with open(output_path, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\nBaseline data saved to: {output_path}")

    return baseline_data


if __name__ == "__main__":
    baseline_data = generate_baseline()
