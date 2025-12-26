# -*- coding: utf-8 -*-
"""
Differentiable MatterSim のテスト

回帰テスト（既存 API との一致確認）と勾配テスト（backward の動作確認）を含みます。
"""
import json
import os
import unittest

import torch
from ase.build import bulk

from mattersim.forcefield import MatterSimCalculator
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator


class TestDifferentiableRegression(unittest.TestCase):
    """既存 MatterSim との数値一致確認"""

    @classmethod
    def setUpClass(cls):
        """Baseline データを読み込み"""
        baseline_path = os.path.join(
            os.path.dirname(__file__), "data", "baseline_outputs.json"
        )
        with open(baseline_path, "r") as f:
            cls.baseline_data = json.load(f)

        # デバイス設定
        cls.device = "cpu"  # 再現性のため CPU を使用

    def test_diamond_si_energy(self):
        """Diamond Si のエネルギーが既存と一致"""
        si = bulk("Si", "diamond", a=5.43)

        # 新 API
        calc_new = DifferentiableMatterSimCalculator(device=self.device)
        output = calc_new.forward(si, include_forces=False, include_stresses=False)
        energy_new = output["total_energy"].item()

        # Baseline と比較
        energy_baseline = self.baseline_data["diamond_si"]["outputs"]["energy"]

        self.assertAlmostEqual(
            energy_new, energy_baseline, places=4,
            msg=f"Energy mismatch: {energy_new} vs {energy_baseline}"
        )

    def test_diamond_si_forces(self):
        """Diamond Si の力が既存と一致"""
        si = bulk("Si", "diamond", a=5.43)

        # 新 API
        calc_new = DifferentiableMatterSimCalculator(device=self.device)
        output = calc_new.forward(si, return_forces=True, return_stress=False)
        forces_new = output["forces"].detach().cpu().numpy()

        # Baseline と比較
        forces_baseline = self.baseline_data["diamond_si"]["outputs"]["forces"]

        # shape 確認
        self.assertEqual(forces_new.shape, (len(si), 3))

        # 値の確認（力は非常に小さいので絶対誤差で比較）
        max_diff = abs(forces_new - forces_baseline).max()
        self.assertLess(
            max_diff, 1e-4,
            msg=f"Forces max diff: {max_diff}"
        )

    def test_fcc_fe_energy(self):
        """FCC-Fe のエネルギーが既存と一致"""
        fe = bulk("Fe", "fcc", a=3.6)

        # 新 API
        calc_new = DifferentiableMatterSimCalculator(device=self.device)
        output = calc_new.forward(fe, include_forces=False, include_stresses=False)
        energy_new = output["total_energy"].item()

        # Baseline と比較
        energy_baseline = self.baseline_data["fcc_fe"]["outputs"]["energy"]

        self.assertAlmostEqual(
            energy_new, energy_baseline, places=4,
            msg=f"Energy mismatch: {energy_new} vs {energy_baseline}"
        )

    def test_stresses_match_baseline(self):
        """Diamond Si / FCC-Fe の応力が既存と一致"""
        calc_new = DifferentiableMatterSimCalculator(device=self.device)

        si = bulk("Si", "diamond", a=5.43)
        out_si = calc_new.forward(si, return_forces=True, return_stress=True)
        stress_si = out_si["stresses"].detach().cpu().numpy()[0]
        stress_si_base = self.baseline_data["diamond_si"]["outputs"]["stress"]

        fe = bulk("Fe", "fcc", a=3.6)
        out_fe = calc_new.forward(fe, return_forces=True, return_stress=True)
        stress_fe = out_fe["stresses"].detach().cpu().numpy()[0]
        stress_fe_base = self.baseline_data["fcc_fe"]["outputs"]["stress"]

        self.assertEqual(stress_si.shape, (3, 3))
        self.assertEqual(stress_fe.shape, (3, 3))
        # Baseline stress is in eV/A^3 (ASE); differentiable output is in GPa.
        import numpy as np
        from ase.units import GPa
        self.assertLess(abs(stress_si - (np.array(stress_si_base) / GPa)).max(), 1e-4)
        self.assertLess(abs(stress_fe - (np.array(stress_fe_base) / GPa)).max(), 1e-4)


class TestDifferentiableGradient(unittest.TestCase):
    """勾配の流れを確認"""

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"

    def test_energy_grad_wrt_positions(self):
        """座標に対する勾配が計算できる"""
        si = bulk("Si", "diamond", a=5.43)

        calc = DifferentiableMatterSimCalculator(device=self.device)

        # 座標を requires_grad=True にして forward
        positions = torch.tensor(
            si.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True
        )

        output = calc.forward(si, positions=positions, include_forces=False)
        energy = output["total_energy"]

        # backward
        energy.backward()

        # 勾配が存在するか確認
        self.assertIsNotNone(positions.grad)
        self.assertEqual(positions.grad.shape, positions.shape)

        # 勾配が非ゼロか確認
        grad_norm = torch.norm(positions.grad)
        self.assertGreater(grad_norm, 0.0)

    def test_energy_grad_wrt_lattice(self):
        """格子に対する勾配が計算できる"""
        si = bulk("Si", "diamond", a=5.43)

        calc = DifferentiableMatterSimCalculator(device=self.device)

        # 格子を requires_grad=True にして forward
        lattice = torch.tensor(
            si.cell.array, dtype=torch.float32, device=self.device, requires_grad=True
        )

        output = calc.forward(si, lattice=lattice, include_forces=False)
        energy = output["total_energy"]

        # backward
        energy.backward()

        # 勾配が存在するか確認
        self.assertIsNotNone(lattice.grad)
        self.assertEqual(lattice.grad.shape, lattice.shape)

        # 勾配が非ゼロか確認
        grad_norm = torch.norm(lattice.grad)
        self.assertGreater(grad_norm, 0.0)

    def test_energy_grad_wrt_atom_types(self):
        """原子種分布に対する勾配が計算できる"""
        si = bulk("Si", "diamond", a=5.43)

        calc = DifferentiableMatterSimCalculator(device=self.device)

        # 原子種を連続値にして requires_grad=True
        atomic_numbers = torch.tensor(
            si.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        atom_types = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=95
        ).float()
        atom_types.requires_grad_(True)

        output = calc.forward(si, atom_types=atom_types, include_forces=False)
        energy = output["total_energy"]

        # backward
        energy.backward()

        # 勾配が存在するか確認
        self.assertIsNotNone(atom_types.grad)
        self.assertEqual(atom_types.grad.shape, atom_types.shape)

    def test_forces_via_autograd(self):
        """torch.autograd.grad で力が計算できる"""
        si = bulk("Si", "diamond", a=5.43)

        calc = DifferentiableMatterSimCalculator(device=self.device)
        output = calc.forward(si, return_forces=True, return_stress=False)

        # 力が計算されているか確認
        self.assertIn("forces", output)
        forces = output["forces"]
        self.assertEqual(forces.shape, (len(si), 3))

    def test_energy_loss_grads_all_inputs(self):
        """energy loss で 3入力に勾配が付く"""
        si = bulk("Si", "diamond", a=5.43)
        calc = DifferentiableMatterSimCalculator(device=self.device)

        atomic_numbers = torch.tensor(
            si.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        atom_types = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=95
        ).float()
        atom_types.requires_grad_(True)

        positions = torch.tensor(
            si.get_positions(), dtype=torch.float32, device=self.device
        )
        positions.requires_grad_(True)

        lattice = torch.tensor(
            si.cell.array, dtype=torch.float32, device=self.device
        )
        lattice.requires_grad_(True)

        output = calc.forward(
            si,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            return_forces=False,
            return_stress=False,
            soft_normalize=True,
        )
        loss = output["total_energy"].sum()
        loss.backward()

        self.assertIsNotNone(atom_types.grad)
        self.assertIsNotNone(positions.grad)
        self.assertIsNotNone(lattice.grad)

    def test_forces_loss_grads_all_inputs(self):
        """forces loss で 3入力に勾配が付く"""
        si = bulk("Si", "diamond", a=5.43)
        calc = DifferentiableMatterSimCalculator(device=self.device)

        atomic_numbers = torch.tensor(
            si.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        atom_types = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=95
        ).float()
        atom_types.requires_grad_(True)

        positions = torch.tensor(
            si.get_positions(), dtype=torch.float32, device=self.device
        )
        positions.requires_grad_(True)

        lattice = torch.tensor(
            si.cell.array, dtype=torch.float32, device=self.device
        )
        lattice.requires_grad_(True)

        output = calc.forward(
            si,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            return_forces=True,
            return_stress=False,
            create_graph_forces=True,
            soft_normalize=True,
        )
        loss = output["forces"].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(atom_types.grad)
        self.assertIsNotNone(positions.grad)
        self.assertIsNotNone(lattice.grad)

    def test_stress_loss_grads_all_inputs(self):
        """stress loss で 3入力に勾配が付く"""
        si = bulk("Si", "diamond", a=5.43)
        calc = DifferentiableMatterSimCalculator(device=self.device)

        atomic_numbers = torch.tensor(
            si.get_atomic_numbers(), dtype=torch.long, device=self.device
        )
        atom_types = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=95
        ).float()
        atom_types.requires_grad_(True)

        positions = torch.tensor(
            si.get_positions(), dtype=torch.float32, device=self.device
        )
        positions.requires_grad_(True)

        lattice = torch.tensor(
            si.cell.array, dtype=torch.float32, device=self.device
        )
        lattice.requires_grad_(True)

        output = calc.forward(
            si,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            return_forces=False,
            return_stress=True,
            create_graph_stress=True,
            soft_normalize=True,
        )
        loss = output["stresses"].pow(2).mean()
        loss.backward()

        self.assertIsNotNone(atom_types.grad)
        self.assertIsNotNone(positions.grad)
        self.assertIsNotNone(lattice.grad)


class TestDifferentiableOptimization(unittest.TestCase):
    """最適化のデモンストレーション"""

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"

    def test_optimize_positions(self):
        """座標の最適化が動作する"""
        si = bulk("Si", "diamond", a=5.43)

        calc = DifferentiableMatterSimCalculator(device=self.device)

        # 座標を少しランダムに摂動
        positions = torch.tensor(
            si.get_positions(), dtype=torch.float32, device=self.device
        )
        positions = positions + torch.randn_like(positions) * 0.01
        positions.requires_grad_(True)

        # オプティマイザ
        optimizer = torch.optim.Adam([positions], lr=0.01)

        # 数ステップ最適化
        initial_energy = None
        initial_positions = positions.detach().clone()
        for step in range(5):
            optimizer.zero_grad()
            output = calc.forward(si, positions=positions, include_forces=False)
            energy = output["total_energy"]

            if initial_energy is None:
                initial_energy = energy.item()

            energy.backward()
            optimizer.step()

        final_energy = energy.item()

        # 勾配が計算され、パラメータが更新されていることを確認
        self.assertTrue(torch.isfinite(torch.tensor(final_energy)))
        self.assertIsNotNone(positions.grad)
        self.assertGreater(torch.norm(positions.detach() - initial_positions), 0.0)


if __name__ == "__main__":
    unittest.main()
