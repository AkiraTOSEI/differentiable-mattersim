# -*- coding: utf-8 -*-
"""
Differentiable MatterSim バッチ処理のテスト
"""
import torch
import torch.nn.functional as F
from ase.build import bulk

from mattersim.forcefield.differentiable_potential import (
    DifferentiableMatterSimCalculator,
    sizes_to_batch_index,
    batch_index_to_sizes,
)


class TestBatchHelpers:
    """ヘルパー関数のテスト"""

    def test_sizes_to_batch_index(self):
        """sizes → batch_index 変換"""
        sizes = torch.tensor([2, 3, 1])
        batch_index = sizes_to_batch_index(sizes)

        expected = torch.tensor([0, 0, 1, 1, 1, 2])
        assert torch.equal(batch_index, expected)

    def test_batch_index_to_sizes(self):
        """batch_index → sizes 変換"""
        batch_index = torch.tensor([0, 0, 1, 1, 1, 2])
        sizes = batch_index_to_sizes(batch_index)

        expected = torch.tensor([2, 3, 1])
        assert torch.equal(sizes, expected)

    def test_roundtrip(self):
        """往復変換"""
        sizes_orig = torch.tensor([5, 3, 7, 2])
        batch_index = sizes_to_batch_index(sizes_orig)
        sizes_reconstructed = batch_index_to_sizes(batch_index)

        assert torch.equal(sizes_orig, sizes_reconstructed)


class TestBatchRegression:
    """バッチ処理の回帰テスト"""

    def test_batch_matches_single(self):
        """バッチ出力と単一出力の一致確認"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # Single で個別に計算
        out_si = calc.forward(si, include_forces=False)
        out_fe = calc.forward(fe, include_forces=False)
        energy_single = torch.tensor([out_si["total_energy"].item(), out_fe["total_energy"].item()])

        # Batch で計算
        atoms_list = [si, fe]
        sizes = torch.tensor([len(si), len(fe)])
        out_batch = calc.forward_batch(atoms_list, sizes=sizes, include_forces=False, soft_normalize=False)
        energy_batch = out_batch["total_energy"]

        # 一致確認
        assert torch.allclose(energy_batch, energy_single, atol=1e-4)

    def test_batch_3structures(self):
        """3構造のバッチ処理"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)
        al = bulk("Al", "fcc", a=4.05)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        atoms_list = [si, fe, al]
        out_batch = calc.forward_batch(atoms_list, include_forces=False)

        # 出力形状確認
        assert out_batch["total_energy"].shape == (3,)

        # 各エネルギーが有限値
        assert torch.all(torch.isfinite(out_batch["total_energy"]))


class TestBatchGradient:
    """バッチ処理の勾配テスト"""

    def test_batch_gradients_positions(self):
        """positions への勾配"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # 入力準備（リーフテンソルとして作成）
        positions_si = torch.tensor(si.get_positions(), dtype=torch.float32)
        positions_fe = torch.tensor(fe.get_positions(), dtype=torch.float32)
        positions = torch.cat([positions_si, positions_fe], dim=0)
        positions.requires_grad_(True)

        sizes = torch.tensor([len(si), len(fe)])

        out = calc.forward_batch([si, fe], positions=positions, sizes=sizes, include_forces=False)
        loss = out["total_energy"].sum()
        loss.backward()

        # 勾配確認
        assert positions.grad is not None
        assert positions.grad.shape == positions.shape
        assert torch.norm(positions.grad) > 0

    def test_batch_gradients_lattice(self):
        """lattice への勾配"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # 格子準備
        lattice_si = torch.tensor(si.cell.array, dtype=torch.float32)
        lattice_fe = torch.tensor(fe.cell.array, dtype=torch.float32)
        lattice = torch.stack([lattice_si, lattice_fe], dim=0)
        lattice.requires_grad_(True)

        sizes = torch.tensor([len(si), len(fe)])

        out = calc.forward_batch([si, fe], lattice=lattice, sizes=sizes, include_forces=False)
        loss = out["total_energy"].sum()
        loss.backward()

        # 勾配確認
        assert lattice.grad is not None
        assert lattice.grad.shape == lattice.shape
        assert torch.norm(lattice.grad) > 0

    def test_batch_gradients_atom_types_soft(self):
        """atom_types への勾配（soft_normalize=True）"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # atom_types 準備（連続分布）
        atomic_numbers_si = torch.tensor(si.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers_fe = torch.tensor(fe.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers = torch.cat([atomic_numbers_si, atomic_numbers_fe], dim=0)

        atom_types = F.one_hot(atomic_numbers, num_classes=95).float()
        atom_types = atom_types + torch.randn_like(atom_types) * 0.01
        atom_types = F.softmax(atom_types, dim=1)
        atom_types.requires_grad_(True)

        sizes = torch.tensor([len(si), len(fe)])

        out = calc.forward_batch(
            [si, fe], atom_types=atom_types, sizes=sizes, include_forces=False, soft_normalize=True
        )
        loss = out["total_energy"].sum()
        loss.backward()

        # 勾配確認
        assert atom_types.grad is not None
        assert atom_types.grad.shape == atom_types.shape
        assert torch.norm(atom_types.grad) > 0

    def test_batch_all_three_gradients(self):
        """atom_types, lattice, positions の3つ全てに勾配"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # atom_types
        atomic_numbers_si = torch.tensor(si.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers_fe = torch.tensor(fe.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers = torch.cat([atomic_numbers_si, atomic_numbers_fe], dim=0)
        atom_types = F.one_hot(atomic_numbers, num_classes=95).float()
        atom_types.requires_grad_(True)

        # positions
        positions_si = torch.tensor(si.get_positions(), dtype=torch.float32)
        positions_fe = torch.tensor(fe.get_positions(), dtype=torch.float32)
        positions = torch.cat([positions_si, positions_fe], dim=0)
        positions.requires_grad_(True)

        # lattice
        lattice_si = torch.tensor(si.cell.array, dtype=torch.float32)
        lattice_fe = torch.tensor(fe.cell.array, dtype=torch.float32)
        lattice = torch.stack([lattice_si, lattice_fe], dim=0)
        lattice.requires_grad_(True)

        sizes = torch.tensor([len(si), len(fe)])

        out = calc.forward_batch(
            [si, fe],
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            sizes=sizes,
            include_forces=False,
            soft_normalize=True,
        )
        loss = out["total_energy"].sum()
        loss.backward()

        # 3つ全てに勾配が付く
        assert atom_types.grad is not None
        assert positions.grad is not None
        assert lattice.grad is not None

        assert torch.norm(atom_types.grad) > 0
        assert torch.norm(positions.grad) > 0
        assert torch.norm(lattice.grad) > 0

    def test_batch_forces_stress_gradients(self):
        """forces/stress loss で勾配が流れる"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        atomic_numbers_si = torch.tensor(si.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers_fe = torch.tensor(fe.get_atomic_numbers(), dtype=torch.long)
        atomic_numbers = torch.cat([atomic_numbers_si, atomic_numbers_fe], dim=0)
        atom_types = F.one_hot(atomic_numbers, num_classes=95).float()
        atom_types.requires_grad_(True)

        positions_si = torch.tensor(si.get_positions(), dtype=torch.float32)
        positions_fe = torch.tensor(fe.get_positions(), dtype=torch.float32)
        positions = torch.cat([positions_si, positions_fe], dim=0)
        positions.requires_grad_(True)

        lattice_si = torch.tensor(si.cell.array, dtype=torch.float32)
        lattice_fe = torch.tensor(fe.cell.array, dtype=torch.float32)
        lattice = torch.stack([lattice_si, lattice_fe], dim=0)
        lattice.requires_grad_(True)

        sizes = torch.tensor([len(si), len(fe)])

        out = calc.forward_batch(
            [si, fe],
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            sizes=sizes,
            return_forces=True,
            return_stress=True,
            create_graph_forces=True,
            create_graph_stress=True,
            soft_normalize=True,
        )
        loss = out["forces"].pow(2).mean() + out["stresses"].pow(2).mean()
        loss.backward()

        assert atom_types.grad is not None
        assert positions.grad is not None
        assert lattice.grad is not None


class TestBatchOptimization:
    """バッチ最適化のデモンストレーション"""

    def test_batch_optimize_positions(self):
        """バッチでの positions 最適化"""
        si = bulk("Si", "diamond", a=5.43)
        fe = bulk("Fe", "fcc", a=3.6)

        calc = DifferentiableMatterSimCalculator(device="cpu")

        # positions に摂動
        positions_si = torch.tensor(si.get_positions(), dtype=torch.float32)
        positions_fe = torch.tensor(fe.get_positions(), dtype=torch.float32)
        positions = torch.cat([positions_si, positions_fe], dim=0)
        positions = positions + torch.randn_like(positions) * 0.02
        positions.requires_grad_(True)
        initial_positions = positions.detach().clone()

        optimizer = torch.optim.Adam([positions], lr=0.01)

        sizes = torch.tensor([len(si), len(fe)])

        initial_energy = None
        for step in range(5):
            optimizer.zero_grad()
            out = calc.forward_batch([si, fe], positions=positions, sizes=sizes, include_forces=False)
            energy = out["total_energy"].sum()

            if initial_energy is None:
                initial_energy = energy.item()

            energy.backward()
            optimizer.step()

        final_energy = energy.item()

        # 勾配が計算され、パラメータが更新されていることを確認
        assert torch.isfinite(torch.tensor(final_energy))
        assert positions.grad is not None
        assert torch.norm(positions.detach() - initial_positions) > 0.0
