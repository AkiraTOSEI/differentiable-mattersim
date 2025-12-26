# -*- coding: utf-8 -*-
"""
微分可能な MatterSim Potential

ASE Atoms から微分可能な経路でエネルギー・力・応力を計算します。
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ase import Atoms
from ase.units import GPa

from mattersim.datasets.utils.differentiable_convertor import DifferentiableGraphBuilder
from mattersim.forcefield.m3gnet.differentiable_m3gnet import DifferentiableM3Gnet
from mattersim.forcefield.potential import Potential


def sizes_to_batch_index(sizes: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    sizes を batch_index に変換

    Args:
        sizes: (nb_graphs,) 各グラフの原子数
        device: torch デバイス

    Returns:
        batch_index: (sum(sizes),) 各原子の所属グラフID

    Example:
        sizes = [2, 3, 1]
        → batch_index = [0, 0, 1, 1, 1, 2]
    """
    return torch.repeat_interleave(
        torch.arange(len(sizes), device=device), sizes
    )


def batch_index_to_sizes(batch_index: torch.Tensor) -> torch.Tensor:
    """
    batch_index を sizes に変換

    Args:
        batch_index: (sum(N_i),) 各原子の所属グラフID

    Returns:
        sizes: (nb_graphs,) 各グラフの原子数

    Example:
        batch_index = [0, 0, 1, 1, 1, 2]
        → sizes = [2, 3, 1]
    """
    return torch.bincount(batch_index)


class DifferentiableMatterSimCalculator:
    """
    微分可能な MatterSim Calculator

    ASE Atoms を受け取り、微分可能な経路でエネルギー・力・応力を計算します。
    """

    def __init__(
        self,
        potential: Potential = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        twobody_cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        **kwargs,
    ):
        """
        Args:
            potential: 事前学習済み Potential（None の場合は自動ロード）
            device: torch デバイス
            twobody_cutoff: 2体カットオフ半径
            threebody_cutoff: 3体カットオフ半径
        """
        if potential is None:
            self.potential = Potential.from_checkpoint(device=device, **kwargs)
        else:
            self.potential = potential

        self.device = device
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff

        # DifferentiableM3Gnet として扱う
        # 既存の M3Gnet モデルから重みをコピー
        self.diff_model = DifferentiableM3Gnet(
            num_blocks=self.potential.model.model_args.get("num_blocks", 4),
            units=self.potential.model.model_args.get("units", 128),
            max_l=self.potential.model.model_args.get("max_l", 4),
            max_n=self.potential.model.model_args.get("max_n", 4),
            cutoff=self.potential.model.model_args.get("cutoff", 5.0),
            device=device,
            max_z=self.potential.model.model_args.get("max_z", 94),
            threebody_cutoff=self.potential.model.model_args.get("threebody_cutoff", 4.0),
        )
        # 既存モデルの重みをコピー
        self.diff_model.load_state_dict(self.potential.model.state_dict())
        self.diff_model.to(device)  # デバイスに移動
        self.diff_model.eval()

        self.graph_builder = None  # 初回呼び出し時に初期化

    def _resolve_return_flags(
        self,
        include_forces: bool,
        include_stresses: bool,
        return_forces: Optional[bool],
        return_stress: Optional[bool],
    ) -> Tuple[bool, bool]:
        if return_forces is None:
            return_forces = include_forces
        if return_stress is None:
            return_stress = include_stresses
        return return_forces, return_stress

    def _atoms_from_tensors(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        positions_are_fractional: bool,
    ) -> Tuple[Atoms, torch.Tensor]:
        if positions_are_fractional:
            positions = torch.einsum("bi,ij->bj", positions, lattice)

        atomic_numbers = torch.argmax(atom_types, dim=1).detach().cpu().numpy()
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions.detach().cpu().numpy(),
            cell=lattice.detach().cpu().numpy(),
            pbc=True,
        )
        return atoms, positions

    def forward(
        self,
        atoms: Atoms,
        atom_types: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
        include_forces: bool = False,
        include_stresses: bool = False,
        return_forces: Optional[bool] = None,
        return_stress: Optional[bool] = None,
        create_graph_forces: bool = False,
        create_graph_stress: bool = False,
        soft_normalize: bool = False,
        positions_are_fractional: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        微分可能な forward パス

        Args:
            atoms: ASE Atoms オブジェクト（トポロジー構築用）
            atom_types: (N, max_z+1) の原子種分布（None の場合は atoms から取得）
            positions: (N, 3) の Cartesian 座標（None の場合は atoms から取得）
            lattice: (3, 3) の格子行列（None の場合は atoms から取得）
            include_forces: 力を計算するか
            include_stresses: 応力を計算するか

        Returns:
            output: エネルギー、力、応力を含む辞書
        """
        # Graph builder を初期化（毎回再初期化：構造が変わる可能性があるため）
        self.graph_builder = DifferentiableGraphBuilder(
            initial_atoms=atoms,
            twobody_cutoff=self.twobody_cutoff,
            threebody_cutoff=self.threebody_cutoff,
            device=self.device,
        )

        num_atoms = len(atoms)

        return_forces, return_stress = self._resolve_return_flags(
            include_forces, include_stresses, return_forces, return_stress
        )

        # 原子種分布を準備
        if atom_types is None:
            atomic_numbers = torch.tensor(
                atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
            )
            # one-hot エンコード
            atom_types = F.one_hot(
                atomic_numbers, num_classes=self.diff_model.max_z + 1
            ).float()
        else:
            atom_types = atom_types.to(self.device)

        # 格子を準備
        if lattice is None:
            lattice = torch.tensor(
                atoms.cell.array, dtype=torch.float32, device=self.device
            )
        else:
            lattice = lattice.to(self.device)

        # 座標を準備
        if positions is None:
            positions = torch.tensor(
                atoms.get_positions(), dtype=torch.float32, device=self.device
            )
        else:
            positions = positions.to(self.device)

        if positions_are_fractional:
            positions = torch.einsum("bi,ij->bj", positions, lattice)

        if lattice.dim() == 2:
            lattice_batch = lattice.unsqueeze(0)  # (3, 3) -> (1, 3, 3)
        else:
            lattice_batch = lattice

        # requires_grad を設定
        if not positions.requires_grad and (
            return_forces or (return_stress and create_graph_stress)
        ):
            positions.requires_grad_(True)

        strain = torch.zeros_like(lattice_batch, device=self.device)
        volume = torch.linalg.det(lattice_batch)

        if return_stress:
            strain.requires_grad_(True)
            lattice_batch = torch.matmul(
                lattice_batch,
                (torch.eye(3, device=self.device)[None, ...] + strain),
            )
            strain_augment = torch.repeat_interleave(
                strain, torch.tensor([num_atoms], device=self.device), dim=0
            )
            positions = torch.einsum(
                "bi, bij -> bj",
                positions,
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(lattice_batch)

        # Graph データを構築
        graph_data = self.graph_builder.forward(
            atom_pos=positions,
            cell=lattice_batch,
            num_atoms=num_atoms,
        )

        # 入力辞書を構築
        input_dict = {
            "atom_attr": atom_types,
            "atom_pos": positions,
            "cell": lattice_batch,
            "edge_index": graph_data["edge_index"],
            "edge_vector": graph_data["edge_vector"],
            "edge_length": graph_data["edge_length"],
            "pbc_offsets": graph_data["pbc_offsets"],
            "three_body_indices": graph_data["three_body_indices"],
            "num_three_body": graph_data["num_three_body"],
            "num_bonds": graph_data["num_bonds"],
            "num_triple_ij": graph_data["num_triple_ij"],
            "num_atoms": num_atoms,
            "num_graphs": 1,
            "batch": torch.zeros(num_atoms, dtype=torch.long, device=self.device),
        }

        # エネルギー計算
        energies = self.diff_model.forward_differentiable(
            input_dict, soft_normalize=soft_normalize
        )

        output = {"total_energy": energies}

        # 力の計算
        if return_forces and not return_stress:
            grad_outputs = [torch.ones_like(energies)]
            grad = torch.autograd.grad(
                outputs=[energies],
                inputs=[positions],
                grad_outputs=grad_outputs,
                create_graph=create_graph_forces,
                retain_graph=create_graph_forces,
            )
            force_grad = grad[0]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                output["forces"] = forces

        # 力と応力の計算
        if return_forces and return_stress:
            grad_outputs = [torch.ones_like(energies)]
            create_graph = create_graph_forces or create_graph_stress
            grad = torch.autograd.grad(
                outputs=[energies],
                inputs=[positions, strain],
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            force_grad = grad[0]
            stress_grad = grad[1]

            if force_grad is not None:
                forces = torch.neg(force_grad)
                output["forces"] = forces

            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad / GPa
                output["stresses"] = stresses

        # 応力のみの計算
        if return_stress and not return_forces:
            grad_outputs = [torch.ones_like(energies)]
            grad = torch.autograd.grad(
                outputs=[energies],
                inputs=[strain],
                grad_outputs=grad_outputs,
                create_graph=create_graph_stress,
                retain_graph=create_graph_stress,
            )
            stress_grad = grad[0]
            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad / GPa
                output["stresses"] = stresses

        return output

    def predict_from_tensors(
        self,
        atom_types: torch.Tensor,
        lattice: torch.Tensor,
        positions: torch.Tensor,
        return_energy: bool = True,
        return_forces: bool = False,
        return_stress: bool = False,
        create_graph_forces: bool = False,
        create_graph_stress: bool = False,
        soft_normalize: bool = True,
        positions_are_fractional: bool = False,
    ) -> Dict[str, torch.Tensor]:
        atoms, positions = self._atoms_from_tensors(
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            positions_are_fractional=positions_are_fractional,
        )
        output = self.forward(
            atoms,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            return_forces=return_forces,
            return_stress=return_stress,
            create_graph_forces=create_graph_forces,
            create_graph_stress=create_graph_stress,
            soft_normalize=soft_normalize,
        )
        if return_energy:
            return output
        output.pop("total_energy", None)
        return output

    def forward_batch(
        self,
        atoms_list: List[Atoms],
        atom_types: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        sizes: Optional[torch.Tensor] = None,
        include_forces: bool = False,
        include_stresses: bool = False,
        return_forces: Optional[bool] = None,
        return_stress: Optional[bool] = None,
        create_graph_forces: bool = False,
        create_graph_stress: bool = False,
        soft_normalize: bool = True,
        positions_are_fractional: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        バッチ処理版 forward

        Args:
            atoms_list: ASE Atoms のリスト（トポロジー構築用、len=nb_graphs）
            atom_types: (sum(N_i), max_z+1) 原子種分布（連続値OK）
            positions: (sum(N_i), 3) Cartesian 座標
            lattice: (nb_graphs, 3, 3) 格子行列
            batch_index: (sum(N_i),) 各原子の所属グラフID（0, 1, 2, ...）
            sizes: (nb_graphs,) 各グラフの原子数（batch_index の代替）
            include_forces: 力を計算するか
            include_stresses: 応力を計算するか
            soft_normalize: True なら argmax せずに soft normalization

        Returns:
            {
                "total_energy": (nb_graphs,),
                "forces": (sum(N_i), 3) if include_forces,
                "stresses": (nb_graphs, 3, 3) if include_stresses
            }

        Note:
            batch_index と sizes のどちらかを指定。両方 None の場合は sizes を自動計算。
        """
        nb_graphs = len(atoms_list)
        return_forces, return_stress = self._resolve_return_flags(
            include_forces, include_stresses, return_forces, return_stress
        )

        # batch_index の準備
        if batch_index is None:
            if sizes is None:
                # atoms_list から sizes を計算
                sizes = torch.tensor([len(atoms) for atoms in atoms_list], dtype=torch.long, device=self.device)
            batch_index = sizes_to_batch_index(sizes, device=self.device)

        # sizes の準備（batch_index から計算）
        if sizes is None:
            sizes = batch_index_to_sizes(batch_index)

        total_atoms = int(sizes.sum().item())

        # Graph builder を初期化（初回のみ、またはバッチの場合は再初期化）
        if not hasattr(self, 'graph_builder_batch') or self.graph_builder_batch is None:
            self.graph_builder_batch = DifferentiableGraphBuilder(
                initial_atoms=atoms_list,
                twobody_cutoff=self.twobody_cutoff,
                threebody_cutoff=self.threebody_cutoff,
                device=self.device,
            )

        # 原子種分布を準備
        if atom_types is None:
            atomic_numbers_list = [torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
                                    for atoms in atoms_list]
            atomic_numbers = torch.cat(atomic_numbers_list, dim=0)
            # one-hot エンコード
            atom_types = F.one_hot(
                atomic_numbers, num_classes=self.diff_model.max_z + 1
            ).float()
        else:
            atom_types = atom_types.to(self.device)

        # 座標を準備
        if positions is None:
            positions_list = [torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
                              for atoms in atoms_list]
            positions = torch.cat(positions_list, dim=0)
        else:
            positions = positions.to(self.device)

        # 格子を準備
        if lattice is None:
            lattice_list = [torch.tensor(atoms.cell.array, dtype=torch.float32, device=self.device)
                            for atoms in atoms_list]
            lattice = torch.stack(lattice_list, dim=0)  # (nb_graphs, 3, 3)
        else:
            lattice = lattice.to(self.device)

        if lattice.dim() == 2:
            lattice = lattice.unsqueeze(0)  # (3, 3) -> (1, 3, 3)

        # requires_grad を設定
        if not positions.requires_grad and (
            return_forces or (return_stress and create_graph_stress)
        ):
            positions.requires_grad_(True)

        if positions_are_fractional:
            positions = torch.einsum("bi,bij->bj", positions, lattice[batch_index])

        strain = torch.zeros_like(lattice, device=self.device)
        volume = torch.linalg.det(lattice)

        if return_stress:
            strain.requires_grad_(True)
            lattice = torch.matmul(
                lattice,
                (torch.eye(3, device=self.device)[None, ...] + strain),
            )
            strain_augment = strain[batch_index]
            positions = torch.einsum(
                "bi, bij -> bj",
                positions,
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(lattice)

        # Graph データを構築
        graph_data = self.graph_builder_batch.forward(
            atom_pos=positions,
            cell=lattice,
            batch_index=batch_index,
        )

        # 入力辞書を構築
        input_dict = {
            "atom_attr": atom_types,
            "atom_pos": positions,
            "cell": lattice,
            "edge_index": graph_data["edge_index"],
            "edge_vector": graph_data["edge_vector"],
            "edge_length": graph_data["edge_length"],
            "pbc_offsets": graph_data["pbc_offsets"],
            "three_body_indices": graph_data["three_body_indices"],
            "num_three_body": graph_data["num_three_body"],
            "num_bonds": graph_data["num_bonds"],
            "num_triple_ij": graph_data["num_triple_ij"],
            "num_atoms": sizes,
            "num_graphs": nb_graphs,
            "batch": batch_index,
        }

        # エネルギー計算
        energies = self.diff_model.forward_differentiable(
            input_dict, soft_normalize=soft_normalize
        )

        output = {"total_energy": energies}

        # 力の計算（簡略版：stresses なし）
        if return_forces and not return_stress:
            grad_outputs = [torch.ones_like(energies)]
            grad = torch.autograd.grad(
                outputs=[energies.sum()],  # scalar にするため sum
                inputs=[positions],
                grad_outputs=None,
                create_graph=create_graph_forces,
                retain_graph=create_graph_forces,
            )
            force_grad = grad[0]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                output["forces"] = forces

        if return_forces and return_stress:
            create_graph = create_graph_forces or create_graph_stress
            grad = torch.autograd.grad(
                outputs=[energies.sum()],
                inputs=[positions, strain],
                grad_outputs=None,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            force_grad = grad[0]
            stress_grad = grad[1]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                output["forces"] = forces
            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad / GPa
                output["stresses"] = stresses

        if return_stress and not return_forces:
            grad = torch.autograd.grad(
                outputs=[energies.sum()],
                inputs=[strain],
                grad_outputs=None,
                create_graph=create_graph_stress,
                retain_graph=create_graph_stress,
            )
            stress_grad = grad[0]
            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad / GPa
                output["stresses"] = stresses

        return output

    def predict_from_batch_tensors(
        self,
        batch_atom_types: torch.Tensor,
        lattice: torch.Tensor,
        positions: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        sizes: Optional[torch.Tensor] = None,
        return_energy: bool = True,
        return_forces: bool = False,
        return_stress: bool = False,
        create_graph_forces: bool = False,
        create_graph_stress: bool = False,
        soft_normalize: bool = True,
        positions_are_fractional: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if batch_index is None:
            if sizes is None:
                raise ValueError("sizes or batch_index is required for batch tensors.")
            batch_index = sizes_to_batch_index(sizes, device=positions.device)
        if sizes is None:
            sizes = batch_index_to_sizes(batch_index)

        if positions_are_fractional:
            positions = torch.einsum("bi,bij->bj", positions, lattice[batch_index])

        atoms_list = []
        offset = 0
        for i, size in enumerate(sizes.tolist()):
            start = offset
            end = offset + size
            offset = end
            atom_types_i = batch_atom_types[start:end]
            positions_i = positions[start:end]
            lattice_i = lattice[i]
            atoms_i, _ = self._atoms_from_tensors(
                atom_types=atom_types_i,
                positions=positions_i,
                lattice=lattice_i,
                positions_are_fractional=False,
            )
            atoms_list.append(atoms_i)

        output = self.forward_batch(
            atoms_list,
            atom_types=batch_atom_types,
            positions=positions,
            lattice=lattice,
            batch_index=batch_index,
            sizes=sizes,
            return_forces=return_forces,
            return_stress=return_stress,
            create_graph_forces=create_graph_forces,
            create_graph_stress=create_graph_stress,
            soft_normalize=soft_normalize,
            positions_are_fractional=False,
        )

        if return_energy:
            return output
        output.pop("total_energy", None)
        return output
