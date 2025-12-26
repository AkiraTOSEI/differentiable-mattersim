# -*- coding: utf-8 -*-
"""
微分可能な Graph 変換

固定トポロジー方式で、近傍リストを初期化時に構築し、
エッジベクトル・エッジ長のみを torch で再計算します。
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from ase import Atoms

from .convertor import compute_threebody_indices, get_fixed_radius_bonding


class DifferentiableGraphBuilder:
    """
    微分可能な Graph 構築クラス

    初期化時に近傍リストのトポロジーを固定し、
    forward 時にエッジベクトルとエッジ長を torch で再計算します。
    """

    def __init__(
        self,
        initial_atoms: Union[Atoms, List[Atoms]],
        twobody_cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        device: str = "cpu",
    ):
        """
        Args:
            initial_atoms: 初期構造（トポロジー構築用）
                - Atoms: 単一構造
                - List[Atoms]: バッチ（複数構造）
            twobody_cutoff: 2体カットオフ半径
            threebody_cutoff: 3体カットオフ半径
            device: torch デバイス
        """
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff
        self.device = device

        # バッチモードか単一モードかを判定
        if isinstance(initial_atoms, list):
            self._init_batch(initial_atoms, twobody_cutoff, threebody_cutoff, device)
        else:
            self._init_single(initial_atoms, twobody_cutoff, threebody_cutoff, device)

    def _init_single(self, initial_atoms: Atoms, twobody_cutoff: float, threebody_cutoff: float, device: str):
        """単一構造の初期化（既存実装）"""
        self.is_batch = False
        self.num_graphs = 1

        # 初期構造で近傍リストを構築（numpy/pymatgen 使用 OK）
        (
            sent_index,
            receive_index,
            shift_vectors,
            distances,
        ) = get_fixed_radius_bonding(initial_atoms, twobody_cutoff, pbc=True)

        # トポロジーを固定
        self.edge_index = torch.from_numpy(
            np.array([sent_index, receive_index])
        ).long().to(device)
        self.pbc_offsets = torch.from_numpy(shift_vectors).float().to(device)
        self.num_bonds = len(sent_index)

        # 3体項のインデックスも固定
        if self.num_bonds > 0:
            (
                triple_bond_index,
                n_triple_ij,
                n_triple_i,
                n_triple_s,
            ) = compute_threebody_indices(
                bond_atom_indices=self.edge_index.cpu().numpy().transpose(1, 0),
                bond_length=distances,
                n_atoms=len(initial_atoms),
                atomic_number=initial_atoms.get_atomic_numbers(),
                threebody_cutoff=threebody_cutoff,
            )
            self.three_body_indices = torch.from_numpy(triple_bond_index).long().to(device)
            self.num_triple_ij = torch.from_numpy(n_triple_ij).long().unsqueeze(-1).to(device)
            self.num_three_body = self.three_body_indices.shape[0]
        else:
            self.three_body_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
            self.num_triple_ij = torch.zeros((self.num_bonds, 1), dtype=torch.long, device=device)
            self.num_three_body = 0

    def _init_batch(self, atoms_list: List[Atoms], twobody_cutoff: float, threebody_cutoff: float, device: str):
        """バッチ構造の初期化"""
        self.is_batch = True
        self.num_graphs = len(atoms_list)

        # 各構造の原子数を記録
        self.sizes = [len(atoms) for atoms in atoms_list]
        self.cumsum_sizes = np.cumsum([0] + self.sizes)

        # 各構造ごとに近傍リストを構築し、連結
        all_sent = []
        all_receive = []
        all_shifts = []
        all_distances = []
        all_triple_indices = []
        all_triple_ij = []

        for i, atoms in enumerate(atoms_list):
            offset = self.cumsum_sizes[i]

            # 近傍リスト構築
            sent, receive, shifts, dists = get_fixed_radius_bonding(
                atoms, twobody_cutoff, pbc=True
            )

            # オフセットを加算して連結
            all_sent.append(sent + offset)
            all_receive.append(receive + offset)
            all_shifts.append(shifts)
            all_distances.append(dists)

            # 3体項インデックス
            if len(sent) > 0:
                triple_idx, n_triple_ij, _, _ = compute_threebody_indices(
                    bond_atom_indices=np.array([sent, receive]).T,
                    bond_length=dists,
                    n_atoms=len(atoms),
                    atomic_number=atoms.get_atomic_numbers(),
                    threebody_cutoff=threebody_cutoff,
                )
                # 3体項インデックスにもオフセットを加算（エッジインデックス空間でのオフセット）
                edge_offset = len(np.concatenate(all_distances[:-1])) if i > 0 else 0
                all_triple_indices.append(triple_idx + edge_offset)
                all_triple_ij.append(n_triple_ij)
            else:
                all_triple_ij.append(np.array([]))

        # 連結
        if len(all_sent) > 0:
            sent_index = np.concatenate(all_sent)
            receive_index = np.concatenate(all_receive)
            shift_vectors = np.concatenate(all_shifts)
            distances = np.concatenate(all_distances)
        else:
            sent_index = np.array([])
            receive_index = np.array([])
            shift_vectors = np.array([]).reshape(0, 3)
            distances = np.array([])

        # Tensor に変換
        self.edge_index = torch.from_numpy(
            np.array([sent_index, receive_index])
        ).long().to(device)
        self.pbc_offsets = torch.from_numpy(shift_vectors).float().to(device)
        self.num_bonds = len(sent_index)

        # 3体項
        if len(all_triple_indices) > 0 and any(len(t) > 0 for t in all_triple_indices):
            triple_bond_index = np.concatenate([t for t in all_triple_indices if len(t) > 0])
            n_triple_ij_concat = np.concatenate(all_triple_ij)
            self.three_body_indices = torch.from_numpy(triple_bond_index).long().to(device)
            self.num_triple_ij = torch.from_numpy(n_triple_ij_concat).long().unsqueeze(-1).to(device)
            self.num_three_body = self.three_body_indices.shape[0]
        else:
            self.three_body_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
            self.num_triple_ij = torch.zeros((self.num_bonds, 1), dtype=torch.long, device=device)
            self.num_three_body = 0

    def forward(
        self,
        atom_pos: torch.Tensor,
        cell: torch.Tensor,
        num_atoms: int = None,
        batch_index: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        微分可能な Graph 構築

        Args:
            atom_pos: (N, 3) または (sum(N_i), 3) の Cartesian 座標（requires_grad 可能）
            cell: (1, 3, 3), (3, 3), または (nb_graphs, 3, 3) の格子行列（requires_grad 可能）
            num_atoms: 原子数（単一構造の場合のみ、バッチの場合は不要）
            batch_index: (sum(N_i),) 各原子の所属グラフID（バッチの場合のみ）

        Returns:
            graph_data: エッジベクトル、エッジ長などを含む辞書
        """
        # cell の次元調整
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)  # (3, 3) -> (1, 3, 3)

        # batch_index の準備
        if batch_index is None:
            # 単一構造モード
            if num_atoms is None:
                num_atoms = len(atom_pos)
            atoms_batch = torch.zeros(num_atoms, dtype=torch.long, device=self.device)
        else:
            # バッチモード
            atoms_batch = batch_index

        # エッジがどのグラフに属するかを計算
        edge_batch = atoms_batch[self.edge_index[0]]

        # エッジベクトルを torch で計算
        # edge_vector = pos[i] - (pos[j] + pbc_offset @ cell)
        pbc_offset_cart = torch.einsum("bi, bij->bj", self.pbc_offsets, cell[edge_batch])
        edge_vector = atom_pos[self.edge_index[0]] - (
            atom_pos[self.edge_index[1]] + pbc_offset_cart
        )

        # エッジ長を計算
        edge_length = torch.linalg.norm(edge_vector, dim=1)

        return {
            "edge_index": self.edge_index,
            "edge_vector": edge_vector,
            "edge_length": edge_length,
            "pbc_offsets": self.pbc_offsets,
            "three_body_indices": self.three_body_indices,
            "num_three_body": self.num_three_body,
            "num_triple_ij": self.num_triple_ij,
            "num_bonds": self.num_bonds,
        }
