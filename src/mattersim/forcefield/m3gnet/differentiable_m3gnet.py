# -*- coding: utf-8 -*-
"""
微分可能な M3GNet 実装

原子種を連続値 one-hot として扱い、完全に微分可能な forward を提供します。
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch_runstats.scatter import scatter

from mattersim.jit_compile_tools.jit import compile_mode

from .m3gnet import M3Gnet


@compile_mode("script")
class DifferentiableM3Gnet(M3Gnet):
    """
    微分可能な M3GNet

    既存の M3Gnet を継承し、forward メソッドを微分可能に書き換えます。
    """

    def forward_differentiable(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
        soft_normalize: bool = False,
    ) -> torch.Tensor:
        """
        微分可能な forward パス

        Args:
            input: 入力辞書
                - atom_attr: (N, max_z+1) の連続値原子種分布（requires_grad 可能）
                - atom_pos: (N, 3) の Cartesian 座標
                - cell: (1, 3, 3) の格子行列
                - edge_index: (2, E) のエッジインデックス
                - edge_vector: (E, 3) のエッジベクトル
                - edge_length: (E,) のエッジ長
                - three_body_indices: (T, 2) の3体項インデックス
                - num_three_body: 3体項の数
                - num_bonds: エッジ数
                - num_triple_ij: (E, 1) の3体項カウント
                - num_atoms: 原子数
                - num_graphs: グラフ数
                - batch: (N,) のバッチインデックス
            dataset_idx: データセットインデックス
            soft_normalize: True なら atom_attr への勾配を維持する soft normalization

        Returns:
            energies: (num_graphs,) のエネルギー
        """
        # データ抽出
        pos = input["atom_pos"]
        cell = input["cell"]
        atom_attr = input["atom_attr"]  # (N, max_z+1) の連続値
        edge_index = input["edge_index"].long()
        edge_vector = input["edge_vector"]
        edge_length_vec = input["edge_length"]
        three_body_indices = input["three_body_indices"].long()
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        batch = input["batch"]

        # 3体項のインデックスバイアス調整
        if isinstance(num_bonds, int):
            num_bonds_tensor = torch.tensor([num_bonds], device=pos.device)
        else:
            num_bonds_tensor = num_bonds

        cumsum = torch.cumsum(num_bonds_tensor, dim=0) - num_bonds_tensor

        if isinstance(num_three_body, int):
            num_three_body_tensor = torch.tensor([num_three_body], device=pos.device)
        else:
            num_three_body_tensor = num_three_body

        # num_atoms も tensor に変換
        if isinstance(num_atoms, int):
            num_atoms_tensor = torch.tensor([num_atoms], device=pos.device)
        else:
            num_atoms_tensor = num_atoms

        if num_three_body_tensor.sum() > 0:
            index_bias = torch.repeat_interleave(
                cumsum, num_three_body_tensor, dim=0
            ).unsqueeze(-1)
            three_body_indices = three_body_indices + index_bias
        else:
            three_body_indices = three_body_indices

        # エッジ長を (E, 1) に reshape
        edge_length = edge_length_vec.unsqueeze(-1)

        # 3体項の計算
        if three_body_indices.shape[0] > 0:
            vij = edge_vector[three_body_indices[:, 0].clone()]
            vik = edge_vector[three_body_indices[:, 1].clone()]
            rij = edge_length_vec[three_body_indices[:, 0].clone()]
            rik = edge_length_vec[three_body_indices[:, 1].clone()]
            cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik + 1e-8)
            cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
            triple_edge_length = rik.view(-1)
        else:
            # 3体項がない場合のダミー
            triple_edge_length = torch.zeros(0, device=pos.device)
            cos_jik = torch.zeros(0, device=pos.device)

        # 原子埋め込み（連続値 one-hot @ embedding_weight）
        # atom_embedding は MLP(in_dim=max_z+1, out_dims=[units], activation=None, use_bias=False)
        # 内部は単一の Linear 層（LinearLayer）なので、重み行列を直接使用
        # MLP.mlp は Sequential で、最初の層は LinearLayer
        atom_attr_embedded = F.linear(
            atom_attr, self.atom_embedding.mlp[0].linear.weight
        )

        # エッジ特徴量
        edge_attr = self.rbf(edge_length_vec.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)

        # 3体項の基底関数
        if three_body_indices.shape[0] > 0:
            three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))
        else:
            # ダミー
            three_basis = torch.zeros(
                (0, self.sbf.max_n * self.sbf.max_l), device=pos.device
            )

        # メインループ
        for idx, conv in enumerate(self.graph_conv):
            atom_attr_embedded, edge_attr = conv(
                atom_attr_embedded,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds_tensor,
                num_triple_ij,
                num_atoms_tensor,
            )

        # 最終層
        energies_i = self.final(atom_attr_embedded).view(-1)

        # Normalizer 適用
        if soft_normalize:
            # 連続値 atom_attr に対して soft normalization（勾配を維持）
            # atom_attr: (N, max_z+1)
            # shift, scale: (max_z+1,)
            shift = torch.matmul(atom_attr, self.normalizer.shift)  # (N,)
            scale = torch.matmul(atom_attr, self.normalizer.scale)  # (N,)
            energies_i = scale * energies_i + shift
        else:
            # 既存方式（argmax）
            # atom_attr は連続値だが、normalizer は整数原子番号を要求
            # argmax で最も確率の高い原子番号を取得
            atomic_numbers = torch.argmax(atom_attr, dim=1).long()
            energies_i = self.normalizer(energies_i, atomic_numbers)

        # グラフごとに集約
        energies = scatter(energies_i, batch, dim=0, dim_size=num_graphs)

        return energies
