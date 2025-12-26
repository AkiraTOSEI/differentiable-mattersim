# -*- coding: utf-8 -*-
"""
Torch ベースの幾何演算ユーティリティ

微分可能な格子変換、座標変換などを提供します。
"""
import torch


def lattice_params_to_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    格子パラメータ (a, b, c, alpha, beta, gamma) を 3x3 行列に変換

    Args:
        params: (6,) または (..., 6) の tensor
                (a, b, c, alpha, beta, gamma) in (Å, Å, Å, deg, deg, deg)

    Returns:
        lattice: (..., 3, 3) の格子行列
    """
    if params.shape[-1] != 6:
        raise ValueError(f"Expected params with shape (..., 6), got {params.shape}")

    # バッチ対応
    original_shape = params.shape[:-1]
    params = params.reshape(-1, 6)

    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    alpha, beta, gamma = params[:, 3], params[:, 4], params[:, 5]

    # 度からラジアンへ
    alpha_rad = alpha * (torch.pi / 180.0)
    beta_rad = beta * (torch.pi / 180.0)
    gamma_rad = gamma * (torch.pi / 180.0)

    cos_alpha = torch.cos(alpha_rad)
    cos_beta = torch.cos(beta_rad)
    cos_gamma = torch.cos(gamma_rad)
    sin_gamma = torch.sin(gamma_rad)

    # 体積計算
    vol = a * b * c * torch.sqrt(
        1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    )

    # 格子ベクトル構築
    zeros = torch.zeros_like(a)

    lattice = torch.stack([
        torch.stack([a, zeros, zeros], dim=-1),
        torch.stack([b * cos_gamma, b * sin_gamma, zeros], dim=-1),
        torch.stack([
            c * cos_beta,
            c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
            vol / (a * b * sin_gamma)
        ], dim=-1)
    ], dim=-2)

    # 元の形状に戻す
    lattice = lattice.reshape(*original_shape, 3, 3)

    return lattice


def frac_to_cart(frac_coords: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    Fractional coordinates を Cartesian coordinates に変換

    Args:
        frac_coords: (N, 3) または (..., N, 3) の fractional coordinates
        lattice: (3, 3) または (..., 3, 3) の格子行列

    Returns:
        cart_coords: (N, 3) または (..., N, 3) の cartesian coordinates
    """
    # frac_coords @ lattice
    return torch.matmul(frac_coords, lattice)


def cart_to_frac(cart_coords: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    Cartesian coordinates を Fractional coordinates に変換

    Args:
        cart_coords: (N, 3) または (..., N, 3) の cartesian coordinates
        lattice: (3, 3) または (..., 3, 3) の格子行列

    Returns:
        frac_coords: (N, 3) または (..., N, 3) の fractional coordinates
    """
    # cart_coords @ inv(lattice)
    lattice_inv = torch.inverse(lattice)
    return torch.matmul(cart_coords, lattice_inv)


def wrap_to_unit_cell(frac_coords: torch.Tensor) -> torch.Tensor:
    """
    Fractional coordinates を [0, 1) の範囲に wrap

    Args:
        frac_coords: (N, 3) の fractional coordinates

    Returns:
        wrapped: (N, 3) の wrapped fractional coordinates
    """
    return torch.fmod(frac_coords, 1.0)


def compute_distance_pbc(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    lattice: torch.Tensor,
    pbc_offset: torch.Tensor,
) -> torch.Tensor:
    """
    PBC を考慮した距離を計算

    Args:
        pos_i: (N, 3) または (..., 3) の座標
        pos_j: (N, 3) または (..., 3) の座標
        lattice: (3, 3) の格子行列
        pbc_offset: (N, 3) の PBC オフセット (整数)

    Returns:
        distance: (N,) または (...,) の距離
    """
    # vector = pos_i - (pos_j + pbc_offset @ lattice)
    offset_cart = torch.matmul(pbc_offset.float(), lattice)
    vector = pos_i - (pos_j + offset_cart)
    distance = torch.norm(vector, dim=-1)
    return distance
