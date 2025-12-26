# Differentiable MatterSim - バッチ処理実装

## 概要

MatterSim の微分可能版にバッチ処理機能を追加しました。複数の結晶を同時に処理し、atom_types, lattice, positions の3つすべてに対して勾配を計算できます。

## 主な機能

- ✅ **バッチ処理**: 可変原子数の複数結晶を同時処理
- ✅ **完全な勾配伝播**: atom_types, lattice, positions すべてに勾配計算
- ✅ **Soft Normalization**: atom_types への完全な勾配維持
- ✅ **PyG 標準形式**: concatenate + batch_index 方式
- ✅ **後方互換性**: 既存 API を破壊せず

## クイックスタート

### Single 構造

```python
import torch
import torch.nn.functional as F
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# 構造準備
si = bulk("Si", "diamond", a=5.43)
calc = DifferentiableMatterSimCalculator(device="cpu")

# 入力準備
atom_types = F.one_hot(torch.tensor(si.get_atomic_numbers()), 95).float()
atom_types.requires_grad_(True)
positions = torch.tensor(si.get_positions(), dtype=torch.float32, requires_grad=True)
lattice = torch.tensor(si.cell.array, dtype=torch.float32, requires_grad=True)

# Forward + Backward
output = calc.forward(si, atom_types=atom_types, positions=positions, lattice=lattice)
output["total_energy"].backward()

# 勾配確認
print(f"atom_types grad: {torch.norm(atom_types.grad):.3e}")
print(f"positions grad:  {torch.norm(positions.grad):.3e}")
print(f"lattice grad:    {torch.norm(lattice.grad):.3e}")
```

### Batch 処理

```python
import torch
import torch.nn.functional as F
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# 複数結晶準備
si = bulk("Si", "diamond", a=5.43)
fe = bulk("Fe", "fcc", a=3.6)
atoms_list = [si, fe]
calc = DifferentiableMatterSimCalculator(device="cpu")

# バッチ入力準備（concatenate 方式）
atom_types = torch.cat([
    F.one_hot(torch.tensor(si.get_atomic_numbers()), 95).float(),
    F.one_hot(torch.tensor(fe.get_atomic_numbers()), 95).float()
], dim=0).requires_grad_(True)

positions = torch.cat([
    torch.tensor(si.get_positions(), dtype=torch.float32),
    torch.tensor(fe.get_positions(), dtype=torch.float32)
], dim=0).requires_grad_(True)

lattice = torch.stack([
    torch.tensor(si.cell.array, dtype=torch.float32),
    torch.tensor(fe.cell.array, dtype=torch.float32)
], dim=0).requires_grad_(True)

sizes = torch.tensor([len(si), len(fe)])

# Batch forward + backward
output = calc.forward_batch(
    atoms_list, atom_types=atom_types, positions=positions,
    lattice=lattice, sizes=sizes, soft_normalize=True
)
energies = output["total_energy"]  # (2,) [Si, Fe]
energies.sum().backward()

# 勾配確認
print(f"Batch energies: {energies.detach()}")
print(f"atom_types grad: {torch.norm(atom_types.grad):.3e}")
print(f"positions grad:  {torch.norm(positions.grad):.3e}")
print(f"lattice grad:    {torch.norm(lattice.grad):.3e}")
```

## ドキュメント

詳細なドキュメントは以下を参照してください：

- **API ドキュメント**: [docs/DIFFERENTIABLE_API.md](docs/DIFFERENTIABLE_API.md)
  - 使用方法
  - API リファレンス
  - 制限事項
  - トラブルシューティング

## デモ

### Python スクリプト

```bash
# Single 構造のデモ
python examples/differentiable_demo.py

# Batch 処理のデモ
python examples/differentiable_batch_demo.py
```

### Jupyter Notebook

```bash
# インタラクティブなデモ（可視化付き）
jupyter notebook examples/differentiable_batch_demo.ipynb
```

## テスト

```bash
# 全テスト実行
python -m pytest tests/test_differentiable.py tests/test_differentiable_batch.py -v

# バッチ処理テストのみ
python -m pytest tests/test_differentiable_batch.py -v
```

**結果**: ✅ 18 passed (8 既存 + 10 新規)

## 実装ファイル

### 変更ファイル（3件）

1. `src/mattersim/forcefield/differentiable_potential.py`
   - `forward_batch()` メソッド追加
   - ヘルパー関数: `sizes_to_batch_index`, `batch_index_to_sizes`

2. `src/mattersim/datasets/utils/differentiable_convertor.py`
   - `DifferentiableGraphBuilder` のバッチ対応
   - 複数構造の近傍リスト構築

3. `src/mattersim/forcefield/m3gnet/differentiable_m3gnet.py`
   - `soft_normalize` オプション追加
   - atom_types への完全な勾配伝播

### 新規ファイル（3件）

4. `tests/test_differentiable_batch.py` - バッチ処理テスト（10テスト）
5. `examples/differentiable_batch_demo.py` - デモスクリプト
6. `examples/differentiable_batch_demo.ipynb` - Jupyter Notebook デモ

## 入力形式

### atom_types
- **形状**: `(sum(N_i), 95)` - 全結晶の原子を連結
- **型**: `torch.Tensor`, `float32`
- **内容**: 原子種分布（one-hot または連続値）
- **勾配**: `soft_normalize=True` で完全な勾配伝播

### positions
- **形状**: `(sum(N_i), 3)` - 全結晶の座標を連結
- **型**: `torch.Tensor`, `float32`
- **座標系**: Cartesian
- **勾配**: 常に計算可能

### lattice
- **形状**: `(nb_graphs, 3, 3)` - 各結晶の格子を stack
- **型**: `torch.Tensor`, `float32`
- **内容**: 格子ベクトル（行列形式）
- **勾配**: 常に計算可能

### batch_index または sizes
- **batch_index**: `(sum(N_i),)` - 各原子の所属グラフID
- **sizes**: `(nb_graphs,)` - 各結晶の原子数
- **どちらか一方を指定**（自動変換可能）

## 出力形式

```python
{
    "total_energy": torch.Tensor,  # (nb_graphs,) 各結晶のエネルギー
    "forces": torch.Tensor,        # (sum(N_i), 3) オプション
}
```

## 制限事項

1. **固定トポロジー**
   - 近傍リストは初期構造で固定
   - 構造が大きく変化する場合は再初期化が必要

2. **stresses 未実装**
   - バッチ処理での応力計算は現状未対応
   - forces のみ対応

3. **カットオフ不連続性**
   - 原子がカットオフ境界を越えると勾配が不連続
   - 実用上は影響小

## 性能

- **メモリ**: バッチサイズに応じて線形増加
- **速度**: ベクトル化により効率的
- **推奨バッチサイズ**: GPU メモリに応じて調整

## 今後の拡張

- [ ] stresses のバッチ対応
- [ ] 動的近傍リスト（構造が大きく変化する場合）
- [ ] gradient checkpointing（メモリ効率改善）

## ライセンス

このプロジェクトは MatterSim の一部であり、同じライセンスに従います。

## 引用

MatterSim を使用する場合は、元の論文を引用してください：

```bibtex
@article{mattersim2024,
  title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
  author={...},
  journal={arXiv preprint arXiv:2405.04967},
  year={2024}
}
```
