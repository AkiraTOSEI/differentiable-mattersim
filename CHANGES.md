# MatterSim 微分可能化の変更サマリー

このドキュメントは、MatterSim を微分可能にするために追加されたファイルと変更内容をまとめたものです。

## 変更日時
2025-12-23

## 概要
MatterSim を PyTorch の autograd と統合し、入力（原子種分布、格子、座標）から出力（エネルギー）まで完全に微分可能にしました。

## 新規追加ファイル

### 1. コア実装ファイル

#### `src/mattersim/utils/geometry_torch.py`
**目的:** torch ベースの幾何演算ユーティリティ

**主な機能:**
- `lattice_params_to_matrix()`: 格子パラメータ (a,b,c,α,β,γ) → 3×3 行列変換
- `frac_to_cart()`: fractional → cartesian 座標変換
- `cart_to_frac()`: cartesian → fractional 座標変換
- `wrap_to_unit_cell()`: fractional 座標を [0,1) に wrap
- `compute_distance_pbc()`: PBC を考慮した距離計算

**行数:** 約 150 行

---

#### `src/mattersim/datasets/utils/differentiable_convertor.py`
**目的:** 微分可能な Graph 変換

**主なクラス:**
- `DifferentiableGraphBuilder`: 固定トポロジーで微分可能な Graph 構築
  - `__init__()`: 初期構造で近傍リストを構築（numpy/pymatgen 使用）
  - `forward()`: エッジベクトルとエッジ長を torch で再計算

**特徴:**
- トポロジー（edge_index, pbc_offsets, three_body_indices）を固定
- エッジ長・角度のみを torch で再計算して微分可能に

**行数:** 約 120 行

---

#### `src/mattersim/forcefield/m3gnet/differentiable_m3gnet.py`
**目的:** 微分可能な M3GNet 実装

**主なクラス:**
- `DifferentiableM3Gnet(M3Gnet)`: 既存の M3Gnet を継承
  - `forward_differentiable()`: 完全に微分可能な forward パス

**主な変更点:**
- 原子種を連続値 one-hot として扱う（`atom_attr @ embedding_weight`）
- `atomic_numbers.long()` による整数化を回避
- num_atoms, num_bonds などを tensor に変換

**行数:** 約 170 行

---

#### `src/mattersim/forcefield/differentiable_potential.py`
**目的:** 微分可能な MatterSim Calculator

**主なクラス:**
- `DifferentiableMatterSimCalculator`: メインの微分可能 API
  - `__init__()`: 事前学習モデルをロードして DifferentiableM3Gnet に変換
  - `forward()`: ASE Atoms + オプショナルな tensor 入力でエネルギー計算

**主な機能:**
- atom_types, positions, lattice を tensor として受け取り
- DifferentiableGraphBuilder で Graph を構築
- DifferentiableM3Gnet で forward
- 力・応力も torch.autograd.grad で計算

**行数:** 約 230 行

---

### 2. テスト・デモファイル

#### `tests/test_differentiable.py`
**目的:** 回帰テスト、勾配テスト、最適化テスト

**テストクラス:**
1. `TestDifferentiableRegression`: 既存 API との数値一致確認
   - `test_diamond_si_energy()`: Diamond Si のエネルギーが baseline と一致
   - `test_diamond_si_forces()`: Diamond Si の力が baseline と一致
   - `test_fcc_fe_energy()`: FCC-Fe のエネルギーが baseline と一致

2. `TestDifferentiableGradient`: 勾配の流れを確認
   - `test_energy_grad_wrt_positions()`: 座標への勾配
   - `test_energy_grad_wrt_lattice()`: 格子への勾配
   - `test_energy_grad_wrt_atom_types()`: 原子種への勾配
   - `test_forces_via_autograd()`: autograd で力が計算できる

3. `TestDifferentiableOptimization`: 最適化のデモ
   - `test_optimize_positions()`: 座標の最適化が動作する

**行数:** 約 200 行

---

#### `scripts/generate_baseline.py`
**目的:** Baseline データの生成

**機能:**
- Diamond Si と FCC-Fe の現状の MatterSim 出力を取得
- JSON ファイルに保存（`tests/data/baseline_outputs.json`）
- テストでの数値比較に使用

**行数:** 約 80 行

---

#### `examples/differentiable_demo.py`
**目的:** コマンドライン用のデモスクリプト

**デモ内容:**
1. 勾配の流れを確認
2. 構造最適化（座標を Adam で最適化）
3. 格子定数の最適化（格子を SGD で最適化）

**行数:** 約 180 行

---

#### `examples/differentiable_mattersim_demo.ipynb`
**目的:** Jupyter Notebook 用のデモ

**デモ内容:**
1. 基本的な使い方：勾配の計算
2. 構造最適化（可視化付き）
3. 格子定数の最適化（可視化付き）
4. 複数パラメータの同時最適化
5. まとめと次のステップ

**セル数:** 約 20 セル

---

### 3. データ・ドキュメントファイル

#### `tests/data/baseline_outputs.json`
**目的:** 回帰テスト用の baseline データ

**内容:**
- Diamond Si の出力（energy, forces, stress）
- FCC-Fe の出力（energy, forces, stress）
- 構造情報（symbols, positions, cell, pbc）

**サイズ:** 約 100 行（JSON 形式）

---

#### `docs/DIFFERENTIABLE_API.md`
**目的:** API ドキュメント

**内容:**
- 概要と主な機能
- 使用方法（基本、最適化、格子最適化、原子種勾配）
- API リファレンス
- 制限事項（固定トポロジー、カットオフの不連続性、normalizer）
- テストの実行方法
- デモの実行方法
- 実装の詳細
- トラブルシューティング
- 今後の拡張

**行数:** 約 400 行

---

## 既存ファイルへの変更

**なし** - 既存の API を壊さないように、新しいファイルとして実装しました。

---

## 全体の統計

- **新規ファイル数:** 10 ファイル
- **総追加行数:** 約 1,600 行（コメント含む）
- **テスト数:** 8 テスト（全て pass）

---

## 使用方法

### 1. Baseline の生成（初回のみ）

```bash
python scripts/generate_baseline.py
```

### 2. テストの実行

```bash
python tests/test_differentiable.py -v
```

### 3. デモの実行

**コマンドライン:**
```bash
python examples/differentiable_demo.py
```

**Jupyter Notebook:**
```bash
jupyter notebook examples/differentiable_mattersim_demo.ipynb
```

### 4. コードでの使用

```python
import torch
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# 構造を作成
si = bulk("Si", "diamond", a=5.43)

# Calculator
calc = DifferentiableMatterSimCalculator(device="cpu")

# 座標を tensor に
positions = torch.tensor(si.get_positions(), dtype=torch.float32, requires_grad=True)

# エネルギー計算
output = calc.forward(si, positions=positions, include_forces=False)
energy = output["total_energy"]

# backward
energy.backward()

# 勾配を取得
print(positions.grad)
```

---

## 設計の原則

1. **既存 API を壊さない**: 新しいクラスとして実装
2. **完全に微分可能**: numpy 化や detach を避ける
3. **テスト駆動**: 回帰テストと勾配テストで検証
4. **ドキュメント充実**: 使い方と制限事項を明記

---

## 制限事項

1. **固定トポロジー**: 近傍リストは初期構造で固定
2. **カットオフの不連続性**: 原子がカットオフ境界を越えると不連続
3. **normalizer の挙動**: 原子種分布への勾配が normalizer で切れる可能性

詳細は `docs/DIFFERENTIABLE_API.md` を参照してください。

---

## 今後の拡張

- 動的近傍リスト
- ソフトカットオフ
- JIT コンパイル対応
- バッチ処理

---

## テスト結果

```
test_diamond_si_energy ... ok
test_diamond_si_forces ... ok
test_fcc_fe_energy ... ok
test_energy_grad_wrt_positions ... ok
test_energy_grad_wrt_lattice ... ok
test_energy_grad_wrt_atom_types ... ok
test_forces_via_autograd ... ok
test_optimize_positions ... ok

Ran 8 tests in 30.605s
OK
```

全テスト通過 ✅
