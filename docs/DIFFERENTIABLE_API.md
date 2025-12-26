# 微分可能な MatterSim API

このドキュメントは、MatterSim を PyTorch の autograd システムと統合し、入力（原子種分布、格子、座標）に対して勾配を計算できるようにする新しい API について説明します。

## 概要

通常の MatterSim API では、ASE Atoms オブジェクトを入力として受け取りますが、内部で numpy 配列に変換されるため、PyTorch の autograd が切れてしまいます。

新しい `DifferentiableMatterSimCalculator` API は、この問題を解決し、以下を可能にします：

- **原子種分布**（連続値 one-hot エンコーディング）への勾配
- **格子パラメータ**（格子定数、格子ベクトル）への勾配
- **原子座標**（Cartesian または fractional）への勾配

これにより、MatterSim を損失関数の一部として使用し、入力を最適化することができます。

## 主な機能

1. ✅ **完全に微分可能**: 入力から出力まで autograd が切れない
2. ✅ **既存 API との互換性**: 既存の `MatterSimCalculator` は変更なし
3. ✅ **回帰テスト済み**: Diamond Si と FCC-Fe で数値的一致を確認
4. ✅ **勾配テスト済み**: backward() が正常に動作することを確認
5. ✅ **バッチ処理対応**: 複数構造を同時に処理可能（可変原子数対応）
6. ✅ **Soft Normalization**: atom_types への完全な勾配伝播

## 使用方法

### 基本的な使い方

```python
import torch
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# 構造を作成
si = bulk("Si", "diamond", a=5.43)

# 微分可能な Calculator を初期化
calc = DifferentiableMatterSimCalculator(device="cpu")

# 座標を tensor として取得し、requires_grad を設定
positions = torch.tensor(
    si.get_positions(), dtype=torch.float32, requires_grad=True
)

# エネルギーを計算
output = calc.forward(si, positions=positions, include_forces=False)
energy = output["total_energy"]

# backward で勾配を計算
energy.backward()

# 座標に対する勾配を取得
print(positions.grad)  # dE/dx
```

### 構造最適化の例

```python
import torch
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# Diamond Si 構造を作成し、ランダムに摂動
si = bulk("Si", "diamond", a=5.43)
positions = torch.tensor(si.get_positions(), dtype=torch.float32)
positions = positions + torch.randn_like(positions) * 0.05  # 摂動を加える
positions.requires_grad_(True)

# Calculator
calc = DifferentiableMatterSimCalculator(device="cpu")

# オプティマイザ
optimizer = torch.optim.Adam([positions], lr=0.02)

# 最適化ループ
for step in range(50):
    optimizer.zero_grad()

    output = calc.forward(si, positions=positions, include_forces=False)
    energy = output["total_energy"]

    energy.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: Energy = {energy.item():.6f} eV")
```

### 格子定数の最適化

```python
import torch
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

si = bulk("Si", "diamond", a=5.5)  # 少しずらした格子定数

calc = DifferentiableMatterSimCalculator(device="cpu")

# 格子を tensor として取得
lattice = torch.tensor(si.cell.array, dtype=torch.float32, requires_grad=True)

# オプティマイザ
optimizer = torch.optim.SGD([lattice], lr=0.01)

# 最適化
for step in range(50):
    optimizer.zero_grad()

    output = calc.forward(si, lattice=lattice, include_forces=False)
    energy = output["total_energy"]

    energy.backward()
    optimizer.step()
```

### 原子種分布への勾配（発展的な使い方）

```python
import torch
import torch.nn.functional as F
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

si = bulk("Si", "diamond", a=5.43)

calc = DifferentiableMatterSimCalculator(device="cpu")

# 原子種を連続値 one-hot として表現
atomic_numbers = torch.tensor(si.get_atomic_numbers(), dtype=torch.long)
atom_types = F.one_hot(atomic_numbers, num_classes=95).float()
atom_types.requires_grad_(True)

output = calc.forward(si, atom_types=atom_types, include_forces=False)
energy = output["total_energy"]

energy.backward()

# 原子種分布への勾配
print(atom_types.grad)  # dE/d(atom_types)
```

### バッチ処理の使い方

複数の結晶を同時に処理できます。

```python
import torch
import torch.nn.functional as F
from ase.build import bulk
from mattersim.forcefield.differentiable_potential import DifferentiableMatterSimCalculator

# 複数の結晶を準備
si = bulk("Si", "diamond", a=5.43)
fe = bulk("Fe", "fcc", a=3.6)
atoms_list = [si, fe]

# Calculator を初期化
calc = DifferentiableMatterSimCalculator(device="cpu")

# バッチ入力を準備（concatenate 方式）
# atom_types: (sum(N_i), 95)
atom_types_si = F.one_hot(torch.tensor(si.get_atomic_numbers()), 95).float()
atom_types_fe = F.one_hot(torch.tensor(fe.get_atomic_numbers()), 95).float()
atom_types = torch.cat([atom_types_si, atom_types_fe], dim=0)
atom_types.requires_grad_(True)

# positions: (sum(N_i), 3)
positions = torch.cat([
    torch.tensor(si.get_positions(), dtype=torch.float32),
    torch.tensor(fe.get_positions(), dtype=torch.float32)
], dim=0)
positions.requires_grad_(True)

# lattice: (nb_graphs, 3, 3)
lattice = torch.stack([
    torch.tensor(si.cell.array, dtype=torch.float32),
    torch.tensor(fe.cell.array, dtype=torch.float32)
], dim=0)
lattice.requires_grad_(True)

# sizes を指定（各結晶の原子数）
sizes = torch.tensor([len(si), len(fe)])

# バッチ forward
output = calc.forward_batch(
    atoms_list,
    atom_types=atom_types,
    positions=positions,
    lattice=lattice,
    sizes=sizes,
    include_forces=False,
    soft_normalize=True,  # atom_types への勾配を維持
)

energies = output["total_energy"]  # (2,) [Si のエネルギー, Fe のエネルギー]

# backward
loss = energies.sum()
loss.backward()

# 勾配が全て計算される
print(atom_types.grad.shape)  # (sum(N_i), 95)
print(positions.grad.shape)    # (sum(N_i), 3)
print(lattice.grad.shape)      # (2, 3, 3)
```

## API リファレンス

### `DifferentiableMatterSimCalculator`

微分可能な MatterSim Calculator クラス。

#### `__init__(potential=None, device="cpu", twobody_cutoff=5.0, threebody_cutoff=4.0, **kwargs)`

**パラメータ:**
- `potential`: 事前学習済み `Potential` オブジェクト（`None` の場合は自動ロード）
- `device`: `"cpu"` または `"cuda"`
- `twobody_cutoff`: 2体カットオフ半径（Å）
- `threebody_cutoff`: 3体カットオフ半径（Å）

#### `forward(atoms, atom_types=None, positions=None, lattice=None, include_forces=False, include_stresses=False, return_forces=None, return_stress=None, create_graph_forces=False, create_graph_stress=False, soft_normalize=False)`

微分可能な forward パス。

**パラメータ:**
- `atoms`: ASE Atoms オブジェクト（トポロジー構築用）
- `atom_types`: `(N, max_z+1)` の原子種分布（`None` の場合は `atoms` から取得）
- `positions`: `(N, 3)` の Cartesian 座標（`None` の場合は `atoms` から取得）
- `lattice`: `(3, 3)` の格子行列（`None` の場合は `atoms` から取得）
- `include_forces`: 力を計算するか
- `include_stresses`: 応力を計算するか
- `return_forces`: `include_forces` の別名（優先）
- `return_stress`: `include_stresses` の別名（優先）
- `create_graph_forces`: forces を損失に使う場合に True
- `create_graph_stress`: stress を損失に使う場合に True

**戻り値:**
- `dict`: `{"total_energy": torch.Tensor, "forces": torch.Tensor (optional), "stresses": torch.Tensor (optional)}`
  - `stresses` は GPa 単位

#### `forward_batch(atoms_list, atom_types=None, positions=None, lattice=None, batch_index=None, sizes=None, include_forces=False, include_stresses=False, return_forces=None, return_stress=None, create_graph_forces=False, create_graph_stress=False, soft_normalize=True)`

バッチ処理版 forward パス。

**パラメータ:**
- `atoms_list`: ASE Atoms のリスト（トポロジー構築用、`len=nb_graphs`）
- `atom_types`: `(sum(N_i), max_z+1)` 原子種分布（連続値OK）
- `positions`: `(sum(N_i), 3)` Cartesian 座標
- `lattice`: `(nb_graphs, 3, 3)` 格子行列
- `batch_index`: `(sum(N_i),)` 各原子の所属グラフID（`0, 1, 2, ...`）
- `sizes`: `(nb_graphs,)` 各グラフの原子数（`batch_index` の代替）
- `include_forces`: 力を計算するか
- `include_stresses`: 応力を計算するか
- `soft_normalize`: `True` なら argmax せずに soft normalization（atom_types への勾配を維持）

**戻り値:**
- `dict`: `{"total_energy": (nb_graphs,), "forces": (sum(N_i), 3) (optional), "stresses": (nb_graphs, 3, 3) (optional)}`

**Note:**
- `batch_index` と `sizes` のどちらかを指定。両方 `None` の場合は `sizes` を自動計算。
- `soft_normalize=True` を推奨（atom_types への完全な勾配伝播）

#### `predict_from_tensors(atom_types, lattice, positions, return_energy=True, return_forces=False, return_stress=False, create_graph_forces=False, create_graph_stress=False, soft_normalize=True)`

Atoms を使わずにテンソル入力から推論するヘルパー。内部でトポロジー用の Atoms を構築します。

#### `predict_from_batch_tensors(batch_atom_types, lattice, positions, batch_index=None, sizes=None, return_energy=True, return_forces=False, return_stress=False, create_graph_forces=False, create_graph_stress=False, soft_normalize=True)`

バッチ版のテンソル入力ヘルパー。

## 制限事項

### 1. 固定トポロジー

現在の実装では、近傍リスト（neighbor list）のトポロジーは初期構造で固定されます。

**影響:**
- 構造が大きく変化すると（例: 格子定数が 2 倍になる）、近傍リストが更新されません
- 通常の最適化（小さな変動）では問題ありません

**回避策:**
- 構造が大きく変化する場合は、Calculator を再初期化してください

```python
# 構造が大きく変化した場合
calc = DifferentiableMatterSimCalculator(device="cpu")  # 再初期化
```

### 2. カットオフの不連続性

近傍リストはカットオフ半径で切られているため、原子がカットオフ境界を越えると不連続になります。

**影響:**
- 勾配が不連続になる可能性があります（まれ）

**回避策:**
- 通常の使用では問題ありません
- 極端な最適化を避ける

### 3. normalizer の挙動

AtomScaling normalizer は、原子種を整数で受け取ります。連続値の原子種分布を使用する場合、argmax で最も確率の高い原子番号を取得します。

**影響:**
- 原子種分布への勾配が normalizer で切れる可能性があります

## 2階微分について

forces/stress を損失に含める場合は `create_graph_forces=True` / `create_graph_stress=True` が必要です。
2階微分のためメモリ使用量が増えるので、必要なときだけ有効化してください。

## テスト

テストを実行して、実装が正しく動作することを確認できます：

### Single 構造のテスト

```bash
# 全テストを実行
python -m pytest tests/test_differentiable.py -v

# 回帰テストのみ
python -m pytest tests/test_differentiable.py::TestDifferentiableRegression -v

# 勾配テストのみ
python -m pytest tests/test_differentiable.py::TestDifferentiableGradient -v
```

### Batch 処理のテスト

```bash
# バッチ処理の全テストを実行
python -m pytest tests/test_differentiable_batch.py -v

# ヘルパー関数テスト
python -m pytest tests/test_differentiable_batch.py::TestBatchHelpers -v

# バッチ勾配テスト
python -m pytest tests/test_differentiable_batch.py::TestBatchGradient -v
```

### 全テスト実行

```bash
# 既存 + バッチのすべてのテストを実行
python -m pytest tests/test_differentiable.py tests/test_differentiable_batch.py -v
```

## デモ

### Single 構造のデモ

```bash
python examples/differentiable_demo.py
```

このスクリプトは以下を実演します：
1. 勾配の流れの確認
2. 構造最適化
3. 格子定数の最適化

### Batch 処理のデモ

```bash
python examples/differentiable_batch_demo.py
```

このスクリプトは以下を実演します：
1. **Single - 3変数同時最適化**: atom_types, lattice, positions を同時に最適化
2. **Single - 連続分布 atom_types**: 非one-hot の atom_types で勾配更新
3. **Batch - 3変数同時最適化**: 2結晶（Si + Fe）で同時最適化
4. **Batch - 連続分布 atom_types**: バッチ + 連続分布の組み合わせ

### Jupyter Notebook デモ

```bash
jupyter notebook examples/differentiable_batch_demo.ipynb
```

インタラクティブなデモで、可視化付きで動作を確認できます。

## 実装の詳細

### アーキテクチャ

新しい API は以下のコンポーネントで構成されています：

1. **`DifferentiableGraphBuilder`** (`src/mattersim/datasets/utils/differentiable_convertor.py`)
   - 初期構造で近傍リストを構築（numpy/pymatgen 使用）
   - トポロジーを固定し、エッジベクトルとエッジ長のみを torch で再計算
   - **バッチ対応**: `List[Atoms]` を受け取り、複数構造の近傍リストを連結

2. **`DifferentiableM3Gnet`** (`src/mattersim/forcefield/m3gnet/differentiable_m3gnet.py`)
   - 既存の M3Gnet を継承
   - 原子種を連続値 one-hot として扱う
   - `forward_differentiable()` メソッドで完全に微分可能な forward パス
   - **Soft Normalization**: `soft_normalize=True` で atom_types への完全な勾配伝播

3. **`DifferentiableMatterSimCalculator`** (`src/mattersim/forcefield/differentiable_potential.py`)
   - ASE Atoms を受け取り、微分可能な経路でエネルギー・力・応力を計算
   - `forward()` メソッド: 単一構造用
   - **`forward_batch()` メソッド**: バッチ処理用（新規）

4. **ヘルパー関数** (`src/mattersim/forcefield/differentiable_potential.py`)
   - `sizes_to_batch_index()`: sizes → batch_index 変換
   - `batch_index_to_sizes()`: batch_index → sizes 変換

5. **幾何演算ユーティリティ** (`src/mattersim/utils/geometry_torch.py`)
   - torch ベースの格子変換、座標変換など

### バッチ処理の実装方式

**採用方式: concatenate + batch_index（PyTorch Geometric 標準）**

- **入力形式:**
  - `atom_types`: `(sum(N_i), 95)` - 全結晶の原子を連結
  - `positions`: `(sum(N_i), 3)` - 全結晶の座標を連結
  - `lattice`: `(nb_graphs, 3, 3)` - 各結晶の格子を stack
  - `batch_index`: `(sum(N_i),)` - 各原子がどの結晶に属するか

- **利点:**
  - PyTorch Geometric との互換性
  - ベクトル化による高速化
  - 可変原子数に対応

### 設計の原則

- **既存 API を壊さない**: 新しいメソッドとして `forward_batch` を追加
- **完全に微分可能**: numpy 化や detach を避ける
- **テスト駆動**: 回帰テストと勾配テストで検証
- **後方互換性**: 既存の `forward()` は変更なし

## トラブルシューティング

### エラー: "Expected all tensors to be on the same device"

**原因:** モデルと入力が異なるデバイスにあります。

**解決策:** `device` パラメータを明示的に指定してください。

```python
calc = DifferentiableMatterSimCalculator(device="cpu")
```

### エラー: "No module named 'mattersim.forcefield.differentiable_potential'"

**原因:** パッケージがインストールされていません。

**解決策:** editable モードでインストールしてください。

```bash
pip install -e .
```

### 勾配が None になる

**原因:** `requires_grad=True` が設定されていないか、計算グラフが途中で切れています。

**解決策:** tensor に `requires_grad=True` を設定してください。

```python
positions.requires_grad_(True)
```

## 今後の拡張

将来的に以下の機能を追加する可能性があります：

- ✨ **動的近傍リスト**: 構造が大きく変化しても対応
- ✨ **ソフトカットオフ**: カットオフ境界を滑らかに
- ✨ **JIT コンパイル対応**: さらなる高速化
- ✨ **バッチ処理**: 複数構造の並列処理

## 参考文献

- MatterSim 論文: [arXiv:2405.04967](https://arxiv.org/abs/2405.04967)
- PyTorch autograd: [公式ドキュメント](https://pytorch.org/docs/stable/autograd.html)

## 貢献

バグ報告や機能リクエストは、GitHub の Issues でお願いします。
