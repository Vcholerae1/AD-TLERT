# ERT examples

本目录集中保存两组可直接运行的 ERT 示例：

- `single_time/`：单时刻正演与反演。
- `window_363/`：365 个时刻按窗口大小 3、步长 1 进行反演，共 363 个窗口。

所有命令均在仓库根目录执行，输出写入 `result/`。可用
`--project-root`、`--forward-dir` 和 `--output-dir` 覆盖默认路径。
pyGIMLi 窗口反演还需要 `PyHydroGeophysX`；Linux 下运行 ResIPy/R2
需要 Wine。ResIPy 3.6.6 要求 NumPy 1.x，与主环境的 JAX/NumPy 2.x
约束冲突，因此需要独立环境：

```bash
uv venv .venv-resipy --python 3.11
uv pip install --python .venv-resipy/bin/python "resipy==3.6.6" psutil
```

## 单时刻正演与反演

Deepert 正演及反演：

```bash
.venv/bin/python example/single_time/forward_deepert.py
.venv/bin/python example/single_time/inversion_deepert.py
```

pyGIMLi 正演及反演：

```bash
.venv/bin/python example/single_time/forward_pygimli.py
.venv/bin/python example/single_time/inversion_pygimli.py \
  --data-file result/1_single_forward_pygimli/synthetic_ert_terrain_vardz.dat
```

ResIPy/R2 反演：

```bash
.venv-resipy/bin/python example/single_time/inversion_resipy.py
```

ResIPy 在这组代码中只承担反演，读取 Deepert 生成的通用 NPZ 观测数据；
仓库内没有单独的 ResIPy 正演实现。

## 363 窗口反演

以下命令读取 `result/1_timelapsedERT_forward_deepert` 中的 365 个时刻。
默认 `--window-size 3 --window-step 1`，因此生成 363 个窗口。

```bash
.venv/bin/python example/window_363/inversion_deepert.py
.venv/bin/python example/window_363/inversion_pygimli.py
.venv-resipy/bin/python example/window_363/inversion_resipy.py
```

ResIPy 入口支持断点续跑：已存在 `wNNN/summary.json` 的窗口会自动跳过。
三套实现计算量均较大，正式运行前可分别使用
`--max-timesteps`（Deepert/pyGIMLi）或较大的 `--window-step`
（ResIPy）做小规模验证。
