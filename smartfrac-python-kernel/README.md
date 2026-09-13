# SmartFrac

基于 AI（神经网络原子势 NNP + 物理信息神经网络 PINN + CNN/Transformer）的材料断裂力学仿真与反演科研原型。

## 技术栈（依据《项目具体开发语言、框架、库、工具明细》）

- **语言**：Python 3.10 ~ 3.11（推荐 3.10；本环境使用 3.11.9）
- **深度学习**：PyTorch 2.2+（GPU 优先，当前开发机无 NVIDIA GPU，使用 CPU 版降级调试）
- **原子模拟**：ASE、DScribe（SOAP 描述符）；LAMMPS 为外部可选 MD 基准引擎
- **科学计算**：NumPy / SciPy / pandas / h5py / PyYAML / OmegaConf
- **可视化**：matplotlib / pyvista / seaborn
- **工程化**：pytest、TensorBoard、git + git-lfs

> 第二阶段起将引入 PyQt5 GUI、scikit-learn、optuna、transformers、FEniCSx；
> 第四阶段引入 Celery+Redis 分布式调度与 FastAPI 后端。

## 环境复现

### 方式 A：venv + pip（当前采用）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# CPU 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 方式 B：conda / mamba

```bash
conda env create -f environment.yml
conda activate smartfrac
```

## 目录结构

```
SmartFrac/
├── configs/            # YAML 算例与训练参数（禁止硬编码）
├── src/smartfrac/      # 业务内核
│   ├── nnp/            # 神经网络原子势
│   ├── pinn/           # 物理信息神经网络
│   ├── md/             # ASE / LAMMPS 接口
│   ├── io_utils/       # 原子构型 / HDF5 读写
│   └── viz/            # matplotlib / pyvista 可视化
├── tests/              # pytest 单元测试
├── data/               # 训练数据（大文件走 git-lfs）
├── results/            # 输出结果
└── docs/               # 设计文档
```

## 快速验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
