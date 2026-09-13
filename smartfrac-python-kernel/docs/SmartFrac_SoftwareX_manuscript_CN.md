# SmartFrac：面向BCC α-铁多尺度断裂仿真的分离式NNP–PINN原型

**第一作者***, 第二作者, 第三作者

*<单位，城市，国家>*
*通讯作者：<邮箱>*

**软件信息。** Python 3.11 / PyTorch；仅CPU训练；源代码、训练检查点与预构建HDF5描述符：https://github.com/shishijingyu/SmartFrac 。许可证：MIT。

---

## 摘要

SmartFrac是一个刻意最小化、完全可复现的Python原型，用于BCC α-铁的多尺度AI断裂仿真。它在同一代码库内组装两条相互独立但共享数据契约的链路：原子尺度神经网络原子势（NNP），基于公开DFT铁数据库以原子位置平滑重叠（SOAP）描述符训练；连续介质尺度物理信息神经网络（PINN），以Westergaard I型裂纹闭式解为基准求解二维平面应力裂纹问题。原子链路在独立保留的2567个构型测试集上复算，能量平均绝对误差达42.5 meV/atom（Pearson相关系数 r=0.983）；完整训练在现代多核CPU上即可完成。连续介质链路以局部集中配点将自动微分平衡残差降至约3×10⁻⁵。软件对外暴露统一的指标集——能量误差、力误差、位移场相对L2误差与MD单步壁钟时间——使后续所有扩展都能在同一基线上被度量。本软件旨在作为耦合NNP与PINN链路的教学与研究脚手架，而非最先进的断裂求解器；其已知局限以代码级精度记录在案。

**关键词：** 神经网络原子势；物理信息神经网络；断裂力学；SOAP描述符；BCC α-铁；多尺度仿真

---

## 1. 引言

跨越原子与连续介质尺度的断裂仿真代价高昂：基于密度泛函理论（DFT）的分子动力学（MD）受限于纳秒与纳米量级，而连续介质有限元分析又依赖单独标定、无法继承原子尺度键断裂信息。断裂力学路线图（Yang, Feng, and Gao, 2026）提出的两个开放问题框定了更宏观的议程：人工智能能否以可承受成本求解多尺度断裂问题（其第16问）？AI又应如何改进统计断裂力学的不确定性量化（其第23问）？

机器学习已在原子侧打破代价瓶颈：神经网络原子势以极小代价复现DFT能量与力，从Behler–Parrinello分解（Behler and Parrinello, 2007）与高斯近似势（Bartók等, 2010），发展到深度势（Zhang等, 2018）、SchNet（Schütt等, 2018）、等变图网络（Batzner等, 2022; Batatia等, 2022）与进化策略势（Fan等, 2021），综述见Unke等（2021）与Behler（2021）。连续介质侧，物理信息神经网络将平衡偏微分方程直接嵌入损失（Raissi等, 2019; Karniadakis等, 2021），提供无网格求解器。

目前尚缺一个轻量、开源、可复现的单一代码库，能将两条链路置于相同数据契约与相同指标词汇之下，使学生或新课题组能在一下午内读完、在CPU上重跑两条链路、并准确看到哪里可行、哪里不可行。SmartFrac填补了这一空白。它不是最先进的断裂求解器，而是一个弱点被准确定位而非隐藏的脚手架。

## 2. 动机与意义

若干成熟软件包单独覆盖原子侧——LAMMPS配机器学习势、n2p2、SchNetPack、MACE——若干PINN框架单独覆盖偏微分方程求解器。要将二者结合用于断裂研究，用户仍需自己编写粘合层：描述符、HDF5存储、携带自身归一化的检查点、Westergaard评估器、计时工具，以及一个两侧含义一致的指标接口。当结果与基准不符时，调试精力花在粘合层而非科学问题上。

SmartFrac的意义有三：

1. **一个代码库，两条链路。** 原子NNP与连续介质PINN位于同一包内，均继承自公共的 `BaseModel`，后者将权重与配置、归一化统计量一同序列化，检查点自描述。
2. **一套指标词汇。** 两条链路经同一接口返回——能量MAE/RMSE、力MAE/RMSE、Pearson r、位移场相对L2误差、MD单步壁钟时间。未来的耦合工作以相同数字被检验。
3. **诚实的CPU可跑基线。** 整条流水线在CPU上训练，无需GPU。已知缺口——力非能量自洽、裂纹面未加无拉力边界、未采用富集裂尖形式、无不确定性量化——均明确记录并定位到具体代码位置，使后续贡献者不必重新推导。

目标读者为刚开始AI-for-fracture课题、需要一个透明可跑参考而非黑盒生产框架的研究生与研究者。

## 3. 软件描述

### 3.1 安装与快速开始

SmartFrac 需要 Python 3.11，依赖 PyTorch、ASE、DScribe、h5py、NumPy 与 Matplotlib。从全新克隆运行：

```bash
git clone https://github.com/shishijingyu/SmartFrac.git
cd SmartFrac
python -m venv .venv
.venv\Scripts\activate            # Windows；Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -q                  # 23项冒烟测试应全部通过
```

（同时提供 `environment.yml` 供 conda 用户使用。）入口脚本在运行时自动将 `src/` 加入 `sys.path`，无需 `pip install .`。已发布的NNP检查点（`results/formal_nnp/best.pt`）与小评估子集随仓库跟踪，安装后即可直接运行 `python eval_checkpoint.py`。完整HDF5描述符集由 `python examples/00_make_fe_dataset.py` 生成，该脚本读取 `data/raw/byggmastar_fe/` 下的公开Byggmästar等（2022）XYZ数据库；原始数据库约1.6 GB，按README指引下载，不在本文重新分发。代码以MIT许可证发布。

### 3.2 架构

SmartFrac以Python 3.11编写，基于PyTorch，原子结构处理使用ASE（Larsen等, 2017），SOAP描述符使用DScribe（Himanen等, 2020）。包内子模块组织如下：

| 子包 | 职责 |
|---|---|
| `core/` | `BaseModel`（权重+配置序列化）、`BaseTrainer`、`datatypes`，以及文件解析器使用的插件式 `registry` |
| `io_utils/` | 物理清洗、去重、增广、标准化、分层划分、压缩HDF5存储；`parsers/`按扩展名注册ASE/LAMMPS-dump/VTK读取器 |
| `nnp/` | SOAP描述符封装、双头MLP、能量/力损失、指标、训练器、ASE `Calculator`适配器 |
| `pinn/` | 平面应力PDE残差、边界条件模块、裂尖集中采样器、训练器，以及Westergaard基准（`benchmarks/westergaard.py`） |
| `md/` | 裂纹初始构型模板（二维中心裂纹、三维单边缺口梁）、ASE Langevin NVT运行器、单步壁钟计时器 |
| `eval/` | 已发布NNP检查点与训练后PINN的独立复算入口 |
| `viz/` | 300 dpi的 parity图、损失曲线、二维位移场对比 |

三条工程约定被强制执行，因为它们是后续耦合工作的承重假设：(i) 所有模型继承 `BaseModel`，检查点自带超参数与归一化统计；(ii) 文件读取器按扩展名经注册器选择，新格式不触碰预处理流水线；(iii) 所有物理常量集中于 `constants.py`，原子尺度单位为Å/eV，连续介质尺度为Pa/m，接口处显式换算。覆盖清洗、可逆归一化、分层划分、模型前向形状与Westergaard评估器的23项冒烟测试在项目虚拟环境中全部通过。

### 3.3 软件功能

仓库对外暴露以下用户操作，每一项对应 `examples/` 下一个可运行脚本：

1. **数据集构建。** `examples/00_make_fe_dataset.py` 读取社区XYZ数据库（此处为Byggmästar等, 2022），施加物理过滤，计算逐原子SOAP向量，将构型—原子索引区间写入单一压缩HDF5文件。
2. **NNP训练。** `examples/02c_train_nnp_formal.py` 训练双头MLP（能量头147→256→128→1；力头147→256→128→3），使用Adam、余弦退火与梯度裁剪；检查点写入 `results/`。
3. **检查点复算。** `eval_checkpoint.py` 加载已发布的 `best.pt`，施加存储的归一化，在保留测试集上报告表1指标。
4. **PINN求解。** `examples/04_solve_pinn.py` 训练4×64 tanh平面应力PINN，采用裂尖集中配点，报告残差损失曲线。
5. **Westergaard对比。** `eval_pinn_paper.py` 在同一网格上评估训练后PINN与Westergaard闭式解，生成并排场图。
6. **MD计时工具。** `examples/06_md_speedup.py` 构建裂纹模板，将NNP安装为ASE计算器，计时单步MD。经典势对照计时在本版本中为桩函数；头对头加速比数字待 `nnp/ase_calculator.py` 中能量自洽力路径补全后给出（见第6节）。

## 4. 可复算示例

本节所有数字均可从仓库根目录运行所列脚本复现。已发布检查点随仓库跟踪，`eval_checkpoint.py` 无需任何下载即可运行；完整HDF5描述符集按需由 `examples/00_make_fe_dataset.py` 重新生成。

### 4.1 原子链路：已发布检查点的独立复算

运行

```bash
python eval_checkpoint.py
```

重新加载 `results/formal_nnp/best.pt` 与预构建HDF5文件，打印表1。在独立保留的2567个构型测试集上，测得误差为：

**表1.** 已发布检查点独立复算的NNP测试集精度。

| 量 | 数值 | 原型目标 |
|---|---|---|
| 能量MAE | 42.5 meV/atom | < 50 meV/atom |
| 能量RMSE | 87.5 meV/atom | — |
| 力MAE | 0.397 eV/Å | — |
| 力RMSE | 0.815 eV/Å | < 0.2 eV/Å |
| 能量Pearson r | 0.983 | — |
| 平均每原子能量 | −7.888 eV/atom | — |

能量目标达成：42.5 meV/atom约为α-Fe实验结合能（~4.3 eV/atom）的1%。力RMSE为设计目标的4倍；这是双头设计的预期特征——力头为独立回归器而非总能量的负梯度（Behler and Parrinello, 2007）——诚实地记录为能量自洽再训练的第一项。

为定位原型与已发表机器学习铁势的差距，表2列出代表性已报告误差。成熟的能量自洽铁势能量误差达数meV/atom、力RMSE达0.05–0.2 eV/Å；本原型在能量上高出一个数量级，力上更远。该差距直接来自双头非能量自洽设计、缺乏描述符自动微分，以及单轮CPU训练未做超参数优化。

**表2.** 代表性机器学习铁势与本原型的已报告精度对比。能量与力误差按所引文献原值列出；MAE与RMSE分行标注，不混列。

| 模型 | 能量误差 | 力RMSE (eV/Å) | 来源 |
|---|---|---|---|
| 本工作（双头） | 42.5 meV/atom MAE | 0.815 | — |
| Fe深度NNP（Fe–H） | 4.8 meV/atom RMSE | 0.072 | Zhang等（2023） |
| GAP, Fe–Cr–Ni合金 | 6 meV/atom（测试集） | 0.2–0.4 | Shenoy等（2023） |
| 进化策略势（Fe–C–H） | 5.1 meV/atom RMSE | 0.096 | Meng等（2025） |

![图1. NNP预测与参考的逐原子能量对比（2567个构型测试集，红色虚线y=x）。](figures/nnp_parity.png)

![图2. 训练损失随epoch变化（对数纵轴），300 epoch余弦退火。](figures/nnp_loss.png)

### 4.2 连续介质链路：PINN与Westergaard对比

运行

```bash
python eval_pinn_paper.py
```

训练4×64 tanh MLP 3000次Adam迭代，内部点8000个（20%集中于各裂尖无量纲半径0.5圆内，圆盘面积均匀），边界点1000个。坐标与位移在训练前无量纲化至O(1)，使Pa·m⁻¹平衡残差与m量级边界失配在数值上可比；绘图时再映射回物理参数（a=5 mm，σ∞=200 MPa，E=210 GPa，ν=0.3）。组合损失从约7.5×10⁻⁴降至约3×10⁻⁵。图3在同一评估网格上比较PINN位移场与Westergaard闭式解：u_x与u_y均复现从裂尖发出的扇形瓣状结构，符号与对称性正确。由于原型尚未沿内部裂纹面施加无拉力Neumann条件，本文不报告位移场的定量相对L2误差；该指标将在该边界条件补全后作为第一项基准重跑（第6节）。定性一致已确认PINN学到了正确的I型弹性解族。

![图3. PINN位移场与Westergaard闭式解对比（二维中心裂纹板，a=5 mm，σ∞=200 MPa）。左：Westergaard参考；右：PINN。上：u_x；下：u_y。](figures/pinn_field.png)

需指出，按近期文献，裂尖附近局部集中配点只是奇异场的最简单基线，既不最精确也不最经济。将Williams近尖基函数嵌入网络结构的富集PINN（Gu等, 2022）、基于复势的全纯神经网络（Calafà等, 2025）、以及由构造满足平衡的Kolosov–Muskhelishvili信息网络（Zhou等, 2026）均证明裂尖奇异性无需密集裂尖配点即可表达。本代码刻意未采用这些架构；引入富集形式是连续介质侧自然的升级方向，分离式模块布局使其可作为 `pinn/model.py` 的直接替换。

### 4.3 实现说明

本节所有数字均来自仓库中重新加载已发布检查点与HDF5文件的评估脚本；图由 `viz/` 模块以300 dpi生成。两点工程发现值得记录：(i) 将SOAP描述符预计算至HDF5，把核函数移出训练循环，使300个CPU epoch代价低廉——对单元素材料，这比端到端图网络是更好的首选设计；(ii) 单位归一化控制PINN训练：早期一棵使用SI单位的求解器产生的损失中，Pa·m⁻¹残差比m量级边界失配高出数十个数量级，必须将坐标与位移重标度至O(1)，优化器才能同时降低两项。

## 5. 影响

SmartFrac面向三类用途：

1. **教学。** 包小到一下午可读完。它以一条连贯流程演示DFT数据库如何变成缓存描述符集、双头MLP如何训练与评估、PINN如何由平面应力方程的自动微分构建、闭式基准如何接入。
2. **研究脚手架。** 课题组开始AI-for-fracture课题时可fork本仓库，一次替换一个模块——能量自洽力、富集裂尖基、裂纹面Neumann条件、尺度耦合接口——而无需重写数据流水线或指标接口。
3. **基线与基准。** 已发布的检查点、HDF5描述符集与指标词汇，为在同一α-铁数据库与同一Westergaard问题上比较更重的GPU架构（MACE、NequIP、NEP）与富集PINN变体，提供了仅CPU可跑的基线。

在研究者需要透明可跑参考而非黑盒生产框架的场景下，本软件将发挥作用。

## 6. 局限与未来工作

第一阶段原型明确**不**耦合原子与连续介质尺度，**不**量化不确定性，**不**产生能量自洽的力。每一项局限对应代码中具体的下一步：(i) 以自动微分替换独立力头，包括描述符梯度；(ii) 在内部裂纹面施加无拉力Neumann条件并重跑Westergaard L2基准；(iii) 引入单一耦合点，将NNP计算的内聚律传递给连续介质域；(iv) 加入不确定性估计；(v) 补全计算器力路径以闭合MD加速比。分离式架构已为这些插入做好准备。

## 7. 结论

SmartFrac是一个最小化、CPU可跑、完全可复现的Python脚手架，将BCC α-铁上的SOAP神经网络原子势（保留测试集能量MAE 42.5 meV/atom）与复现定性Westergaard I型场的平面应力PINN结合在一起。其持久产物不是最先进的数字，而是共享数据契约、统一断裂指标集，以及一份代码级的剩余缺口清单。仓库本身是首要产物；本文档说明如何阅读它。

---

## 致谢

<基金致谢待补；例如：受<基金名称>第<编号>号资助。计算资源由<超算中心>提供。作者感谢Byggmästar及其合作者公开发布本文所用铁DFT数据库。>

---

## 软件与数据可用性

SmartFrac源代码、训练检查点与预构建HDF5描述符文件以开源形式发布于 https://github.com/shishijingyu/SmartFrac 。底层DFT数据库为Byggmästar等（2022）的公开发布版，引用见参考文献。

---

## 参考文献

Bartók, A. P., Payne, M. C., Kondor, R., & Csányi, G. (2010). Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. *Physical Review Letters*, 104(13), 136403.

Batatia, I., Kovács, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems*, 35, 11423–11436.

Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453.

Behler, J. (2021). Four generations of high-dimensional neural network potentials. *Chemical Reviews*, 121(16), 10037–10076.

Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.

Byggmästar, J., Nikoulis, G., Fellman, A., Granberg, F., Djurabekova, F., & Nordlund, K. (2022). Multiscale machine-learning interatomic potentials for ferromagnetic and liquid iron. *Journal of Physics: Condensed Matter*, 34, 305402. (arXiv:2201.10237)

Calafà, M., Jensen, H., Engsig-Karup, A. P., & Andriollo, T. (2025). Solving plane crack problems via enriched holomorphic neural networks. *Engineering Fracture Mechanics*, 317, 111133.

Fan, Z., Dong, H., Ding, J., & Wang, T. (2021). Neuroevolution machine learning potentials: Combining high accuracy and low cost in atomistic simulations and application to heat transport. *Physical Review B*, 104, 104309.

Gu, Y., Zhang, C., Zhang, P., & Golub, M. V. (2022). Enriched physics-informed neural networks for in-plane crack problems: Theory and MATLAB codes. arXiv:2206.08750.

Himanen, L., Jäger, M. O. J., Morooka, E. V., Federici Canova, F., Ranawat, Y. S., Gao, D. Z., Rinke, P., & Foster, A. S. (2020). DScribe: Library of descriptors for machine learning in materials science. *Computer Physics Communications*, 247, 106949.

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.

Larsen, A. H., Mortensen, J. J., Blomqvist, J., Castelli, I. E., Christensen, R., Dulak, M., Friis, J., Groves, M. N., Hammer, B., Hargus, C., Hermes, E. D., Jennings, P. C., Jensen, P., Kermode, J., Kitchin, J. R., Kolsbjerg, E. L., Kubal, J., Kaasbjerg, K., Lysgaard, S., … Jacobsen, K. W. (2017). The atomic simulation environment—A Python library for working with atoms. *Journal of Physics: Condensed Matter*, 29(27), 273002.

Meng, F.-S., Shinzato, S., Zhao, Z., Du, J.-P., Gao, L., Fan, Z., & Ogata, S. (2025). Achieving empirical potential efficiency with DFT accuracy: A neuroevolution potential for the α-Fe–C–H system. arXiv:2510.17151.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., & Müller, K.-R. (2018). SchNet—A deep learning architecture for molecules and materials. *Journal of Chemical Physics*, 148(24), 241722.

Shenoy, L., Woodgate, C. D., Staunton, J. B., Bartók, A. P., Becquart, C. S., Domain, C., & Kermode, J. R. (2023). A collinear-spin machine learned interatomic potential for Fe₇Cr₂Ni alloy. arXiv:2309.08689.

Tada, H., Paris, P. C., & Irwin, G. R. (2000). *The Stress Analysis of Cracks Handbook* (3rd ed.). ASME Press.

Unke, O. T., Chmiela, S., Sauceda, H. E., Gastegger, M., Poltavsky, I., Schütt, K. T., Tkatchenko, A., & Müller, K.-R. (2021). Machine learning force fields. *Chemical Reviews*, 121(16), 10142–10186.

Westergaard, H. M. (1939). Bearing pressures and cracks. *Journal of Applied Mechanics*, 61, A49–A53.

Yang, W., Feng, X.-Q., & Gao, H. (2026). Outstanding issues and emerging frontiers in fracture mechanics. *International Journal of Fracture*, 250, 10.

Zhang, L., Han, J., Wang, H., Car, R., & E, W. (2018). Deep potential molecular dynamics: A scalable model with the accuracy of quantum mechanics. *Physical Review Letters*, 120(14), 143001.

Zhang, S., Meng, F., Fu, R., & Ogata, S. (2023). Highly efficient and transferable interatomic potentials for α-iron and α-iron/hydrogen binary systems using deep neural networks. arXiv:2311.18686.

Zhou, S., Haeffner, C., Wang, S., Stebner, S., Liao, Z., Yang, B., Wei, Z., & Muenstermann, S. (2026). Transfer-learned Kolosov–Muskhelishvili informed neural networks for fracture mechanics. *Theoretical and Applied Fracture Mechanics* (in press). arXiv:2601.00491.
