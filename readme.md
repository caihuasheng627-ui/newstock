# 代码说明

## 环境配置

- Python: `>=3.10,<3.13`（Docker 镜像内为 3.12）
- 包管理: `uv`（依赖锁定见 `uv.lock`）
- 主要依赖版本见 `pyproject.toml`，核心包括：
  - `torch>=2.6.0`
  - `pandas>=2.3.2`
  - `scikit-learn>=1.7.2`
  - `ta-lib>=0.6.8`
  - `joblib`、`tqdm`、`tensorboardX`
- 系统依赖：需安装 TA-Lib C 库（Dockerfile 已包含编译安装）
- Docker 镜像名：`bdc2026`
- 工作目录：`/app`

本地开发：

```bash
uv sync
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

## 数据

使用沪深300成分股日行情数据（通过 baostock / 公开接口下载），与大赛基准代码一致：

- 原始数据：`data/stock_data.csv`
- 训练集：`data/train.csv`
- 测试集：`data/test.csv`（用于本地评分参考）
- 数据获取脚本：`get_stock_data.py`
- 划分脚本：`data/split_train_test.py`

复现训练/预测时**不联网**，直接读取镜像或挂载目录中的本地 CSV。

## 预训练模型

未使用外部预训练权重。模型从随机初始化开始训练（Xavier 初始化），训练产物保存在 `model/`。

当前提交使用的主模型目录：`model/30_158+39/`（`sequence_length=30`，特征 `158+39`），包含多随机种子权重：

- `seed_42/best_model.pth`
- `seed_123/best_model.pth`
- 以及对应的 `scaler.pkl`、`stockid2idx.pkl`、`train_medians.pkl`、`leader_stock_ids.pkl`、`config.json`

## 算法

### 整体思路介绍

将“预测未来一周收益最大的股票组合”建模为**日内横截面排序**问题：

1. 对每只股票构建固定长度历史窗口特征序列；
2. 用时序 Transformer 编码单票历史，再用跨股票注意力建模同日相对强弱；
3. 输出每只股票的排序分数，取 Top-K 并按预设权重构成组合。

标签定义：\((P^{open}_{T+5}-P^{open}_{T+1})/P^{open}_{T+1}\)，与赛方评估口径一致。

### 方法的创新点（如果有）

- **Top-K ListMLE + 可微组合收益**联合损失，直接对齐“重仓头部股票”目标；
- **龙头股池筛选**（按近期成交额保留头部比例），降低噪声票干扰；
- **多 seed 集成**预测时取均值，提升稳定性；
- 推理阶段对极端波动股票做风险过滤。

### 网络结构

`StockTransformer`（`code/src/model.py`）：

1. 输入投影 + 正弦位置编码
2. TransformerEncoder 时序编码
3. FeatureAttention 时间维聚合
4. CrossStockAttention 多层跨股票交互（padding mask 屏蔽无效票）
5. ranking_layers + score_head 输出排序分数

关键超参（见 `code/src/config.py`）：`d_model=128`，`nhead=4`，`num_layers=2`，`cross_stock_layers=2`，`sequence_length=30`。

### 损失函数

`CombinedRankingLoss`：

- Top-K ListMLE（默认 `topk_mle_k=3`）
- 可微 SoftTopK 组合收益辅助项（`portfolio_loss_weight=0.4`）

选模指标：验证集加权组合绝对收益 `weighted_port_return`。

### 数据扩增

未使用随机数据增强。主要数据处理包括：

- `158+39` 特征工程（Alpha158 风格特征 + 技术指标）
- 标签 1%/99% 分位截断
- 训练集中位数填补缺失、StandardScaler 标准化
- 按最后一段时间划分验证集，并保留序列上下文

### 模型集成

训练 `ensemble_seeds=[42, 123]`；推理时加载各 seed 的 `best_model.pth`，对分数取均值后再排序选股。

### 算法的其他细节

- 输出 Top5，权重：`[0.50, 0.47, 0.01, 0.01, 0.01]`
- `train_leader_ratio=0.75`，`leader_lookback_days=30`
- 推理可限制在训练保存的龙头池内
- 固定随机种子以保证可复现

## 训练流程

入口：`train.sh` → `python code/src/train.py`

主要步骤（详见 `code/src/train.py` 注释）：

1. `set_seed` 固定随机性
2. 读取 `data/train.csv`
3. 按时间切分训练/验证，并筛选龙头股池
4. 多进程特征工程（`featurework.py` → `utils.py`）
5. 构建标签、标准化、保存 scaler/映射/中位数
6. 向量化构建按日排序样本
7. 训练 StockTransformer，早停并保存 `best_model.pth`
8. 对 `ensemble_seeds` 逐个训练

## 推理流程

入口：`test.sh` → `python code/src/test.py`

主要步骤（详见 `code/src/test.py` 注释）：

1. 定位模型目录并加载 `config.json` / scaler / 映射 / 龙头池
2. 读取 `data/train.csv`，取最新交易日做推理窗口
3. 与训练一致的特征工程与标准化
4. 构建长度 `sequence_length` 的序列
5. 风险过滤（极端 `volatility_20`）
6. 多 seed 模型打分并取均值
7. 输出 Top-K 权重到 `output/result.csv`

兼容入口：`python code/src/predict.py`（等价调用 `test.py`）。

## 其他注意事项

1. **目录规范**（Docker 内 WORKDIR=`/app`）：

```text
|-- app
|   |-- code
|   |   |-- src
|   |       |-- featurework.py
|   |       |-- test.py
|   |       |-- train.py
|   |-- data          # compose 挂载
|   |-- model         # 镜像内
|   |-- output        # compose 挂载
|   |-- temp          # compose 挂载
|   |-- init.sh
|   |-- train.sh
|   |-- test.sh
|   |-- readme.md
```

2. 验证集划分：按最后约一个月交易日切分，并向前保留 `sequence_length` 天上下文。
3. 复现要求：固定种子；镜像运行期间不得联网；预测 ≤5 分钟，训练 ≤8 小时。
4. Docker：

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
docker save -o 队伍名称.tar bdc2026:latest
```

5. `data/run.sh` 默认执行 `init.sh` + `test.sh`（训练步骤可按复现需要取消注释 `train.sh`）。
6. 本地打分参考：`python test/score_self.py` 或 `python code/src/score_self.py`。
