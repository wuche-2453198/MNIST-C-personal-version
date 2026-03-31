# MNIST C++ 手写数字识别项目

## 项目简介

这是一个使用**纯 C++ 从零实现**的 MNIST 手写数字识别项目，目标是通过最基础的方式理解神经网络的核心流程，而不是依赖现成深度学习框架直接调用接口。

项目当前已经实现了以下内容：

- MNIST `idx` 二进制文件读取
- 图像与标签解析
- 单隐藏层全连接神经网络
- 前向传播
- 反向传播
- 基于均方误差（MSE）的训练与评估
- 固定学习率训练实验
- 不同学习率的公平对比实验
- 训练集 / 测试集分离评估
- 每个 `epoch` 的样本打乱（shuffle）
- 最佳测试集结果记录

这个项目更偏向**教学版 / 原理版实现**，适合用来学习：

- MNIST 数据集的基本组织方式
- 神经网络前向传播与反向传播的代码落地
- 学习率、样本量、shuffle、训练 / 测试划分对结果的影响
- 从“能跑”到“能分析实验现象”的完整实践过程

## 项目特点

- **不依赖深度学习框架**：没有使用 PyTorch、TensorFlow 等框架。
- **便于理解原理**：核心逻辑都由 C++ 手工实现，结构清晰。
- **支持多种调试模式**：既能查看数据，也能做训练实验。
- **适合作为课程作业复现项目**：方便结合实验报告撰写分析过程与结论。

## 项目结构

```text
MNIST C++/
├── main.cpp              # 程序入口，包含多种调试 / 实验模式
├── mnist_reader.h/.cpp   # MNIST 图像与标签读取
├── network.h/.cpp        # 网络结构、初始化、前向传播
├── trainer.h/.cpp        # 反向传播、梯度计算、参数更新、训练流程
├── evaluator.h/.cpp      # 损失与准确率评估
├── math_utils.h/.cpp     # 数学辅助函数
├── images/               # 本地 MNIST 数据集目录（默认不上传）
├── README.md             # 项目说明文档
└── .gitignore            # Git 忽略规则
```

## 环境要求

- 操作系统：Windows（当前项目主要在 Windows 环境下完成）
- 编译器：
  - `g++` / `clang++`（支持 C++17）
  - 或 Visual Studio / MSVC

## 数据集准备

本仓库默认**不上传 MNIST 原始数据文件**和编译生成的可执行文件。

请你自行准备以下 4 个文件，并放入项目下的 `images/` 目录：

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

目录示例：

```text
images/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

## 编译方法

### 方式一：使用 g++ / MinGW

在项目根目录下执行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp mnist_reader.cpp network.cpp trainer.cpp evaluator.cpp math_utils.cpp -o main.exe
```

如果你使用 GCC / Clang，也可以把 `-O2` 换成 `-O3`。

注意：

- `-O3` 中的 `O` 是**大写字母 O**，不是数字 `0`，也不是小写字母 `o`
- 如果你使用的是 **MSVC**，应使用 `/O2`，而不是 `-O3`

### 方式二：使用 MSVC

在 Developer Command Prompt 中执行：

```bash
cl /EHsc /std:c++17 /O2 main.cpp mnist_reader.cpp network.cpp trainer.cpp evaluator.cpp math_utils.cpp /Fe:main.exe
```

## 运行方式

程序支持 5 种模式。

### 交互方式运行

```bash
.\main.exe
```

运行后根据提示输入模式编号和数据集路径。

### 模式说明

- **模式 1**：图像浏览
  - 用于查看 MNIST 图像与标签是否读取正确
- **模式 2**：前向传播测试
  - 用于检查网络输出维度、预测标签和目标向量
- **模式 3**：固定学习率训练测试
  - 默认使用部分训练样本进行若干轮训练，输出每轮 loss 和 accuracy
- **模式 4**：不同固定学习率训练效果比较
  - 对不同学习率进行公平对比
  - 已处理“同一起跑线”问题：对比实验使用相同初始网络与相同 shuffle 随机种子
- **模式 5**：训练集训练 + 测试集评估
  - 支持训练集 / 测试集分离
  - 支持配置训练样本数、测试样本数和 `epoch` 数
  - 每轮输出训练集与测试集的 loss / accuracy
  - 自动记录最佳测试集结果

## 命令行示例

### 模式 3：固定学习率训练

```bash
.\main.exe images\train-images.idx3-ubyte images\train-labels.idx1-ubyte 3
```

### 模式 4：不同学习率比较

```bash
.\main.exe images\train-images.idx3-ubyte images\train-labels.idx1-ubyte 4
```

### 模式 5：训练集训练 + 测试集评估

```bash
.\main.exe images\train-images.idx3-ubyte images\train-labels.idx1-ubyte images\t10k-images.idx3-ubyte images\t10k-labels.idx1-ubyte 5 10000 1000 100
```

命令含义如下：

```text
main.exe <训练图像> <训练标签> <测试图像> <测试标签> 5 <训练样本数> <测试样本数> <epoch数>
```

如果不在命令行中完整传参，程序也支持交互输入。

## 当前实验结果概览

在当前实现和本地实验设置下，得到过如下代表性结果：

- 训练样本数约 `1000` 时，最佳测试集准确率约为 `0.857`
- 训练样本数约 `5000` 时，最佳测试集准确率约为 `0.906`
- 训练样本数约 `10000` 时，最佳测试集准确率约为 `0.925`

这说明：

- 在当前项目阶段，**训练数据量**对泛化能力影响很明显
- 单纯增加训练轮数并不能无限提升测试集准确率
- shuffle 有帮助，但不是决定性因素
- 训练集准确率很高而测试集提升有限时，需要关注泛化能力而不只是继续堆 `epoch`

## 当前实现的局限性

这个项目的目标是“看懂原理并完成实验”，因此当前实现仍然有一些明显限制：

- 主要是**CPU 单线程串行训练**
- 采用逐样本训练，速度相对较慢
- 没有使用矩阵库、并行库或 GPU 加速
- 目前更适合教学与分析，不是工业级高性能实现

## 后续可扩展方向

如果以后继续迭代这个项目，可以考虑：

- 将逐样本训练改为 `mini-batch`
- 引入更高效的数据结构或矩阵库
- 尝试 Softmax + Cross Entropy
- 增加隐藏层规模或改进网络结构
- 使用多线程或 GPU 进行加速
- 使用 PyTorch / LibTorch 复现同一任务，对比原理版与框架版差异

## 适用场景

这个仓库特别适合以下用途：

- 人工智能导论 / 机器学习基础课程作业
- C++ 实现神经网络基础练习
- 反向传播流程理解与复现
- 实验报告撰写素材整理

## 说明

本项目仓库默认建议：

- **上传源代码与说明文档**
- **不上传可执行文件和原始 MNIST 数据集**

这样仓库会更整洁，也更适合公开分享与后续维护。
