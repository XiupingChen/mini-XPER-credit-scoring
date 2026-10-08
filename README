# Mini-XPER 信用评分示例

这是一个简单的信用评分小项目，用来演示 XPER 的基本思路：把模型整体预测表现的变化分解为不同特征的贡献。

项目使用一组模拟的借款人数据，包含以下三个特征：

- `stable_income`：收入是否稳定；
- `low_debt`：债务水平是否较低；
- `no_overdue`：是否没有逾期记录。

程序使用 `1 - Brier loss` 衡量模型表现，并通过 Shapley value 计算每个特征对整体模型表现的贡献。所有特征贡献之和等于完整模型表现与基准模型表现之差。

## 文件说明

- `xper_demo.py`：模拟信用评分数据并计算特征的表现贡献；
- `README.md`：项目介绍和运行说明。

## 运行方法

本项目只需要 Python 3，不需要安装其他第三方库。在终端进入项目文件夹后运行：

```bash
python3 xper_demo.py
```

程序会输出基准模型表现、完整模型表现，以及三个特征各自的贡献。

## 说明

这个程序是帮助理解 XPER 思路的小型示例，并不是对论文全部方法和实证结果的完整复现。

本项目使用 ChatGPT/Codex 辅助梳理代码结构和检查运行结果，代码已经在本地运行验证。

## 参考文献

Hué, S., Hurlin, C., Pérignon, C., & Saurin, S. (2026). Measuring the driving forces of predictive performance: Application to credit scoring. *Management Science*.
