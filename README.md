# 股票分析系统 (Stock Analysis System)
## 提要
- 基于 https://github.com/DR-lin-eng/stock-scanner 二次修改，感谢原作者
- 需要在环境中配置DS_API_KEY 使用deepseek-v3分析
- stock_analyzer.py是分析a股的关键文件，在21-22行中记得填入apiurl防止无法使用ai分析

## 项目简介 (Project Overview)

这是一个专业的A股股票分析系统，提供全面的技术指标分析和投资建议。系统包括以下主要组件：
- 股票分析引擎
- 高级技术指标分析引擎

This is a professional A-share stock analysis system that provides comprehensive technical indicator analysis and investment recommendations. The system includes the following main components:
- Stock Analysis Engine
- Advanced Technical Indicator Analysis Engine

## 功能特点 (Key Features)

### 股票分析 (Stock Analysis)
- 实时计算多种技术指标
- 生成详细的股票分析报告
- 提供投资建议
- 支持单股和批量分析

### 全市场扫描 (Market-Wide Scanning)
- 扫描全部A股股票
- 根据多维度技术指标进行评分
- 筛选高潜力股票
- 按价格区间生成分析报告

## 技术指标 (Technical Indicators)
- 移动平均线 (Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 布林带 (Bollinger Bands)
- 能量潮指标 (OBV)
- 随机指标 (Stochastic Oscillator)
- 平均真实波动范围 (ATR)

## 系统依赖 (System Dependencies)
- Python 3.8+
- Pandas
- NumPy
- AkShare
- Markdown2

## 快速开始 (Quick Start)

### 安装依赖 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 运行应用 (Run Application)
#### 全市场股票扫描
```bash
python 全部股票分析推荐1.py
```

## 配置 (Configuration)
- 在 `.env` 文件中配置 deepseek API 密钥（DS_API_KEY）
- 可在 `stock_analyzer.py` 中调整技术指标参数

## 输出 (Outputs)
分析结果将保存在 `scanner` 目录下：
- `price_XX_YY.txt`：按价格区间的详细分析
- `summary.txt`：市场扫描汇总报告

## 注意事项 (Notes)
- 股票分析仅供参考，不构成投资建议
- 使用前请确保网络连接正常
- 建议在实盘前充分测试

## 贡献 (Contributing)
欢迎提交 issues 和 pull requests！

## 许可证 (License)
[待添加具体许可证信息]

## 免责声明 (Disclaimer)
本系统仅用于学习和研究目的，投资有风险，入市需谨慎。
