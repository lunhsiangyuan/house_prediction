# Intro to Data Cleaning, EDA, and Machine Learning Dataset

## 數據集概述

這是一個用於數據清理、探索性數據分析（EDA）和機器學習入門的教學數據集。

**作者**: Walekhwa Philip Tambiti Leo  
**發布日期**: 2024-10-11  
**數據集 ID**: `walekhwatlphilip/intro-to-data-cleaning-eda-and-machine-learning`

## 文件結構

```
intro-to-data-cleaning-eda-and-machine-learning/
├── bi.csv                                      # 主要數據文件
├── intro-to-data-cleaning-eda-and-machine-learning.html  # 教學筆記
└── README.md                                   # 本說明文件
```

## 數據集描述

### 主要數據文件: `bi.csv`

這是一個包含學生信息的數據集，包含以下欄位：

| 欄位名稱 | 描述 | 數據類型 |
|---------|------|----------|
| fNAME | 名字 | String |
| lNAME | 姓氏 | String |
| Age | 年齡 | Integer |
| gender | 性別 | String (M/F/Male/Female) |
| country | 國家 | String |
| residence | 居住地 | String |
| entryEXAM | 入學考試分數 | Integer |
| prevEducation | 先前教育程度 | String |
| studyHOURS | 學習時數 | Integer |
| Python | Python 分數 | Integer |
| DB | 數據庫分數 | Integer |

### 數據特徵

- **總記錄數**: 約 25 筆學生記錄
- **文件大小**: ~5KB
- **用途**: 數據清理、探索性數據分析、機器學習入門
- **適合技能等級**: 初學者到中級

## 學習目標

這個數據集特別適合學習：

1. **數據清理 (Data Cleaning)**
   - 處理缺失值
   - 數據類型轉換
   - 異常值檢測
   - 數據標準化

2. **探索性數據分析 (EDA)**
   - 描述性統計
   - 數據可視化
   - 相關性分析
   - 分佈分析

3. **機器學習入門**
   - 特徵工程
   - 數據預處理
   - 模型訓練
   - 預測分析

## 潛在的分析方向

### 1. 學術表現分析
- 分析入學考試分數與學科成績的關係
- 研究學習時數對成績的影響
- 探索先前教育背景對學術表現的影響

### 2. 人口統計分析
- 年齡分佈分析
- 性別比例統計
- 國籍/居住地分佈

### 3. 預測建模
- 預測學生 Python 成績
- 預測學生數據庫成績
- 建立綜合學術表現預測模型

## 數據清理挑戰

這個數據集包含一些典型的數據清理挑戰：

1. **不一致的性別表示**: M/Male, F/Female
2. **缺失值**: 某些記錄可能有空的學科分數
3. **數據類型混合**: 需要適當的類型轉換
4. **拼寫變異**: 國家名稱可能有不同的拼寫方式

## 建議的分析流程

1. **數據載入與初步檢查**
   ```python
   import pandas as pd
   df = pd.read_csv('bi.csv')
   df.info()
   df.head()
   ```

2. **數據清理**
   - 處理缺失值
   - 標準化性別欄位
   - 檢查數據類型

3. **探索性數據分析**
   - 基本統計描述
   - 可視化分析
   - 相關性分析

4. **機器學習建模**
   - 特徵選擇
   - 數據分割
   - 模型訓練與評估

## 相關資源

- [Kaggle 原始數據集頁面](https://www.kaggle.com/datasets/walekhwatlphilip/intro-to-data-cleaning-eda-and-machine-learning)
- 教學筆記: `intro-to-data-cleaning-eda-and-machine-learning.html`

## 注意事項

- 這是一個教學用的合成數據集
- 數據規模較小，適合學習和實驗
- 包含真實世界中常見的數據品質問題

---

**下載時間**: 2024年12月  
**下載工具**: Kaggle MCP  
**建議開始**: 先閱讀 HTML 教學筆記，然後開始數據探索
