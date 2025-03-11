# 房價預測探索性數據分析摘要
# House Price Prediction EDA Summary

## 概述 (Overview)

本文檔總結了對Kaggle房價預測競賽數據集的探索性分析結果。分析從房產專家角度出發，聚焦於最能影響房屋銷售價格的關鍵特徵。

This document summarizes the exploratory data analysis (EDA) results of the Kaggle House Prices prediction competition dataset. The analysis takes a real estate expert's perspective, focusing on key features that most significantly impact house sale prices.

## 數據基本資訊 (Dataset Information)

- 訓練集大小 (Training set size): 1460 筆資料 × 81 個特徵 (1460 observations × 81 features)
- 測試集大小 (Test set size): 1459 筆資料 × 80 個特徵 (1459 observations × 80 features)
- 目標變量 (Target variable): SalePrice (銷售價格)

## 銷售價格統計摘要 (Sale Price Statistics)

```
平均值 (Mean)    : $180,921.20
標準差 (Std Dev) : $79,442.50
最小值 (Min)     : $34,900.00
25%分位數        : $129,975.00
中位數 (Median)  : $163,000.00
75%分位數        : $214,000.00
最大值 (Max)     : $755,000.00
```

## 關鍵發現 (Key Findings)

### 1. 與銷售價格相關性最高的特徵 (Features with highest correlation to sale price)

| 特徵 (Feature) | 說明 (Description) | 相關係數 (Correlation) |
|----------------|-------------------|---------------------|
| OverallQual    | 整體品質 (Overall Quality) | 0.7910 |
| GrLivArea      | 居住面積 (Living Area) | 0.7086 |
| GarageCars     | 車庫容量 (Garage Capacity) | 0.6404 |
| GarageArea     | 車庫面積 (Garage Area) | 0.6234 |
| TotalBsmtSF    | 地下室總面積 (Total Basement Area) | 0.6136 |
| FullBath       | 全套浴室數量 (Full Bathrooms) | 0.5607 |
| YearBuilt      | 建造年份 (Year Built) | 0.5229 |
| YearRemodAdd   | 翻修年份 (Year Remodeled) | 0.5071 |

這表明整體品質和居住面積是影響房價的最重要因素。
This indicates that overall quality and living area are the most important factors affecting house prices.

### 2. 最高價社區 TOP 5 (Top 5 highest-priced neighborhoods)

| 社區 (Neighborhood) | 平均價格 (Average Price) |
|---------------------|------------------------|
| NoRidge             | $335,295.32 |
| NridgHt             | $316,270.62 |
| StoneBr             | $310,499.00 |
| Timber              | $242,247.45 |
| Veenker             | $238,772.73 |

房屋所在社區對價格有顯著影響，頂級社區的房屋價格可以比低價社區高出3倍。
The neighborhood has a significant impact on house prices, with top neighborhoods commanding prices up to 3 times higher than lower-priced areas.

### 3. 整體品質與價格的關係 (Relationship between overall quality and price)

最高品質(10分)的房屋平均價格是最低品質(1分)的8.7倍。這表明房屋品質可能是最能影響價格的單一因素。

The average price of houses with the highest quality (10 points) is 8.7 times that of houses with the lowest quality (1 point). This suggests that house quality may be the single most influential factor on price.

### 4. 面積與價格 (Area and price)

- 居住面積與價格的相關性 (Correlation between living area and price): 0.7086
- 每增加1000平方呎的居住面積，價格平均增加約 (Price increase per 1000 sq ft of living area): $119,383.39

### 5. 建築類型影響 (Building type impact)

| 建築類型 (Building Type) | 說明 (Description) | 平均價格 (Average Price) |
|--------------------------|-------------------|------------------------|
| 1Fam                     | 單一家庭獨立屋 (Single-family Detached) | $185,763.81 |
| TwnhsE                   | 連棟屋盡頭單元 (Townhouse End Unit) | $181,959.34 |
| Twnhs                    | 連棟屋內部單元 (Townhouse Inside Unit) | $135,911.63 |
| Duplex                   | 雙拼屋 (Duplex) | $133,541.08 |
| 2fmCon                   | 雙家庭轉換住宅 (Two-family Conversion) | $128,432.26 |

單一家庭獨立屋的平均價格最高，比雙家庭轉換住宅高出約45%。
Single-family detached houses have the highest average price, about 45% higher than two-family conversions.

### 6. 其他影響因素 (Other influential factors)

- **房屋年齡** (House Age): 較新的房屋通常價格更高，但高品質的老房子可能比低品質的新房子價格更高。
- **廚房品質** (Kitchen Quality): 廚房品質從低到高 (Po -> Ex) 對應的房價呈現明顯的階梯式上升。
- **車庫容量** (Garage Capacity): 車位數量與房價呈正相關，3車位車庫的房屋平均價格遠高於1車位車庫的房屋。

## 圖表說明 (Charts Description)

所有圖表已保存到 `eda_results/plots/` 目錄下：

1. **price_distribution.png**: 房屋銷售價格分佈及其對數轉換
   (House sale price distribution and its logarithmic transformation)
   
2. **correlation_heatmap.png**: 數值特徵與銷售價格的相關性熱圖
   (Correlation heatmap between numeric features and sale price)
   
3. **neighborhood_prices.png**: 不同社區的房價分佈箱型圖
   (Box plot of house prices across different neighborhoods)
   
4. **quality_vs_price.png**: 不同整體品質等級的房價分佈
   (House price distribution by overall quality rating)
   
5. **area_vs_price.png**: 居住面積與銷售價格的關係(按整體品質著色)
   (Relationship between living area and sale price, colored by overall quality)
   
6. **year_vs_price.png**: 建造年份與銷售價格的關係(按整體品質著色)
   (Relationship between year built and sale price, colored by overall quality)
   
7. **building_type_prices.png**: 不同建築類型的房價分佈
   (House price distribution by building type)
   
8. **kitchen_quality_prices.png**: 不同廚房品質的房價分佈
   (House price distribution by kitchen quality)
   
9. **garage_capacity_prices.png**: 不同車庫容量的房價分佈
   (House price distribution by garage capacity)
   
10. **age_vs_price.png**: 房屋年齡與銷售價格的關係
    (Relationship between house age and sale price)

## 預測建模啟示 (Implications for Predictive Modeling)

基於EDA分析結果，建議在建模時特別關注：

1. **特徵工程** (Feature Engineering):
   - 創建新特徵：房屋年齡(YrSold-YearBuilt)、翻修狀態(YrSold-YearRemodAdd)
   - 為離散特徵使用適當的編碼方法，特別是社區(Neighborhood)和品質評分類特徵
   - 處理異常值，特別是在面積相關特徵中

2. **模型選擇** (Model Selection):
   - 使用能捕捉特徵間交互作用的模型(如隨機森林、梯度提升樹)
   - 考慮線性模型(如彈性網)作為基線模型

3. **特徵重要性** (Feature Importance):
   - 優先關注與銷售價格高度相關的特徵
   - 不要忽略類別特徵(如社區、建築類型)的影響

從房產專家角度來看，地點(Neighborhood)、品質(OverallQual、KitchenQual)和面積(GrLivArea、TotalBsmtSF)是決定房價的關鍵因素。

From a real estate expert's perspective, location (Neighborhood), quality (OverallQual, KitchenQual), and area (GrLivArea, TotalBsmtSF) are the key factors determining house prices.
