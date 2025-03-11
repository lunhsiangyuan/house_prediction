# Kaggle房價預測競賽專案

## 專案概述
本專案是針對Kaggle房價預測競賽的完整解決方案，目標是預測美國愛荷華州艾姆斯市住宅房屋的銷售價格。我們使用了多種機器學習模型和優化技術，包括特徵工程、模型訓練、超參數調優和集成學習。

## 評估指標
競賽使用**對數轉換後的均方根誤差(RMSLE)**作為評估標準：
- 對預測結果和實際銷售價格取對數再計算RMSE
- 這種方法讓低價房屋和高價房屋的預測誤差權重相同

## 專案結構
```
house_prices_data/
├── dataset/                      # 原始數據集
├── preprocessing_results/        # 數據預處理結果
├── feature_engineering_results/  # 特徵工程結果
│   └── advanced/                 # 高級特徵工程結果
├── model_results/                # 模型訓練結果和摘要
├── data_preprocessing.py         # 數據預處理腳本
├── feature_engineering.py        # 基本特徵工程腳本
├── advanced_feature_engineering.py # 高級特徵工程腳本
├── train_simple_xgboost.py       # 簡單XGBoost模型訓練腳本
├── train_fixed_xgboost.py        # 優化XGBoost模型訓練腳本
├── bayesian_xgboost_model.py     # 貝葉斯優化XGBoost模型腳本
├── stacking_ensemble_model.py    # 堆疊集成模型腳本
├── run_optimization.py           # 優化執行腳本
├── optimization_summary.md       # 優化總結
├── TODO.md                       # 專案待辦事項清單
└── README.md                     # 專案說明文件
```

## 主要成果
我們實施了三種主要的優化方案，以提高房價預測模型的性能：

1. **高級特徵工程**：
   - 多項式特徵：為重要特徵創建二次多項式特徵，捕捉非線性關係
   - 主成分分析 (PCA)：創建15個主成分特徵，累積解釋方差達73.58%
   - 聚類特徵：使用K-means聚類創建聚類標籤和距離特徵

2. **貝葉斯優化XGBoost**：
   - 使用貝葉斯優化方法調整XGBoost超參數
   - 相比傳統的網格搜索和隨機搜索，更有效地探索超參數空間
   - 交叉驗證RMSE: 0.125640 (對數空間)

3. **堆疊集成模型**：
   - 結合7個基礎模型：Ridge、Lasso、ElasticNet、隨機森林、梯度提升樹、XGBoost、LightGBM
   - 使用Ridge回歸作為元模型，學習如何最佳組合基礎模型的預測
   - 交叉驗證RMSE: 0.124580 (對數空間)

## 模型性能比較

| 模型 | 交叉驗證 RMSE (對數空間) | 訓練時間 | 參數優化 |
|------|------------------------|---------|---------|
| XGBoost (優化) | 0.125871 | ~3.5分鐘 | 網格搜索 + 隨機搜索 |
| LightGBM | 0.134672 | ~0.1分鐘 | 預設參數微調 |
| 隨機森林 | 0.142635 | ~1.2分鐘 | 基本參數設定 |
| SVR | 0.147298 | ~5分鐘 | 僅使用最重要特徵 |
| 集成模型 (加權平均) | 0.135640 | ~0.2分鐘 | XGBoost(0.40), LightGBM(0.40), RF(0.20) |
| 貝葉斯優化XGBoost | 0.125640 | ~0.1分鐘 | 貝葉斯優化 |
| 堆疊集成 | 0.124580 | ~0.2分鐘 | 堆疊集成 (7個基礎模型) |

## 使用方法

### 環境設置
```bash
pip install -r requirements.txt
```

### 數據預處理和特徵工程
```bash
python data_preprocessing.py
python feature_engineering.py
python advanced_feature_engineering.py
```

### 模型訓練
```bash
# 訓練基本模型
python train_simple_xgboost.py
python train_fixed_xgboost.py

# 訓練優化模型
python bayesian_xgboost_model.py
python stacking_ensemble_model.py

# 或使用優化執行腳本一次性執行所有優化
python run_optimization.py
```

### 查看結果
訓練完成後，可以在`model_results/@summary.html`中查看詳細的模型比較和分析結果。

## 結論與建議

1. **最佳模型**：堆疊集成模型表現最佳，RMSE為0.124580，略優於貝葉斯優化XGBoost和原始優化XGBoost。

2. **效率考量**：
   - LightGBM在速度和性能之間取得了良好的平衡，訓練時間短且性能不錯
   - 貝葉斯優化XGBoost在保持高性能的同時，大幅減少了訓練時間

3. **後續改進方向**：
   - 實施SHAP值分析和偏依存圖分析，進一步理解模型預測機制
   - 進行模型殘差分析，找出模型預測效果較差的情況
   - 嘗試神經網路模型，特別是針對特徵間的複雜交互
   - 開發模型API和網頁界面，實現實時預測服務

## 參考資源
- Kaggle競賽頁面：[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- 相關文獻中的房價預測方法論
- 過去競賽獲勝者的解決方案
