# Kaggle房價預測競賽專案

## 專案概述
本專案是針對Kaggle房價預測競賽的完整解決方案，目標是預測美國愛荷華州艾姆斯市住宅房屋的銷售價格。我們使用了多種機器學習模型和優化技術，包括特徵工程、模型訓練、超參數調優、集成學習和殘差校準。

## 評估指標
競賽使用**對數轉換後的均方根誤差(RMSLE)**作為評估標準：
- 對預測結果和實際銷售價格取對數再計算RMSE
- 這種方法讓低價房屋和高價房屋的預測誤差權重相同
- 相對誤差率 = RMSE * 100%（顯示為百分比）

> **重要**: 所有模型評估必須遵循[RMSE計算指南](docs/rmse_calculation_guide.md)，確保先計算RMSE，再進行後續處理。

## 專案結構
```
house_prices_data/
├── dataset/                      # 原始數據集
├── preprocessing_results/        # 數據預處理結果
├── feature_engineering_results/  # 特徵工程結果
├── model_results/               # 模型訓練結果和摘要
│   └── @summary.html           # 模型性能比較摘要
├── docs/                        # 專案文檔
│   └── rmse_calculation_guide.md # RMSE計算指南
├── data_handling.py            # 數據加載和處理
├── model_definition.py         # 模型定義和初始化
├── ensemble_training.py        # 集成模型訓練邏輯
├── visualization.py            # 視覺化函數
├── utils.py                    # 工具函數
├── main_ensemble.py            # 主程序
├── setup_environment.py        # 環境設置
└── README.md                   # 專案說明文件
```

## 主要功能模塊

### 1. 數據處理 (`data_handling.py`)
- 數據加載和預處理
- 特徵轉換和標準化
- 訓練集和測試集處理

### 2. 模型定義 (`model_definition.py`)
- 基礎模型初始化
- 模型參數設置
- 模型工廠函數

### 3. 集成訓練 (`ensemble_training.py`)
- 多層堆疊集成
- 分段模型訓練
- 交叉驗證評估
- 殘差校準機制

### 4. 視覺化 (`visualization.py`)
- 模型性能比較圖
- 預測分布圖
- 分段模型性能圖

### 5. 工具函數 (`utils.py`)
- 結果保存和加載
- HTML摘要生成
- 輔助函數

### 6. 殘差校準 (`residual_calibration.py`)
- 預測誤差學習
- 自適應校準
- 分段誤差修正

## 模型性能比較

| 模型類型 | 模型名稱 | RMSE (對數空間) | 相對誤差率 | 訓練時間 | 改進率 |
|---------|---------|----------------|------------|---------|--------|
| 基礎模型 | Ridge   | 0.008856 | 0.89% | ~0.1分鐘 | - |
| 基礎模型 | Lasso   | 0.007671 | 0.77% | ~0.1分鐘 | - |
| 基礎模型 | ElasticNet | 0.007785 | 0.78% | ~0.1分鐘 | - |
| 優化模型 | Random Forest | 0.008148 | 0.81% | ~0.5分鐘 | - |
| 優化模型 | XGBoost | 0.017980 | 1.80% | ~0.3分鐘 | - |
| 優化模型 | LightGBM | 0.007939 | 0.79% | ~0.2分鐘 | - |
| 堆疊模型 | Super Stacking with RC | 0.132307 | 13.23% | ~1.3分鐘 | +90.71% |

## 使用方法

### 環境設置
```bash
python setup_environment.py
pip install -r requirements.txt
```

### 運行模型
```bash
python main_ensemble.py
```

### 查看結果
訓練完成後，可以在`model_results/@summary.html`中查看詳細的模型比較和分析結果。

### 啟動預測結果展示 Web App
本專案提供可部署至 Vercel 的簡易網頁介面。
在本地啟動後端 API：
```bash
uvicorn api.prediction_api:app --reload
```
然後在瀏覽器中開啟 `index.html` 查看預測結果。

若要部署至 [Vercel](https://vercel.com/)，安裝 Vercel CLI 並執行 `vercel` 即可。儲存庫已包含 `vercel.json` 設定。

## 可視化說明

1. **模型性能比較圖**
   - `first_layer_model_performance.png`: 第一層基礎模型性能比較
   - `all_model_performance.png`: 所有模型性能總覽

2. **預測分布圖**
   - `prediction_distribution.png`: 預測值分布情況
   - 包含異常值檢測和分布偏態分析

3. **分段模型性能圖**
   - `segmented_model_performance.png`: 不同價格區間的模型表現
   - 顯示每個區間的最佳模型和RMSE

## 開發規範

1. **RMSE計算規範**
   - 所有模型評估必須遵循[RMSE計算指南](docs/rmse_calculation_guide.md)
   - 確保先計算原始RMSE，再計算對數空間RMSE
   - 在報告中明確標明RMSE的計算空間

2. **代碼風格**
   - 遵循PEP 8規範
   - 使用有意義的變量名
   - 添加適當的註釋和文檔字符串

## 後續優化方向

1. **模型改進**
   - 優化殘差校準機制
   - 實施神經網絡模型
   - 優化特徵選擇策略
   - 改進分段模型的區間劃分

2. **工程化改進**
   - 添加模型部署支持
   - 實現自動化測試
   - 優化代碼結構

3. **可視化增強**
   - 添加交互式圖表
   - 實現實時監控
   - 優化展示界面

## 貢獻指南

1. Fork 本專案
2. 創建新的功能分支
3. 提交更改
4. 發起 Pull Request

## 授權
MIT License
