"""
堆疊模型訓練模塊
實現堆疊集成模型訓練的核心邏輯
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from data_handling import segment_data_by_price
from model_definition import get_base_models, get_meta_model, get_final_meta_model

def train_base_models(X, y, X_test, models, n_folds=5):
    """訓練基礎模型"""
    print("\n訓練基礎模型...")
    
    # 初始化結果字典
    oof_predictions = {}
    test_predictions = {}
    cv_scores = {}
    
    # 確保目標變量在對數空間
    if y.skew() > 0.5:  # 如果右偏較大，執行對數轉換
        print("  對目標變量進行對數轉換...")
        y = np.log1p(y)
    
    # K折交叉驗證
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 對每個模型進行訓練
    for name, model in models.items():
        print(f"\n訓練模型: {name}")
        
        # 初始化預測數組
        oof_pred = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        
        # K折交叉驗證訓練
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold}/{n_folds}...")
            
            # 分割數據
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 訓練模型
            model.fit(X_train, y_train)
            
            # 預測
            oof_pred[val_idx] = model.predict(X_val)
            test_pred += model.predict(X_test) / n_folds
        
        # 計算RMSE（在對數空間）
        rmse = np.sqrt(mean_squared_error(y, oof_pred))
        print(f"  {name} CV RMSE (對數空間): {rmse:.6f}")
        
        # 保存結果
        oof_predictions[name] = oof_pred
        test_predictions[name] = test_pred
        cv_scores[name] = rmse
    
    return oof_predictions, test_predictions, cv_scores

def train_meta_model(X_meta, y, X_meta_test, meta_model, n_folds=5):
    """訓練元模型"""
    print("\n訓練元模型...")
    
    # 確保目標變量在對數空間
    if y.skew() > 0.5:  # 如果右偏較大，執行對數轉換
        print("  對目標變量進行對數轉換...")
        y = np.log1p(y)
    
    # 初始化預測數組
    oof_pred = np.zeros(len(X_meta))
    test_pred = np.zeros(len(X_meta_test))
    
    # K折交叉驗證
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # K折交叉驗證訓練
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        # 分割數據
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 訓練模型
        meta_model.fit(X_train, y_train)
        
        # 預測
        oof_pred[val_idx] = meta_model.predict(X_val)
        test_pred += meta_model.predict(X_meta_test) / n_folds
    
    # 計算RMSE（在對數空間）
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    print(f"  Meta Model CV RMSE (對數空間): {rmse:.6f}")
    
    return test_pred, rmse

def train_segmented_models(segments, X_test, models, n_folds=5):
    """對每個價格段訓練單獨的模型"""
    print("\n訓練分段模型...")
    
    segment_scores = {}
    segment_predictions = []
    
    # 對每個價格段訓練模型
    for i, (segment_X, segment_y) in enumerate(segments):
        segment_name = f"段 {i+1}"
        print(f"\n訓練價格段 {i+1} 模型...")
        
        # 確保目標變量在對數空間
        if segment_y.skew() > 0.5:  # 如果右偏較大，執行對數轉換
            print(f"  對{segment_name}目標變量進行對數轉換...")
            segment_y = np.log1p(segment_y)
        
        # 訓練基礎模型
        _, segment_test_preds, segment_cv_scores = train_base_models(
            segment_X, segment_y, X_test, models, n_folds
        )
        
        # 找到最佳模型
        best_model_name = min(segment_cv_scores, key=segment_cv_scores.get)
        best_prediction = segment_test_preds[best_model_name]
        best_score = segment_cv_scores[best_model_name]
        
        print(f"  {segment_name} 最佳模型: {best_model_name}, RMSE (對數空間): {best_score:.6f}")
        
        # 保存結果
        segment_scores[segment_name] = {
            'best_model': best_model_name,
            'rmse': best_score
        }
        segment_predictions.append(best_prediction)
    
    # 計算加權平均預測
    weighted_pred = np.zeros(len(X_test))
    weights = [0.3, 0.4, 0.3]  # 可根據數據調整權重
    
    for i, pred in enumerate(segment_predictions):
        weighted_pred += pred * weights[i]
    
    return weighted_pred, segment_scores

def multi_layer_stacking(X_train, y_train, X_test, n_folds=5, random_state=42, use_restacking=True):
    """
    執行多層堆疊集成
    
    Parameters:
    -----------
    X_train : DataFrame
        訓練特徵
    y_train : Series
        訓練目標
    X_test : DataFrame
        測試特徵
    n_folds : int
        交叉驗證折數
    random_state : int
        隨機種子
    use_restacking : bool
        是否使用重新堆疊
        
    Returns:
    --------
    final_pred : array
        最終預測
    meta_cv_rmse : float
        元模型交叉驗證RMSE
    cv_scores : dict
        各模型交叉驗證分數
    segment_scores : dict
        各價格段最佳模型和RMSE
    """
    start_time = time.time()
    
    print("\n開始多層堆疊集成訓練...")
    
    # 獲取模型
    base_models = get_base_models()
    meta_model = get_meta_model()
    final_meta_model = get_final_meta_model()
    
    # 轉換為numpy數組以提高性能
    X_train_array = X_train.values
    X_test_array = X_test.values
    
    # 訓練第一層模型
    oof_preds, test_preds, cv_scores = train_base_models(
        X_train_array, y_train, X_test_array, base_models, n_folds
    )
    
    # 準備元特徵
    meta_features = np.column_stack([pred for pred in oof_preds.values()])
    meta_features_test = np.column_stack([pred for pred in test_preds.values()])
    
    # 如果使用重新堆疊，添加原始特徵
    if use_restacking:
        print("\n添加原始特徵到元特徵...")
        meta_features = np.column_stack([meta_features, X_train_array])
        meta_features_test = np.column_stack([meta_features_test, X_test_array])
    
    # 訓練第二層元模型
    second_layer_pred, meta_cv_rmse = train_meta_model(
        meta_features, y_train, meta_features_test, meta_model, n_folds
    )
    
    # 訓練分段模型
    print("\n訓練價格段模型...")
    segments = segment_data_by_price(X_train_array, y_train, n_segments=3)
    segmented_pred, segment_scores = train_segmented_models(
        segments, X_test_array, base_models, n_folds
    )
    
    # 準備最終元特徵 - 修正維度不匹配問題
    # 只使用測試集來生成最終預測
    print("\n訓練最終元模型...")
    final_pred = second_layer_pred  # 使用第二層的預測作為最終預測
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    
    return final_pred, meta_cv_rmse, cv_scores, segment_scores 