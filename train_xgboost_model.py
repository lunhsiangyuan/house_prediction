#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - XGBoost模型訓練主程式
House Price Prediction Competition - XGBoost Model Training Main Program
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import advanced_xgboost_model as axgb

def main():
    """主程式"""
    print("==== 房價預測XGBoost模型訓練 ====")
    
    # 檢查並創建結果目錄
    output_dir = 'house_prices_data/model_results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 載入特徵工程後的資料
        print("\n=== 1. 載入資料 ===")
        X_train, X_test, y_train = axgb.load_engineered_data()
        
        # 如果特徵工程結果不存在，從預處理資料或原始資料開始
        if X_train is None or X_test is None or y_train is None:
            print("無法載入特徵工程後的資料，程序終止")
            return
            
        # 載入測試集ID
        test_ids = axgb.load_test_ids()
        
        print(f"訓練集大小: {X_train.shape}")
        print(f"測試集大小: {X_test.shape}")
        print(f"目標變量大小: {y_train.shape}")
        
        # 準備目標變量（對數轉換）
        print("\n=== 2. 準備目標變量 ===")
        y_train_log = axgb.prepare_target(y_train, log_transform=True)
        
        # 調優XGBoost超參數
        print("\n=== 3. XGBoost超參數調優 ===")
        best_model, best_params, best_score = axgb.tune_xgboost_hyperparameters(X_train, y_train_log)
        
        # 使用最佳參數訓練XGBoost模型
        print("\n=== 4. 訓練和評估XGBoost模型 ===")
        model, mean_rmse, std_rmse = axgb.train_and_evaluate_xgboost(X_train, y_train_log, best_params)
        model_performance = (mean_rmse, std_rmse)
        
        # 分析特徵重要性
        print("\n=== 5. 特徵重要性分析 ===")
        feature_importance = axgb.analyze_feature_importance(model, X_train)
        
        # 使用XGBoost模型生成預測
        print("\n=== 6. 生成預測 ===")
        y_pred, pred_stats = axgb.predict_with_xgboost(model, X_test, log_transformed=True)
        
        # 創建提交文件
        print("\n=== 7. 創建提交文件 ===")
        submission = axgb.create_submission_file(test_ids, y_pred, filename='xgboost_submission.csv')
        
        # 可視化學習曲線
        print("\n=== 8. 可視化學習曲線 ===")
        learning_curve_results = axgb.visualize_learning_curve(model, X_train, y_train_log)
        
        # 獲取調優結果
        print("\n=== 9. 獲取調優結果 ===")
        import pickle
        tuning_results_path = f'{output_dir}/xgboost_tuning_results.pkl'
        with open(tuning_results_path, 'rb') as f:
            tuning_results = pickle.load(f)
        
        # 更新HTML報告
        print("\n=== 10. 更新HTML報告 ===")
        axgb.update_html_report(model, model_performance, feature_importance, pred_stats, 
                              tuning_results, learning_curve_results)
        
        print("\n==== 房價預測XGBoost模型訓練完成 ====")
        print(f"結果已儲存到 {output_dir} 目錄")
    
    except Exception as e:
        print(f"程序執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
