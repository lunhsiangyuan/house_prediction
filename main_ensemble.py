#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超級高級多層堆疊集成模型的主程序。
整合了高級非線性特徵、多種模型、分段預測等策略。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 從模塊導入函數
from data_handling import load_data
from visualization import plot_model_performance, plot_prediction_distribution, plot_segmented_model_performance
from ensemble_training import multi_layer_stacking
from utils import generate_submission, update_html_summary

def main():
    # 定義輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_results/super_ensemble_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("開始訓練超級多層堆疊集成模型...")
    start_time = time.time()
    
    # 加載數據
    X_train, X_test, y_train, feature_names = load_data()
    
    # 訓練多層堆疊集成模型
    final_pred, meta_rmse, cv_scores, segment_scores = multi_layer_stacking(
        X_train, y_train, X_test, n_folds=5, random_state=42
    )
    
    # 計算訓練時間
    training_time = (time.time() - start_time) / 60
    print(f"\n總訓練時間: {training_time:.2f}分鐘")
    print(f"元模型 CV RMSE: {meta_rmse:.6f}")
    
    # 繪製模型性能圖
    print("\n繪製模型性能比較圖...")
    plot_model_performance(cv_scores, output_dir)
    
    # 繪製分段模型性能圖
    if segment_scores:
        print("\n繪製分段模型性能圖...")
        plot_segmented_model_performance(segment_scores, output_dir)
    
    # 生成提交文件
    print("\n生成提交文件...")
    submission_path = generate_submission(final_pred, output_dir)
    
    # 繪製預測分佈圖
    print("\n繪製預測分佈圖...")
    plot_prediction_distribution(final_pred, output_dir)
    
    # 更新HTML摘要
    print("\n更新HTML摘要...")
    results_summary = {
        'rmse': meta_rmse,
        'training_time': training_time,
        'model_name': 'Super Advanced Multi-layer Stacking Ensemble',
        'features_count': X_train.shape[1],
        'timestamp': timestamp,
        'best_models': {
            segment: info['best_model'] 
            for segment, info in segment_scores.items()
        } if segment_scores else {}
    }
    update_html_summary(results_summary, output_dir)
    
    print("\n模型訓練和評估完成！")
    print(f"結果已保存到: {output_dir}")
    
    # 返回結果以供後續分析使用
    return {
        'final_pred': final_pred,
        'meta_rmse': meta_rmse,
        'cv_scores': cv_scores,
        'segment_scores': segment_scores,
        'output_dir': output_dir,
        'training_time': training_time
    }

if __name__ == "__main__":
    main() 