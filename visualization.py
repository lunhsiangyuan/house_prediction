"""
視覺化模塊
用於繪製模型性能和預測分佈圖表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.font_manager as fm
import platform

# 設置中文字體
def setup_chinese_font():
    """設置中文字體"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/mingliu.ttc'
    else:  # Linux
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    
    if Path(font_path).exists():
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        print("警告：找不到中文字體文件，將使用系統默認字體")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

def plot_model_performance(cv_scores, output_dir):
    """
    繪製模型性能比較圖
    
    Parameters:
    -----------
    cv_scores : dict
        包含每個模型的cross-validation RMSE
    output_dir : str
        輸出目錄路徑
    """
    setup_chinese_font()
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 用matplotlib自帶的樣式代替seaborn
    plt.style.use('ggplot')  # 改用ggplot樣式
    
    # 收集一層模型和二層模型的分數
    first_layer_scores = {k: v for k, v in cv_scores.items() if k in ['ridge', 'lasso', 'elasticnet', 'rf', 'gbm', 'xgb', 'lgb', 'svr', 'knn']}
    second_layer_scores = {k: v for k, v in cv_scores.items() if k.startswith('meta')}
    
    # 為不同類型的模型創建顏色映射
    model_colors = {
        'ridge': 'blue',
        'lasso': 'blue',
        'elasticnet': 'blue',
        'rf': 'green',
        'gbm': 'green',
        'xgb': 'green',
        'lgb': 'green',
        'svr': 'orange',
        'knn': 'orange',
        'meta_ridge': 'red',
        'meta_lasso': 'red',
        'meta_elasticnet': 'red',
        'meta_rf': 'purple',
        'meta_xgb': 'purple',
        'meta_lgb': 'purple'
    }
    
    # 繪製一層模型的性能
    plt.figure(figsize=(14, 8))
    models = list(first_layer_scores.keys())
    scores = list(first_layer_scores.values())
    colors = [model_colors.get(model, 'gray') for model in models]
    
    # 按照RMSE排序
    sorted_indices = np.argsort(scores)
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    bars = plt.bar(models, scores, color=colors)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('第一層模型性能比較', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 在每個柱子上顯示具體RMSE值
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{score:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'first_layer_model_performance.png', dpi=300)
    
    # 如果有第二層模型，繪製它們的性能
    if second_layer_scores:
        plt.figure(figsize=(14, 8))
        models = list(second_layer_scores.keys())
        scores = list(second_layer_scores.values())
        colors = [model_colors.get(model, 'gray') for model in models]
        
        # 按照RMSE排序
        sorted_indices = np.argsort(scores)
        models = [models[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        bars = plt.bar(models, scores, color=colors)
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('第二層模型性能比較', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 在每個柱子上顯示具體RMSE值
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{score:.6f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'second_layer_model_performance.png', dpi=300)
    
    # 繪製所有模型的性能
    plt.figure(figsize=(16, 8))
    all_scores = {**first_layer_scores, **second_layer_scores}
    models = list(all_scores.keys())
    scores = list(all_scores.values())
    colors = [model_colors.get(model, 'gray') for model in models]
    
    # 按照RMSE排序
    sorted_indices = np.argsort(scores)
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    bars = plt.bar(models, scores, color=colors)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('所有模型性能比較', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 在每個柱子上顯示具體RMSE值
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'all_model_performance.png', dpi=300)
    
    plt.close('all')

def plot_prediction_distribution(predictions, output_dir, filename='prediction_distribution.png'):
    """
    繪製預測分佈圖
    
    Parameters:
    -----------
    predictions : array-like
        預測值數組
    output_dir : str
        輸出目錄路徑
    filename : str
        輸出文件名
    """
    setup_chinese_font()
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 用matplotlib自帶的樣式代替seaborn
    plt.style.use('ggplot')  # 改用ggplot樣式
    
    plt.figure(figsize=(12, 7))
    plt.hist(predictions, bins=50, alpha=0.7, color='steelblue')
    plt.axvline(np.mean(predictions), color='red', linestyle='dashed', linewidth=2, 
                label=f'平均值: {np.mean(predictions):.2f}')
    plt.axvline(np.median(predictions), color='green', linestyle='dashed', linewidth=2, 
                label=f'中位數: {np.median(predictions):.2f}')
    
    plt.title('預測值分佈', fontsize=14)
    plt.xlabel('預測銷售價格 (log scale)', fontsize=12)
    plt.ylabel('頻率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()

def plot_segmented_model_performance(segment_scores, output_dir):
    """
    繪製分段模型性能比較圖
    
    Parameters:
    -----------
    segment_scores : dict
        包含每個價格段的最佳模型和RMSE
    output_dir : str
        輸出目錄路徑
    """
    setup_chinese_font()
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 用matplotlib自帶的樣式代替seaborn
    plt.style.use('ggplot')  # 改用ggplot樣式
    
    plt.figure(figsize=(10, 6))
    
    segments = list(segment_scores.keys())
    best_models = [info['best_model'] for info in segment_scores.values()]
    rmse_values = [info['rmse'] for info in segment_scores.values()]
    
    # 為不同類型的模型創建顏色映射
    model_type_colors = {
        'ridge': 'blue',
        'lasso': 'blue',
        'elasticnet': 'blue',
        'rf': 'green',
        'gbm': 'green',
        'xgb': 'green',
        'lgb': 'green',
        'svr': 'orange',
        'knn': 'orange',
    }
    
    bars = plt.bar(segments, rmse_values, color=[model_type_colors.get(model, 'gray') for model in best_models])
    
    # 在每個柱子上標註模型名稱和RMSE值
    for i, (bar, model, rmse) in enumerate(zip(bars, best_models, rmse_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5, 
                 model, ha='center', va='center', color='white', fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'RMSE: {rmse:.6f}', ha='center', va='bottom')
    
    plt.title('不同價格段的最佳模型性能', fontsize=14)
    plt.xlabel('價格段', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'segmented_model_performance.png', dpi=300)
    plt.close() 