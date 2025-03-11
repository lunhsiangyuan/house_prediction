#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房價預測競賽 - 優化執行腳本
House Price Prediction Competition - Optimization Runner
==============================================================================

此腳本用於執行一系列優化方案，包括高級特徵工程、貝葉斯優化XGBoost和堆疊集成模型。
"""

import os
import time
import subprocess
import argparse

def run_command(command, description):
    """執行命令並顯示進度"""
    print(f"\n{'='*80}")
    print(f"執行: {description}")
    print(f"命令: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # 實時輸出命令執行結果
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"完成: {description}")
    print(f"耗時: {end_time - start_time:.2f} 秒")
    print(f"狀態: {'成功' if process.returncode == 0 else '失敗'}")
    print(f"{'='*80}\n")
    
    return process.returncode

def create_directories():
    """創建必要的目錄"""
    directories = [
        'model_results',
        'feature_engineering_results',
        'feature_engineering_results/advanced'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"創建目錄: {directory}")

def main():
    """主函數：執行所有優化方案"""
    parser = argparse.ArgumentParser(description='房價預測競賽優化執行腳本')
    parser.add_argument('--skip-feature-engineering', action='store_true', help='跳過高級特徵工程步驟')
    parser.add_argument('--skip-bayesian', action='store_true', help='跳過貝葉斯優化XGBoost步驟')
    parser.add_argument('--skip-stacking', action='store_true', help='跳過堆疊集成模型步驟')
    args = parser.parse_args()
    
    # 創建必要的目錄
    create_directories()
    
    # 執行高級特徵工程
    if not args.skip_feature_engineering:
        run_command('python advanced_feature_engineering.py', '高級特徵工程')
    else:
        print("跳過高級特徵工程步驟")
    
    # 執行貝葉斯優化XGBoost
    if not args.skip_bayesian:
        run_command('python bayesian_xgboost_model.py', '貝葉斯優化XGBoost')
    else:
        print("跳過貝葉斯優化XGBoost步驟")
    
    # 執行堆疊集成模型
    if not args.skip_stacking:
        run_command('python stacking_ensemble_model.py', '堆疊集成模型')
    else:
        print("跳過堆疊集成模型步驟")
    
    # 創建優化總結
    run_command('python create_optimization_summary.py', '創建優化總結')
    
    print("\n所有優化方案執行完成！")
    print("請查看 model_results/@summary.html 獲取詳細的模型比較和分析結果。")

if __name__ == "__main__":
    main() 