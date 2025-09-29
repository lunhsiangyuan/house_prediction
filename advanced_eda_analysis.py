#!/usr/bin/env python3
"""
Advanced EDA Analysis with Multiple Tools
使用最佳 EDA 工具全方位分析 Kaggle 學生數據

作者: AI Assistant
日期: 2024年12月
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體和樣式
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_and_prepare_data(file_path):
    """
    載入並準備數據用於分析
    """
    print("=" * 60)
    print("📊 載入數據進行 EDA 分析")
    print("=" * 60)
    
    # 載入數據
    df = pd.read_csv(file_path, encoding='latin1')
    
    # 基本數據資訊
    print(f"數據維度: {df.shape}")
    print(f"列名稱: {list(df.columns)}")
    print("\n數據預覽:")
    print(df.head())
    
    # 數據類型檢查
    print("\n數據類型:")
    print(df.dtypes)
    
    # 缺失值檢查
    print("\n缺失值統計:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    return df


def comprehensive_pandas_analysis(df):
    """
    使用 Pandas + Matplotlib + Seaborn 進行全方位分析
    """
    print("\n" + "=" * 60)
    print("🔍 Pandas + Matplotlib + Seaborn 全方位分析")
    print("=" * 60)
    
    analysis_results = {}
    
    # 1. 基本統計資訊
    print("\n📈 數值欄位基本統計:")
    numeric_cols = ['Age', 'entryEXAM', 'studyHOURS', 'Python', 'DB']
    numeric_df = df[numeric_cols].dropna()
    stats = numeric_df.describe()
    print(stats)
    analysis_results['basic_stats'] = stats
    
    # 2. 數據分佈分析
    print("\n📊 數據分佈分析:")
    
    # 年齡分佈
    age_dist = df['Age'].describe()
    print(f"年齡統計: 平均 {age_dist['mean']:.1f}歲, 範圍 {age_dist['min']}-{age_dist['max']}歲")
    
    # 性別分佈
    print("\n性別分佈:")
    gender_dist = df['gender'].value_counts()
    print(gender_dist)
    
    # 國家分佈  
    print("\n前5名國家:")
    country_dist = df['country'].value_counts().head()
    print(country_dist)
    
    analysis_results['distributions'] = {
        'gender': gender_dist,
        'country': country_dist,
        'age_stats': age_dist
    }
    
    # 3. 成績分析
    print("\n🎯 成績相關分析:")
    
    # 學科成績統計
    scores_stats = df[['Python', 'DB']].describe()
    print("學科成績統計:")
    print(scores_stats)
    
    # 成績相關性
    scores_corr = df[['Python', 'DB']].corr()
    print(f"\n兩科成績相關性: {scores_corr.iloc[0,1]:.3f}")
    
    # 學習時間與成績關係
    study_corr_python = df[['studyHOURS', 'Python']].corr().iloc[0,1]
    study_corr_db = df[['studyHOURS', 'DB']].corr().iloc[0,1]
    print(f"學習時間與Python成績相關性: {study_corr_python:.3f}")
    print(f"學習時間與DB成績相關性: {study_corr_db:.3f}")
    
    analysis_results['performance'] = {
        'scores_stats': scores_stats,
        'scores_correlation': scores_corr.iloc[0,1],
        'study_python_corr': study_corr_python,
        'study_db_corr': study_corr_db
    }
    
    return analysis_results


def create_visualizations(df):
    """
    創建多種視覺化圖表
    """
    print("\n" + "=" * 60)
    print("📊 創建視覺化圖表")
    print("=" * 60)
    
    # 設置圖表大小
    plt.figure(figsize=(20, 15))
    
    # 1. 數據分佈圖
    plt.subplot(2, 4, 1)
    df['Age'].hist(bins=10, alpha=0.7, color='skyblue')
    plt.title('年齡分佈', fontsize=12)
    plt.xlabel('年齡')
    plt.ylabel('人數')
    
    plt.subplot(2, 4, 2)
    df['Python'].hist(bins=10, alpha=0.7, color='lightgreen')
    plt.title('Python 成績分佈', fontsize=12)
    plt.xlabel('分數')
    plt.ylabel('人數')
    
    plt.subplot(2, 4, 3)
    df['studyHOURS'].hist(bins=10, alpha=0.7, color='salmon')
    plt.title('學習時數分佈', fontsize=12)
    plt.xlabel('時數')
    plt.ylabel('人數')
    
    # 2. 性別分佈圓餅圖
    plt.subplot(2, 4, 4)
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('性別分佈', fontsize=12)
    
    # 3. 國家分佈 (前10名)
    plt.subplot(2, 4, 5)
    top_countries = df['country'].value_counts().head(10)
    plt.bar(range(len(top_countries)), top_countries.values)
    plt.xticks(range(len(top_countries)), top_countries.index, rotation=45)
    plt.title('國家學生數 (前10)', fontsize=12)
    plt.ylabel('人數')
    
    # 4. 散點圖: 學習時間 vs Python成績
    plt.subplot(2, 4, 6)
    plt.scatter(df['studyHOURS'], df['Python'], alpha=0.6)
    plt.xlabel('學習時數')
    plt.ylabel('Python 成績')
    plt.title('學習時間 vs Python成績', fontsize=12)
    
    # 5. 散點圖: Python vs DB 成績
    plt.subplot(2, 4, 7)
    plt.scatter(df['Python'], df['DB'], alpha=0.6, color='purple')
    plt.xlabel('Python 成績')
    plt.ylabel('DB 成績')
    plt.title('Python vs DB 成績關係', fontsize=12)
    
    # 6. 箱線圖: 不同住宿的成績比較
    plt.subplot(2, 4, 8)
    df.boxplot(column='Python', by='residence', ax=plt.gca())
    plt.title('不同住宿類型的Python成績', fontsize=12)
    plt.suptitle('')  # 移除自動標題
    
    plt.tight_layout()
    
    # 儲存圖表
    output_file = Path('/Users/lunhsiangyuan/Desktop/ai side projects/kaggle/doc/comprehensive_eda_visualizations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 視覺化圖表已儲存至: {output_file}")
    
    plt.show()


def advanced_correlation_analysis(df):
    """
    高級相關性分析
    """
    print("\n" + "=" * 60)
    print("🔗 高級相關性分析")
    print("=" * 60)
    
    # 選擇數值欄位進行相關性分析
    numeric_columns = ['Age', 'entryEXAM', 'studyHOURS', 'Python', 'DB']
    correlation_matrix = df[numeric_columns].corr()
    
    print("相關性矩陣:")
    print(correlation_matrix.round(3))
    
    # 創建相關性熱圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'shrink': 0.8})
    plt.title('數值欄位相關性熱圖', fontsize=16)
    plt.tight_layout()
    
    # 儲存相關性圖表
    corr_file = Path('/Users/lunhsiangyuan/Desktop/ai side projects/kaggle/doc/correlation_heatmap.png')
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    print(f"✅ 相關性熱圖已儲存至: {corr_file}")
    
    plt.show()
    
    return correlation_matrix


def generate_ydata_profiling_report(df):
    """
    使用 YData Profiling 生成專業報告
    """
    print("\n" + "=" * 60)
    print("📋 YData Profiling 專業報告生成")
    print("=" * 60)
    
    try:
        from ydata_profiling import ProfileReport
        
        # 創建 ProfileReport
        profile = ProfileReport(
            df, 
            title="Kaggle 學生數據專業分析報告",
            explorative=True,
            dark_mode=True,
            sample={'head': 5, 'tail': 5}
        )
        
        # 儲存 HTML 報告
        html_file = '/Users/lunhsiangyuan/Desktop/ai side projects/kaggle/doc/ydata_profiling_report.html'
        profile.to_file(html_file)
        
        print(f"✅ YData Profiling 報告已生成: {html_file}")
        
        # 顯示報告摘要
        print("\n📊 YData Profiling 摘要:")
        print("- 數據概覽和品質評估")
        print("- 單變量分佈分析") 
        print("- 交互式視覺化圖表")
        print("- 缺失值模式和異常值檢測")
        print("- 數值對數值關聯分析")
        
        return profile
        
    except ImportError:
        print("❌ YData Profiling 未安裝，將跳過此分析")
        print("安裝命令: pip install ydata-profiling")
        return None
    except Exception as e:
        print(f"❌ YData Profiling 生成失敗: {str(e)}")
        return None


def generate_sweetviz_report(df):
    """
    使用 Sweetviz 生成美觀報告
    """
    print("\n" + "=" * 60)
    print("🍭 Sweetviz 美觀報告生成")
    print("=" * 60)
    
    try:
        import sweetviz as sv
        
        # 創建 Sweetviz 報告
        report = sv.analyze(
            df,
            target_feat="Python",  # 設定目標變量為Python成績
            feat_cfg=None
        )
        
        # 儲存 HTML 報告
        html_file = '/Users/lunhsiangyuan/Desktop/ai side projects/kaggle/doc/sweetviz_report.html'
        report.show_html(filepath=html_file, open_browser=False)
        
        print(f"✅ Sweetviz 報告已生成: {html_file}")
        
        # 顯示報告內容
        print("\n🍭 Sweetviz 特色功能:")
        print("- 美觀的網頁界面")
        print("- 目標特徵分析 (Python成績)")
        print("- 特徵關聯和分佈視覺化")
        print("- 數據品質和模式檢測")
        
        return report
        
    except ImportError:
        print("❌ Sweetviz 未安裝，將跳過此分析")
        print("安裝命令: pip install sweetviz")
        return None
    except Exception as e:
        print(f"❌ Sweetviz 生成失敗: {str(e)}")
        return None


def generate_dtale_analysis(df):
    """
    啟動 D-Tale 互動式分析
    """
    print("\n" + "=" * 60)
    print("🌐 D-Tale 互動式分析")
    print("=" * 60)
    
    try:
        import dtale
        
        # 啟動 D-Tale 服務
        d = dtale.show(df, subprocess=False)
        
        print(f"✅ D-Tale 已啟動!")
        print(f"📍 瀏覽器地址: {d._url}")
        print("\n🔧 D-Tale 功能特色:")
        print("- 互動式數據表格瀏覽")
        print("- 即時資料視覺化")
        print("- 統計分析和過濾功能")
        print("- 資料匯出和分享")
        print("- Web 介面操作")
        
        # 保持服務運行一段時間
        print("\n⏳ D-Tale 服務將保持運行 30 秒...")
        import time
        time.sleep(30)
        
        return d
        
    except ImportError:
        print("❌ D-Tale 未安裝，將跳過此分析")
        print("安裝命令: pip install dtale")
        return None
    except Exception as e:
        print(f"❌ D-Tale 啟動失敗: {str(e)}")
        return None


def comprehensive_summary_report(df, pandas_results, correlation_matrix):
    """
    生成綜合摘要報告
    """
    print("\n" + "=" * 60)
    print("📊 綜合 EDA 分析摘要")
    print("=" * 60)
    
    # 資料集基本資訊
    print(f"📈 資料集概覽:")
    print(f"  - 總記錄數: {len(df)}")
    print(f"  - 欄位數: {len(df.columns)}")
    print(f"  - 記憶體使用: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # 資料品質評估
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (len(df) * len(df.columns))) * 100
    print(f"\n🔍 資料品質評估:")
    print(f"  - 缺失值總數: {missing_count}")
    print(f"  - 缺失值比例: {missing_pct:.2f}%")
    print(f"  - 資料完整性: {'優秀' if missing_pct < 5 else '良好' if missing_pct < 10 else '需改善'}")
    
    # 重要發現
    print(f"\n🎯 重要發現:")
    print(f"  - 學生年齡範圍: {df['Age'].min()}-{df['Age'].max()}歲")
    print(f"  - 性別分佈: {df['gender'].value_counts().to_dict()}")
    print(f"  - 最高Python成績: {df['Python'].max()}")
    print(f"  - 學習時間與成績相關性: {pandas_results['performance']['study_python_corr']:.3f}")
    
    # 推薦的後續分析
    print(f"\n🚀 推薦的後續分析:")
    print("  1. 數據清理和預處理")
    print("  2. 機器學習模型建立 (迴歸/分類)")
    print("  3. 學生成績預測分析")
    print("  4. 學習效率因子分析")
    print("  5. 人口統計學特徵分析")


def main():
    """
    主要執行函數
    """
    print("🎯 Kaggle 學生數據 - 全方位 EDA 分析")
    print("使用多種最佳工具進行探索性數據分析")
    print("\n" + "=" * 80)
    
    # 設定數據檔案路徑
    data_file = Path('/Users/lunhsiangyuan/Desktop/ai side projects/kaggle/intro-to-data-cleaning-eda-and-machine-learning/bi.csv')
    
    # 1. 載入數據
    df = load_and_prepare_data(data_file)
    
    # 2. Pandas 綜合分析
    pandas_results = comprehensive_pandas_analysis(df)
    
    # 3. 視覺化分析
    create_visualizations(df)
    
    # 4. 相關性分析
    correlation_matrix = advanced_correlation_analysis(df)
    
    # 5. YData Profiling 報告
    ydata_profile = generate_ydata_profiling_report(df)
    
    # 6. Sweetviz 報告
    sweetviz_report = generate_sweetviz_report(df)
    
    # 7. D-Tale 互動式分析
    dtale_instance = generate_dtale_analysis(df)
    
    # 8. 綜合摘要報告
    comprehensive_summary_report(df, pandas_results, correlation_matrix)
    
    print("\n🎉 全方位 EDA 分析完成!")
    print("\n📁 生成的檔案:")
    print("  - comprehensive_eda_visualizations.png")
    print("  - correlation_heatmap.png")
    print("  - ydata_profiling_report.html")
    print("  - sweetviz_report.html")
    print("  - D-Tale 互動式分析 (需要在瀏覽器開啟)")


if __name__ == "__main__":
    main()
