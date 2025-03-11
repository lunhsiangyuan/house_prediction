"""
房價預測探索性數據分析 - 房產專家視角 (最終中文字體版)
House Price Prediction EDA - Real Estate Expert Perspective (Final Chinese Font)
==============================================================================

這個腳本針對Kaggle房價預測競賽數據集進行探索性分析，重點關注房產專家最關心的特徵。
使用多種方法設置中文字體以確保中文標籤正確顯示。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import platform
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 中文字體設置 - 使用多種方法確保中文顯示
# 方法1: 使用plt.rc設置字體族
plt.rc('font', family='Arial Unicode MS')  # 這是Mac上通常可用的中文字體
plt.rc('axes', unicode_minus=False)  # 正確顯示負號

# 方法2: 使用FontProperties直接指定字體
chinese_font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # Mac系統字體

# 方法3: 使用rcParams設置字體
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Heiti TC', 'PingFang TC', 
                                  'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 設定繪圖風格
sns.set_style("whitegrid")
try:
    plt.style.use("seaborn-v0_8-pastel")
except:
    try:
        plt.style.use("seaborn-pastel")  # 對於較新版本的matplotlib
    except:
        print("無法設置seaborn-pastel風格，使用默認風格")

# 設定圖片保存路徑
output_dir = 'eda_results/plots_final'
os.makedirs(output_dir, exist_ok=True)

# 載入數據
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print(f"訓練集形狀: {train_data.shape}")
print(f"測試集形狀: {test_data.shape}")

# 基本情況概覽
print("\n銷售價格(SalePrice)基本統計量:")
print(train_data['SalePrice'].describe())

# 房產專家角度：重點關注的特徵
key_features = [
    # 位置因素
    'Neighborhood',  # 社區
    # 面積因素
    'GrLivArea',     # 居住面積
    'TotalBsmtSF',   # 地下室總面積
    'LotArea',       # 地塊面積
    # 品質因素
    'OverallQual',   # 整體品質
    'OverallCond',   # 整體狀況
    # 房屋類型
    'BldgType',      # 建築類型
    'HouseStyle',    # 房屋風格
    # 年份因素
    'YearBuilt',     # 建造年份
    'YearRemodAdd',  # 翻修年份
    # 重要設施
    'FullBath',      # 全浴室數量
    'BedroomAbvGr',  # 臥室數量
    'KitchenQual',   # 廚房品質
    'GarageCars',    # 車庫容量
    'GarageArea',    # 車庫面積
    # 目標變量
    'SalePrice'      # 銷售價格
]

# 檢查這些重要特徵的缺失值
missing_values = train_data[key_features].isnull().sum()
print("\n重要特徵的缺失值數量:")
print(missing_values[missing_values > 0])

# 創建一個只包含關鍵特徵的數據框
key_data = train_data[key_features].copy()

# 簡單測試圖表，檢查中文顯示是否正確
plt.figure(figsize=(8, 6))
plt.title('測試中文顯示', fontproperties=chinese_font, fontsize=20)
plt.xlabel('X軸中文標籤', fontproperties=chinese_font, fontsize=15)
plt.ylabel('Y軸中文標籤', fontproperties=chinese_font, fontsize=15)
plt.text(0.5, 0.5, '中文顯示測試', ha='center', va='center', fontproperties=chinese_font, fontsize=20)
plt.savefig(f'{output_dir}/chinese_text_test.png', dpi=300)
plt.close()

# 1. 銷售價格分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(key_data['SalePrice'], kde=True)
plt.title('房屋銷售價格分佈', fontproperties=chinese_font, fontsize=14)
plt.xlabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=12)
plt.ylabel('頻率', fontproperties=chinese_font, fontsize=12)

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(key_data['SalePrice']), kde=True)
plt.title('對數轉換後的房屋銷售價格分佈', fontproperties=chinese_font, fontsize=14)
plt.xlabel('log(銷售價格+1)', fontproperties=chinese_font, fontsize=12)
plt.ylabel('頻率', fontproperties=chinese_font, fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/price_distribution.png', dpi=300)
plt.close()

# 2. 相關性分析 - 數值特徵與價格的關係
numeric_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YearRemodAdd', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'GarageArea']

plt.figure(figsize=(10, 8))
correlation = key_data[numeric_features + ['SalePrice']].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
plt.title('數值特徵與銷售價格的相關性', fontproperties=chinese_font, fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300)
plt.close()

# 打印與銷售價格相關性排序
price_correlation = correlation['SalePrice'].sort_values(ascending=False)
print("\n與銷售價格相關性排名:")
print(price_correlation)

# 3. 分析不同社區(Neighborhood)的價格分佈
plt.figure(figsize=(14, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=key_data)
plt.title('不同社區的房價分佈', fontproperties=chinese_font, fontsize=16)
plt.xlabel('社區', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{output_dir}/neighborhood_prices.png', dpi=300)
plt.close()

# 計算每個社區的平均價格並排序
neighborhood_avg_price = key_data.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
print("\n各社區平均房價排名(由高至低):")
print(neighborhood_avg_price)

# 4. 分析整體品質(OverallQual)與價格的關係
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=key_data)
plt.title('不同整體品質等級的房價分佈', fontproperties=chinese_font, fontsize=16)
plt.xlabel('整體品質 (1-10)', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/quality_vs_price.png', dpi=300)
plt.close()

# 5. 分析居住面積(GrLivArea)與價格的關係
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('居住面積與銷售價格的關係 (按整體品質著色)', fontproperties=chinese_font, fontsize=16)
plt.xlabel('居住面積(平方呎)', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/area_vs_price.png', dpi=300)
plt.close()

# 6. 分析建造年份(YearBuilt)與價格的關係
plt.figure(figsize=(12, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('建造年份與銷售價格的關係 (按整體品質著色)', fontproperties=chinese_font, fontsize=16)
plt.xlabel('建造年份', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/year_vs_price.png', dpi=300)
plt.close()

# 7. 分析不同建築類型(BldgType)的價格分佈
plt.figure(figsize=(10, 6))
sns.boxplot(x='BldgType', y='SalePrice', data=key_data)
plt.title('不同建築類型的房價分佈', fontproperties=chinese_font, fontsize=16)
plt.xlabel('建築類型', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/building_type_prices.png', dpi=300)
plt.close()

# 8. 分析廚房品質(KitchenQual)與價格的關係
plt.figure(figsize=(10, 6))
order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']  # 從低到高排序
sns.boxplot(x='KitchenQual', y='SalePrice', data=key_data, order=order)
plt.title('不同廚房品質的房價分佈', fontproperties=chinese_font, fontsize=16)
plt.xlabel('廚房品質', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/kitchen_quality_prices.png', dpi=300)
plt.close()

# 9. 分析車庫容量(GarageCars)與價格的關係
plt.figure(figsize=(10, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=key_data)
plt.title('不同車庫容量的房價分佈', fontproperties=chinese_font, fontsize=16)
plt.xlabel('車庫容量(車位數)', fontproperties=chinese_font, fontsize=14)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/garage_capacity_prices.png', dpi=300)
plt.close()

# 10. 房屋年齡與價格的關係（創建新特徵）
key_data['HouseAge'] = train_data['YrSold'] - train_data['YearBuilt']
key_data['RemodAge'] = train_data['YrSold'] - train_data['YearRemodAdd']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='HouseAge', y='SalePrice', data=key_data)
plt.title('房屋年齡與銷售價格的關係', fontproperties=chinese_font, fontsize=14)
plt.xlabel('房屋年齡(銷售年份-建造年份)', fontproperties=chinese_font, fontsize=12)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=12)

plt.subplot(1, 2, 2)
sns.scatterplot(x='RemodAge', y='SalePrice', data=key_data)
plt.title('上次翻修至銷售的時間與銷售價格的關係', fontproperties=chinese_font, fontsize=14)
plt.xlabel('翻修年齡(銷售年份-翻修年份)', fontproperties=chinese_font, fontsize=12)
plt.ylabel('銷售價格(美元)', fontproperties=chinese_font, fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/age_vs_price.png', dpi=300)
plt.close()

# 房產專家總結
print("\n===== 房產專家觀點：影響房價的重要因素 =====")
print("\n1. 與銷售價格相關性最高的前5個數值特徵:")
for feature, corr in price_correlation.items():
    if feature != 'SalePrice':
        print(f"   - {feature}: {corr:.4f}")
    if len([f for f in price_correlation.items() if f[0] != 'SalePrice']) == 5:
        break

print("\n2. 最高價社區TOP 5:")
for neighborhood, price in neighborhood_avg_price.head(5).items():
    print(f"   - {neighborhood}: ${price:.2f}")

print("\n3. 整體品質對價格的影響:")
qual_price = key_data.groupby('OverallQual')['SalePrice'].mean()
lowest_qual = qual_price.min()
highest_qual = qual_price.max()
print(f"   - 最高品質(10分)的房屋平均價格是最低品質(1分)的{highest_qual/lowest_qual:.1f}倍")

print("\n4. 面積與價格:")
area_corr = key_data[['GrLivArea', 'SalePrice']].corr().iloc[0,1]
print(f"   - 居住面積與價格的相關性: {area_corr:.4f}")
print(f"   - 每增加1000平方呎的居住面積，價格平均增加約: ${key_data['SalePrice'].mean()/key_data['GrLivArea'].mean()*1000:.2f}")

print("\n5. 建築類型影響:")
bldg_price = key_data.groupby('BldgType')['SalePrice'].mean().sort_values(ascending=False)
for btype, price in bldg_price.items():
    print(f"   - {btype}: ${price:.2f}")

print("\n======================================")
print(f"EDA分析完成! 所有中文標題圖表已保存到{output_dir}目錄")
print("本版本使用FontProperties直接指定字體，應能正確顯示中文")
