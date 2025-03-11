"""
房價預測探索性數據分析 - 房產專家視角
House Price Prediction EDA - Real Estate Expert Perspective
==========================================================

這個腳本針對Kaggle房價預測競賽數據集進行探索性分析，重點關注房產專家最關心的特徵。
This script performs exploratory analysis on the Kaggle House Prices prediction competition dataset,
focusing on features that real estate experts care most about.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 設定繪圖風格 (Setting plot style)
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-pastel")

# 載入數據
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print(f"訓練集形狀 (Training set shape): {train_data.shape}")
print(f"測試集形狀 (Test set shape): {test_data.shape}")

# 基本情況概覽 (Basic overview)
print("\n銷售價格(SalePrice)基本統計量 (Basic statistics of Sale Price):")
print(train_data['SalePrice'].describe())

# 房產專家角度：重點關注的特徵 (Key features from a real estate expert's perspective)
key_features = [
    # 位置因素 (Location factors)
    'Neighborhood',  # 社區 (Neighborhood)
    # 面積因素 (Area factors)
    'GrLivArea',     # 居住面積 (Above grade living area)
    'TotalBsmtSF',   # 地下室總面積 (Total basement area)
    'LotArea',       # 地塊面積 (Lot size)
    # 品質因素 (Quality factors)
    'OverallQual',   # 整體品質 (Overall quality)
    'OverallCond',   # 整體狀況 (Overall condition)
    # 房屋類型 (House type)
    'BldgType',      # 建築類型 (Building type)
    'HouseStyle',    # 房屋風格 (House style)
    # 年份因素 (Year factors)
    'YearBuilt',     # 建造年份 (Year built)
    'YearRemodAdd',  # 翻修年份 (Year remodeled)
    # 重要設施 (Important facilities)
    'FullBath',      # 全浴室數量 (Full bathrooms)
    'BedroomAbvGr',  # 臥室數量 (Bedrooms)
    'KitchenQual',   # 廚房品質 (Kitchen quality)
    'GarageCars',    # 車庫容量 (Garage capacity)
    'GarageArea',    # 車庫面積 (Garage area)
    # 目標變量 (Target variable)
    'SalePrice'      # 銷售價格 (Sale price)
]

# 檢查這些重要特徵的缺失值 (Check missing values for key features)
missing_values = train_data[key_features].isnull().sum()
print("\n重要特徵的缺失值數量 (Missing values count for key features):")
print(missing_values[missing_values > 0])

# 創建一個只包含關鍵特徵的數據框 (Create a dataframe with only key features)
key_data = train_data[key_features].copy()

# 1. 銷售價格分布 (Sale price distribution)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(key_data['SalePrice'], kde=True)
plt.title('房屋銷售價格分佈\nHouse Sale Price Distribution')
plt.xlabel('銷售價格(美元)\nSale Price (USD)')
plt.ylabel('頻率\nFrequency')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(key_data['SalePrice']), kde=True)
plt.title('對數轉換後的房屋銷售價格分佈\nLog-transformed House Sale Price Distribution')
plt.xlabel('log(銷售價格+1)\nlog(Sale Price+1)')
plt.ylabel('頻率\nFrequency')

plt.tight_layout()
plt.savefig('plots/price_distribution.png')
plt.close()

# 2. 相關性分析 - 數值特徵與價格的關係 (Correlation analysis - relationship between numeric features and price)
numeric_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YearRemodAdd', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'GarageArea']

plt.figure(figsize=(10, 8))
correlation = key_data[numeric_features + ['SalePrice']].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
plt.title('數值特徵與銷售價格的相關性\nCorrelation Between Numeric Features and Sale Price')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 打印與銷售價格相關性排序 (Print correlation ranking with sale price)
price_correlation = correlation['SalePrice'].sort_values(ascending=False)
print("\n與銷售價格相關性排名 (Correlation ranking with sale price):")
print(price_correlation)

# 3. 分析不同社區(Neighborhood)的價格分佈 (Analyze price distribution across different neighborhoods)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=key_data)
plt.title('不同社區的房價分佈\nHouse Price Distribution Across Neighborhoods')
plt.xlabel('社區\nNeighborhood')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/neighborhood_prices.png')
plt.close()

# 計算每個社區的平均價格並排序 (Calculate and sort average price by neighborhood)
neighborhood_avg_price = key_data.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
print("\n各社區平均房價排名(由高至低) (Neighborhood average price ranking from high to low):")
print(neighborhood_avg_price)

# 4. 分析整體品質(OverallQual)與價格的關係 (Analyze relationship between overall quality and price)
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=key_data)
plt.title('不同整體品質等級的房價分佈\nHouse Price Distribution by Overall Quality')
plt.xlabel('整體品質 (1-10)\nOverall Quality (1-10)')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/quality_vs_price.png')
plt.close()

# 5. 分析居住面積(GrLivArea)與價格的關係 (Analyze relationship between living area and price)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('居住面積與銷售價格的關係 (按整體品質著色)\nLiving Area vs. Sale Price (colored by Overall Quality)')
plt.xlabel('居住面積(平方呎)\nLiving Area (square feet)')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/area_vs_price.png')
plt.close()

# 6. 分析建造年份(YearBuilt)與價格的關係 (Analyze relationship between year built and price)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('建造年份與銷售價格的關係 (按整體品質著色)\nYear Built vs. Sale Price (colored by Overall Quality)')
plt.xlabel('建造年份\nYear Built')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/year_vs_price.png')
plt.close()

# 7. 分析不同建築類型(BldgType)的價格分佈 (Analyze price distribution across different building types)
plt.figure(figsize=(10, 6))
sns.boxplot(x='BldgType', y='SalePrice', data=key_data)
plt.title('不同建築類型的房價分佈\nHouse Price Distribution by Building Type')
plt.xlabel('建築類型\nBuilding Type')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/building_type_prices.png')
plt.close()

# 8. 分析廚房品質(KitchenQual)與價格的關係 (Analyze relationship between kitchen quality and price)
plt.figure(figsize=(10, 6))
order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']  # 從低到高排序 (Sort from low to high)
sns.boxplot(x='KitchenQual', y='SalePrice', data=key_data, order=order)
plt.title('不同廚房品質的房價分佈\nHouse Price Distribution by Kitchen Quality')
plt.xlabel('廚房品質\nKitchen Quality')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/kitchen_quality_prices.png')
plt.close()

# 9. 分析車庫容量(GarageCars)與價格的關係 (Analyze relationship between garage capacity and price)
plt.figure(figsize=(10, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=key_data)
plt.title('不同車庫容量的房價分佈\nHouse Price Distribution by Garage Capacity')
plt.xlabel('車庫容量(車位數)\nGarage Capacity (car count)')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')
plt.tight_layout()
plt.savefig('plots/garage_capacity_prices.png')
plt.close()

# 10. 房屋年齡與價格的關係（創建新特徵）(Relationship between house age and price - creating new features)
key_data['HouseAge'] = train_data['YrSold'] - train_data['YearBuilt']
key_data['RemodAge'] = train_data['YrSold'] - train_data['YearRemodAdd']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='HouseAge', y='SalePrice', data=key_data)
plt.title('房屋年齡與銷售價格的關係\nHouse Age vs. Sale Price')
plt.xlabel('房屋年齡(銷售年份-建造年份)\nHouse Age (Year Sold - Year Built)')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='RemodAge', y='SalePrice', data=key_data)
plt.title('上次翻修至銷售的時間與銷售價格的關係\nTime Since Remodeling vs. Sale Price')
plt.xlabel('翻修年齡(銷售年份-翻修年份)\nRemodel Age (Year Sold - Year Remodeled)')
plt.ylabel('銷售價格(美元)\nSale Price (USD)')

plt.tight_layout()
plt.savefig('plots/age_vs_price.png')
plt.close()

# 房產專家總結 (Real Estate Expert Summary)
print("\n===== 房產專家觀點：影響房價的重要因素 =====")
print("===== Real Estate Expert View: Key Factors Affecting House Prices =====")
print("\n1. 與銷售價格相關性最高的前5個數值特徵:")
print("1. Top 5 numerical features with highest correlation to sale price:")
for feature, corr in price_correlation.items():
    if feature != 'SalePrice':
        print(f"   - {feature}: {corr:.4f}")
    if len([f for f in price_correlation.items() if f[0] != 'SalePrice']) == 5:
        break

print("\n2. 最高價社區TOP 5:")
print("2. TOP 5 highest-priced neighborhoods:")
for neighborhood, price in neighborhood_avg_price.head(5).items():
    print(f"   - {neighborhood}: ${price:.2f}")

print("\n3. 整體品質對價格的影響:")
print("3. Impact of overall quality on price:")
qual_price = key_data.groupby('OverallQual')['SalePrice'].mean()
lowest_qual = qual_price.min()
highest_qual = qual_price.max()
print(f"   - 最高品質(10分)的房屋平均價格是最低品質(1分)的{highest_qual/lowest_qual:.1f}倍")

print("\n4. 面積與價格:")
print("4. Area and price:")
area_corr = key_data[['GrLivArea', 'SalePrice']].corr().iloc[0,1]
print(f"   - 居住面積與價格的相關性: {area_corr:.4f}")
print(f"   - 每增加1000平方呎的居住面積，價格平均增加約: ${key_data['SalePrice'].mean()/key_data['GrLivArea'].mean()*1000:.2f}")

print("\n5. 建築類型影響:")
print("5. Building type impact:")
bldg_price = key_data.groupby('BldgType')['SalePrice'].mean().sort_values(ascending=False)
for btype, price in bldg_price.items():
    print(f"   - {btype}: ${price:.2f}")

print("\n======================================")
print("EDA分析完成! 所有圖表已保存到plots/目錄")
print("EDA analysis completed! All charts have been saved to the plots/ directory")
