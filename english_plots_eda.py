"""
房價預測探索性數據分析 - 房產專家視角 (英文圖表版)
House Price Prediction EDA - Real Estate Expert Perspective (English Plots)
==============================================================================

這個腳本針對Kaggle房價預測競賽數據集進行探索性分析，重點關注房產專家最關心的特徵。
使用英文標籤以確保圖表正確顯示。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plot style
sns.set_style("whitegrid")
try:
    plt.style.use("seaborn-v0_8-pastel")
except:
    try:
        plt.style.use("seaborn-pastel")  # For newer matplotlib versions
    except:
        print("Unable to set seaborn-pastel style, using default")

# Set output directory
output_dir = 'eda_results/plots_en'
os.makedirs(output_dir, exist_ok=True)

# Load data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")

# Basic overview
print("\nSale Price basic statistics:")
print(train_data['SalePrice'].describe())

# Key features from a real estate expert's perspective
key_features = [
    # Location factors
    'Neighborhood',  # Neighborhood
    # Area factors
    'GrLivArea',     # Above grade living area
    'TotalBsmtSF',   # Total basement area
    'LotArea',       # Lot size
    # Quality factors
    'OverallQual',   # Overall quality
    'OverallCond',   # Overall condition
    # House type
    'BldgType',      # Building type
    'HouseStyle',    # House style
    # Year factors
    'YearBuilt',     # Year built
    'YearRemodAdd',  # Year remodeled
    # Important facilities
    'FullBath',      # Full bathrooms
    'BedroomAbvGr',  # Bedrooms
    'KitchenQual',   # Kitchen quality
    'GarageCars',    # Garage capacity
    'GarageArea',    # Garage area
    # Target variable
    'SalePrice'      # Sale price
]

# Check missing values for key features
missing_values = train_data[key_features].isnull().sum()
print("\nMissing values count for key features:")
print(missing_values[missing_values > 0])

# Create a dataframe with only key features
key_data = train_data[key_features].copy()

# 1. Sale price distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(key_data['SalePrice'], kde=True)
plt.title('House Sale Price Distribution')
plt.xlabel('Sale Price (USD)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(key_data['SalePrice']), kde=True)
plt.title('Log-transformed House Sale Price Distribution')
plt.xlabel('log(Sale Price+1)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{output_dir}/price_distribution.png', dpi=300)
plt.close()

# 2. Correlation analysis - relationship between numeric features and price
numeric_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual', 'OverallCond', 
                    'YearBuilt', 'YearRemodAdd', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'GarageArea']

plt.figure(figsize=(10, 8))
correlation = key_data[numeric_features + ['SalePrice']].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
plt.title('Correlation Between Numeric Features and Sale Price')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300)
plt.close()

# Print correlation ranking with sale price
price_correlation = correlation['SalePrice'].sort_values(ascending=False)
print("\nCorrelation ranking with sale price:")
print(price_correlation)

# 3. Analyze price distribution across different neighborhoods
plt.figure(figsize=(14, 8))
sns.boxplot(x='Neighborhood', y='SalePrice', data=key_data)
plt.title('House Price Distribution Across Neighborhoods')
plt.xlabel('Neighborhood')
plt.ylabel('Sale Price (USD)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{output_dir}/neighborhood_prices.png', dpi=300)
plt.close()

# Calculate and sort average price by neighborhood
neighborhood_avg_price = key_data.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
print("\nNeighborhood average price ranking from high to low:")
print(neighborhood_avg_price)

# 4. Analyze relationship between overall quality and price
plt.figure(figsize=(10, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=key_data)
plt.title('House Price Distribution by Overall Quality')
plt.xlabel('Overall Quality (1-10)')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/quality_vs_price.png', dpi=300)
plt.close()

# 5. Analyze relationship between living area and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('Living Area vs. Sale Price (colored by Overall Quality)')
plt.xlabel('Living Area (square feet)')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/area_vs_price.png', dpi=300)
plt.close()

# 6. Analyze relationship between year built and price
plt.figure(figsize=(12, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', hue='OverallQual', palette='viridis', data=key_data)
plt.title('Year Built vs. Sale Price (colored by Overall Quality)')
plt.xlabel('Year Built')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/year_vs_price.png', dpi=300)
plt.close()

# 7. Analyze price distribution across different building types
plt.figure(figsize=(10, 6))
sns.boxplot(x='BldgType', y='SalePrice', data=key_data)
plt.title('House Price Distribution by Building Type')
plt.xlabel('Building Type')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/building_type_prices.png', dpi=300)
plt.close()

# 8. Analyze relationship between kitchen quality and price
plt.figure(figsize=(10, 6))
order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']  # Sort from low to high
sns.boxplot(x='KitchenQual', y='SalePrice', data=key_data, order=order)
plt.title('House Price Distribution by Kitchen Quality')
plt.xlabel('Kitchen Quality')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/kitchen_quality_prices.png', dpi=300)
plt.close()

# 9. Analyze relationship between garage capacity and price
plt.figure(figsize=(10, 6))
sns.boxplot(x='GarageCars', y='SalePrice', data=key_data)
plt.title('House Price Distribution by Garage Capacity')
plt.xlabel('Garage Capacity (car count)')
plt.ylabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig(f'{output_dir}/garage_capacity_prices.png', dpi=300)
plt.close()

# 10. Relationship between house age and price - creating new features
key_data['HouseAge'] = train_data['YrSold'] - train_data['YearBuilt']
key_data['RemodAge'] = train_data['YrSold'] - train_data['YearRemodAdd']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='HouseAge', y='SalePrice', data=key_data)
plt.title('House Age vs. Sale Price')
plt.xlabel('House Age (Year Sold - Year Built)')
plt.ylabel('Sale Price (USD)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='RemodAge', y='SalePrice', data=key_data)
plt.title('Time Since Remodeling vs. Sale Price')
plt.xlabel('Remodel Age (Year Sold - Year Remodeled)')
plt.ylabel('Sale Price (USD)')

plt.tight_layout()
plt.savefig(f'{output_dir}/age_vs_price.png', dpi=300)
plt.close()

# Real Estate Expert Summary
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
print(f"   - The average price of houses with the highest quality (10) is {highest_qual/lowest_qual:.1f} times that of houses with the lowest quality (1)")

print("\n4. 面積與價格:")
print("4. Area and price:")
area_corr = key_data[['GrLivArea', 'SalePrice']].corr().iloc[0,1]
print(f"   - 居住面積與價格的相關性: {area_corr:.4f}")
print(f"   - Correlation between living area and price: {area_corr:.4f}")
avg_price_increase = key_data['SalePrice'].mean()/key_data['GrLivArea'].mean()*1000
print(f"   - 每增加1000平方呎的居住面積，價格平均增加約: ${avg_price_increase:.2f}")
print(f"   - For each additional 1000 square feet of living area, the price increases by approximately: ${avg_price_increase:.2f}")

print("\n5. 建築類型影響:")
print("5. Building type impact:")
bldg_price = key_data.groupby('BldgType')['SalePrice'].mean().sort_values(ascending=False)
for btype, price in bldg_price.items():
    print(f"   - {btype}: ${price:.2f}")

print("\n======================================")
print(f"EDA分析完成! 所有英文標題圖表已保存到{output_dir}目錄")
print(f"EDA analysis completed! All charts with English titles have been saved to the {output_dir} directory")
print("注意：為了確保圖表能夠正確顯示，本版本使用了純英文標題")
print("Note: To ensure proper display of charts, this version uses English-only titles")
