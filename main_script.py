# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:11 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#40 soybean cultivars * 4 replications * 2 seasons = 320 samples
#first sowing - 11 Nov 2022 (? 2023 in article)
#second sowing - 02 Dec 2022 (? 2023 in article)
#each exp unit - 8 rows, spaced 0.5m apaart and 10m long => 40 m2
#only four central lines considered => 16 m2

#1 seed per ml -> 20.000 density per ml

#PH = plant height
#IFP = insertion of first pod
#NS = number of stems
#NLP = number of legumes per plant
#NGP = number of grains per plant
#NGL = number of grains per pod
#TSW = thousand seed weight
#GY = Grain yield
cultivars_db = pd.read_csv('data/data.csv')
print(cultivars_db.shape)

# Step 1: Check for missing values
missing_values = cultivars_db.isna().sum()
if missing_values.sum() == 0:
    print("No missing values were found!")
else:
    print("Missing values were found:")
    print(missing_values)
    
# Step 2: Check for data types - all cultivars are read as strings => shown as object series
#if string identified in column => object series
data_types = cultivars_db.dtypes
print("\nData types:")
print(data_types)

# Step 3: Check for outliers or implausible values (example for numeric columns)
numeric_columns = cultivars_db.select_dtypes(include=['float64', 'int64'])
numeric_columns1 = cultivars_db.iloc[:,2:]
summary_statistics = numeric_columns.describe()
print("\nSummary statistics:")
print(summary_statistics)


#print(cultivars_db.head())
print(cultivars_db.describe())
#fanelus = cultivars_db.to_numpy()

cultivarsDB_simple = cultivars_db.iloc[:,3:]

correlation_matrix = cultivarsDB_simple.corr()
s = correlation_matrix.unstack()
max_s = max(s[s != 1])
print(max_s)
min_s = min(s)
print(min_s)
#print(max(correlation_matrix))
#print(s.iloc[5])

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cross-Correlation Matrix')
plt.show()

#print(fanelus[2, 3])