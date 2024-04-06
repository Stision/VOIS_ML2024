# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:11 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import math

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

# Step 1: Checking for missing values
missing_values = cultivars_db.isna().sum()
if missing_values.sum() == 0:
    print("No missing values were found!")
else:
    print("Missing values were found:")
    print(missing_values)
    
# Step 2: Checking for data types - all cultivars are read as strings => shown as object series
#if string identified in column => object series
data_types = cultivars_db.dtypes
print("\nData types:")
print(data_types)

# Step 3: Checking for outliers
numeric_columns = cultivars_db.select_dtypes(include=['float64', 'int64']) #selecting numeric columns
column_indices = [cultivars_db.columns.get_loc(column) for column in numeric_columns.columns] #return indeces of extracted column
#NOTE: column 1 was extracted!!!

for column in numeric_columns.columns:
    print("column = " + column)
    print(cultivars_db.columns.get_loc(column))

#numeric_columns1 = cultivars_db.iloc[:,2:]
summary_statistics = numeric_columns.describe()
print("\nSummary statistics:")
print(summary_statistics)

kurt1 = kurtosis(numeric_columns) #21, 156 => heavy tails; centered around mean
skew1 = skew(numeric_columns)

#https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1
whis_factor = 3.0
lower_bound = Q1 - whis_factor * IQR #default whis argument is 1.5!!!
upper_bound = Q3 + whis_factor * IQR

# Identify outliers - no outliers in season & repetition columns + others
#big problems at column 6 (idx 5), 7 (idx 6) and 10 (idx 9)
#IQR method
outliers = numeric_columns[(numeric_columns < lower_bound) | (numeric_columns > upper_bound)]
print(outliers.count()) #outliers per column
print(outliers.count().sum()) #total outliers -> 47/ 6

#print("Outliers:", outliers.values.tolist())
#fanel = outliers.values.tolist()

# Create a box plot
plt.boxplot(numeric_columns.iloc[:,9], whis = whis_factor)
#plt.boxplot(numeric_columns)
plt.title('Box Plot of Data')
plt.show()

#Showing histogram
sns.histplot(data=numeric_columns.iloc[:,7], kde=True)
plt.show()

#z-score method
threshold = 3
#https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-sample/a/population-and-sample-standard-deviation-review
#z_scores = np.abs((numeric_columns - numeric_columns.mean() / numeric_columns.std()))
#std computes std for sample (denominator: n-1), not for population (denominator: n)
#sample standard deviation => data.std(ddof = 1) (default option)
#population standard deviation => data.std(ddof = 0)
z_scores = (numeric_columns - numeric_columns.mean()) / numeric_columns.std()
#outliers_z = numeric_columns[(z_scores > threshold).any(axis = 1)]
outliers_z = numeric_columns[(z_scores > threshold)]
#euqivalent to ((z_scores>threshold).sum(axis=0)).sum()
print(outliers_z.count()) #outliers per column
print(outliers_z.count().sum()) #total outliers -> 10


#write the labels of the outliers - IRQ
l_outliers = []
for row_label, row in outliers.iterrows():
    for column_label, value in row.items():
        if pd.notna(value):
            l_outliers.append((row_label, column_label))
            
#write the labels of the outliers - z-score
l_outliers_z = []
for row_label, row in outliers_z.iterrows():
    for column_label, value in row.items():
        if pd.notna(value):
            l_outliers_z.append((row_label, column_label))
            
#outliers list
print("Outliers found using IRQ")
print("Season | Cultivar | Parameter | Value")
for i in l_outliers:
    print(f"{cultivars_db.loc[i[0], 'Season']} | {cultivars_db.loc[i[0], 'Cultivar']} | {i[1]} | {cultivars_db.loc[i[0], i[1]]}")

print("Outliers found using Z-Score")
print("Season | Cultivar | Parameter | Value")
for i in l_outliers_z:
    print(f"{cultivars_db.loc[i[0], 'Season']} | {cultivars_db.loc[i[0], 'Cultivar']} | {i[1]} | {cultivars_db.loc[i[0], i[1]]}")


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