# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:11 2024

@author: Admin
"""

import pandas as pd
#import pandas.io.formats.odf
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
print(cultivars_db.describe())
statistic1 = cultivars_db.describe()

#PART 1: PREPROCESSING DATA (name verification of cultivars between files & min-max scaling were done in part 2)

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
print(f"Outliers found using IRQ ({len(l_outliers)} elements)")
print("Season | Cultivar | Parameter | Value")
for i in l_outliers:
    print(f"{cultivars_db.loc[i[0], 'Season']} | {cultivars_db.loc[i[0], 'Cultivar']} | {i[1]} | {cultivars_db.loc[i[0], i[1]]}")

print("Outliers found using Z-Score ({len(l_outliers_z)} elements)")
print("Season | Cultivar | Parameter | Value")
for i in l_outliers_z:
    print(f"{cultivars_db.loc[i[0], 'Season']} | {cultivars_db.loc[i[0], 'Cultivar']} | {i[1]} | {cultivars_db.loc[i[0], i[1]]}")

#replace values - using only irq rule oultiers
#OPTION 1: replace with median value (non-uniform distro)
#OPTION 2: replace with mean value (uniform distro)
#OPTION 3: replace with upper_bound values > upper_bound or 
#with lower_bound values < lower_bound
#NOTE: This should be applied to the SAME cultivars, for one season/ both seasons!!!!


list_non_outliers = []
for ref in l_outliers:
    ref_cultivar_name = cultivars_db.loc[ref[0], "Cultivar"]
    temp_non_outliers = []
    for row_label, row in cultivars_db.iterrows():
        if row_label != ref[0] and row.iloc[1] == ref_cultivar_name: #exlude the record of the outlier
            temp_non_outliers.append(row_label)
    list_non_outliers.append(temp_non_outliers) #list_values always has 7 values!
    
#for l_outliers[0]
#season 1 => mean value = 2.753; median value = 2.550
#season 2 => mean value = 2.2375; median value = 2.135
#both seasons => mean_value = 2.458; median value = 2.44

#for l_outliers[1]
#season 1 => mean value = 222.533; median value = 220.865
#season 2 => mean value = 163.843; median value = 101.000
#both seasons => mean_value = 197.380; median value = 219.2

#for l_outliers[2]
#season 1 => mean value = 2.563; median value = 2.615
#season 2 => mean value = 3.603; median value = 2.21
#both seasons => mean_value = 3.009; median value = 2.56        

idx = 2
ref = l_outliers[idx]
print("Cultivar | Season | Value")
print(f"{cultivars_db.loc[ref[0], 'Cultivar']} | {cultivars_db.loc[ref[0], 'Season']} | {cultivars_db.loc[ref[0], ref[1]]} (OUTLIER)")
for i in list_non_outliers[idx]:
    print(f"{cultivars_db.loc[i, 'Cultivar']} | {cultivars_db.loc[i, 'Season']} | {cultivars_db.loc[i, ref[1]]}")


#SOLUTION: MEDIAN VALUE, FOR EACH SEASON!
#median value because values vary greatly in the same season
#for each season because values tend to vary greatly between seasons

#normalized values (wihtout outliers)
outliers_norm_values = []
for outlier_idx in range(0, len(l_outliers)):
    temp_list = []
    ref = l_outliers[outlier_idx][0]
    for i in list_non_outliers[outlier_idx]:
        if i > ref - 4 and i < ref + 4: #cultivars are grouped 4 by 4
            temp_list.append(i) #each time it will have 3 elements
            #print(i)
    
    values_list = []
    ref_label = l_outliers[outlier_idx][1]
    for i in temp_list:
        values_list.append(cultivars_db.loc[i, ref_label])
    #print(values_list)
    values_list.sort()
    #print(values_list)
    outliers_norm_values.append(values_list[1]) #median value = element with index 1
        
#replacing outliers with normalized values
for outlier_idx in range(0, len(l_outliers)):
    ref = l_outliers[outlier_idx]
    cultivars_db.loc[ref[0], ref[1]] = outliers_norm_values[outlier_idx]
    
#writing dataframe to csv (file without outliers, without scaling)
cultivars_db.to_csv('data/data_noOutliers.csv', index=False)

#analyzing results
cultivars_db = pd.read_csv('data/data_noOutliers.csv')
#print(cultivars_db.describe())
statistic2 = cultivars_db.describe()


#Step 4: Min-max feature scaling (we've removed outliers & we want to preserve the original distribution)
numeric_columns = cultivars_db.select_dtypes(include=['float64', 'int64'])
numeric_columns_minMax = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
cultivars_db_minMax = cultivars_db

for column in numeric_columns_minMax:
    cultivars_db_minMax[column] = numeric_columns_minMax[column]

#writing dataframe to csv (file without outliers, with min-max scaling)
cultivars_db_minMax.to_csv('data/data_normalized.csv', index=False)

#analyzing results
cultivars_db_minMax = pd.read_csv('data/data_normalized.csv')
statistic3 = cultivars_db_minMax.describe()

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#PART 2: ANALYZING DATA
#PART 2.1: CORRELATION COEFFICIENTS
#in cultivars-description.ods, we can see that for every record, the following formula applies:
#density = seeds *20000
#with the exception of LAT 1330BT.11 and LTT 7901 IPRO (aprox seeds * 19000)

#FILE 1: cultivars-description.ods
description_df = pd.read_excel('data/cultivars-description.ods', engine = 'odf')
numeric_columns = description_df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_columns.corr()
s = correlation_matrix.unstack()
max_s = max(s[s != 1])
print(max_s)
min_s = min(s)
print(min_s)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cross-Correlation Matrix')
plt.show()

#1) formula from above demonstrated by the correlation matrix => we can use either the seed column
#or the density column (using both will be redundant) => we choose seeds column
#2) we observe a strong negative correlation (-0.601) between the maturation group and seed column
#which can be further used in analyzing the data

#FILE 2: data_normalized.csv
cultivars_df = pd.read_csv('data/data_normalized.csv')
#cultivars_df = pd.read_csv('data/data_normalized.csv')
numeric_columns = cultivars_df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_columns.corr()
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

#1) we observe a moderate negative correlation(-0.506) between NS (number of stems) and season number
#2) we observe a strong positive correlation (0.825) between NGP (number of grains per plant) and NLP (number of legumes per plant)
#-> normal result, because every plant contains multiples legumes, while every legume contains multipe grains (NGP > NLP always)
#3) we have moderate positive correlation (0.522) between NS (number of stems) and NLP (number of legumes per plant)
#4) we have moderate positive correlation (0.517) between NS (number of stems) and NGP (number of grains per plant))
#NOTE: normal result, considering the fact that NLP and NGP are strongly correlated
#5) we have moderate negative correlation (-0.506) between NS and season number
#6) we have relatively weak correlation (0.308) between MHG and season number
#7) next, we have a very weak negative correlation (-0.129) between MHG and NS
#8) we have relatively weak positive correlation (0.260) between GY and NGP
#9) we have a weak correlation positive correlation (0.199) between GY and NLP


#Creation data_unified.csv - contains both data_normalized.csv and cultivars-description.ods
cultivars_df = pd.read_csv('data/data_normalized.csv')
description_df = pd.read_excel('data/cultivars-description.ods', engine = 'odf')

#first we verify if all the names from cultivars-description.ods are written as in data_normalized.csv
name_status = []
for i, j in description_df.iterrows():
    flag = False
    for row_label, row in cultivars_df.iterrows():
        if j.iloc[0] == row.iloc[1]:
            flag = True
            break
    name_status.append((j.iloc[0], flag))
    if not flag:
        print(j.iloc[0])
    #print(i, j)
#NAMES NOT IDENTIFIED: M 8606I2X, BRASMAX OLÍMPO IPRO, LAT 1330BT.11, GNS7900IPRO - AMPLA, GNS7700IPRO
#CORRECT REPRESENTATION IN FILES (cultivars-description -> data_normalized):
#1) M 8606I2X -> MONSOY M8606I2X
#2) BRASMAX OLÍMPO IPRO -> BRASMAX OLIMPO IPRO
#3) LAT 1330BT.11 -> LAT 1330BT
#4) GNS7900IPRO - AMPLA -> GNS7900 IPRO - AMPLA
#5) GNS7700IPRO -> GNS7700 IPRO

#Rewritting new .ods file with corrected name => cultivars-description_corrected.ods
description_df = pd.read_excel('data/cultivars-description_corrected.ods', engine = 'odf')

l_maturation = []
l_seed = []
l_density = []

for row_label, row in cultivars_df.iterrows():
    for i, j in description_df.iterrows():
        if j.iloc[0] == row.iloc[1]:
            l_maturation.append(j.iloc[1])
            l_seed.append(j.iloc[2])
            l_density.append(j.iloc[3])
    #print(row_label, row)

#min-max normalizing values from cultivars-description
def minMax_scaling(data):
    minimum = min(data)
    maximum = max(data)
    minMax_data = [(n - minimum) / (maximum - minimum) for n in data]
    return minMax_data

l_maturation_scaled = minMax_scaling(l_maturation)
l_seed_scaled = minMax_scaling(l_seed)
l_density_scaled = minMax_scaling(l_density)

#writing new columns to dataframe
cultivars_df['Maturation group'] = l_maturation_scaled
cultivars_df['Seeds per meter/linear'] = l_seed_scaled
cultivars_df['Density per meter/linear'] = l_density_scaled

#writing unified dataframe to new file
cultivars_df.to_csv('data/data_unified.csv', index=False)


#FILE 3: data_unified.csv
cultivars_unified_df = pd.read_csv('data/data_unified.csv')
numeric_columns = cultivars_unified_df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_columns.corr()
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
#1) we have relatively weak correlation (0.382) between GY and maaturation group
#2) we have relatively weak negative correlation (-0.240) between GY and seeds per meter (and density)
#3) we have a very weak positive correlation (0.183) between MHG and seeds per meter

#STRONGEST CORRELATION FOR MHG AND GY:
#1) MHG -> Season number (0.308); Seeds per meter (0.183); NS (-0.129)
#2) GY -> Maturation group (0.382); NGP (0.260); Seeds per meter (-0.240); 

#PART 2.2: MULTIPLE REGRESSION


















