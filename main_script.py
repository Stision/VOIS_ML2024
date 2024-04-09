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

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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

#PART 1: PREPROCESSING DATA (name verification of cultivars names between files was done in part 2)

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
#EXTRA: Exercise 13
cultivars_db = pd.read_csv('data/data_noOutliers.csv')

#CASE 1: FOR ALL THE CULTIVARS
MHG_Season1 = cultivars_db.loc[cultivars_db['Season'] == 1, 'MHG']
MHG_Season2 = cultivars_db.loc[cultivars_db['Season'] == 2, 'MHG']
delta_MHG = MHG_Season2.mean() - MHG_Season1.mean() #12.067 => increase

GY_Season1 = cultivars_db.loc[cultivars_db['Season'] == 1, 'GY']
GY_Season2 = cultivars_db.loc[cultivars_db['Season'] == 2, 'GY']
delta_GY = GY_Season2.mean() - GY_Season1.mean() #-19.797 => decrease

#CASE 2: FOR THE SAME CULTIVARS
#sort = False keeps the same order from cultivars_db
cultivar_grouped = cultivars_db.groupby('Cultivar', sort = False)
#cultivar_grouped.first()
#cultivar_grouped.get_group('NEO 760 CE')

delta_list = []
for group_name, group_data in cultivar_grouped:
    #print("Group:", group_name)
    #print(group_data)
    MHG_Season1_temp = group_data.loc[group_data['Season'] == 1, 'MHG'].mean()
    MHG_Season2_temp = group_data.loc[group_data['Season'] == 2, 'MHG'].mean()
    MHG_delta = MHG_Season2_temp - MHG_Season1_temp
    
    GY_Season1_temp = group_data.loc[group_data['Season'] == 1, 'GY'].mean()
    GY_Season2_temp = group_data.loc[group_data['Season'] == 2, 'GY'].mean()
    GY_delta = GY_Season2_temp - GY_Season1_temp
    
    dict_delta = {'Cultivar Name': group_name, 'Delta MHG': MHG_delta, 'Delta GY': GY_delta}
    delta_list.append(dict_delta)

cultivar_delta = pd.DataFrame(delta_list)
#cultivar_delta.loc[:,'Delta MHG'].mean()#verification1: 12.067
#cultivar_delta.loc[:,'Delta GY'].mean()#verification2: -19.797
print(cultivar_delta)


#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#PART 2: ANALYZING DATA (ex 10 - 12; 14)
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
comp1 = cultivars_df['Cultivar']
comp2 = description_df['Cultivars']

not_found_cultivars = []
for term in description_df['Cultivars']:
    if term not in cultivars_df['Cultivar'].values:
        not_found_cultivars.append(term)
        


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

description_list = []


for term_cultivar in cultivars_df['Cultivar']:
    for term_description in description_df['Cultivars']:
        if term_description == term_cultivar:
            temp = description_df[description_df.eq(term_description).any(axis=1)]
            description_dict = {"Maturation group":temp.iloc[0, 1], "Seeds per meter/linear": temp.iloc[0, 2], "Density per meter/linear": temp.iloc[0, 3]}
            description_list.append(description_dict)
            
            

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


#PART 2.2: MULTIPLE LINEAR REGRESSION
cultivars_df = pd.read_csv('data/data_unified.csv')
kf = KFold(n_splits=5, shuffle=True, random_state=42) #kfold cross-validation
cv_list = []
perm_list = []
#Analysis (for both MGH and GY):
#1) Coefficients from Multiple Regression
#2) MSE from Multiple Regression and Random Forest Regressor
#3) Gini Importance for Random Forest Regressor
#4) Permutation Feature Importance for both Multiple Regression and Random Forest Regressor
#5) Cross-validation for both Multiple Regression and Random Forest Regressor

#PART 2.2.1 - MHG (std = 0.221)
#data preparation - split the data into training and testing sets
#NOTE: having highly correlated predictor variables adds redundancy to the model.
x = cultivars_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
#MHG is removed becaused it is the target feature
#Cultivar is removed because it contains only strings
#Density was removed because it is strongly correlated to Seeds (>0.99)
#NGP was removed because it is strongly correlated to NLP (0.825)
y = cultivars_df['MHG']  #target feature

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#selecting regression model
model = LinearRegression()

#applying kfold cross-validation
cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
#cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='r2')
cv_scores = -cv_scores

print("Cross-validation MSE scores:", cv_scores)
print("Mean MSE:", cv_scores.mean()) #consistent scores

dict_cv = {f'Split {i+1}': score for i, score in enumerate(cv_scores)}
dict_cv['Mean'] = np.mean(cv_scores)
dict_cv['Name'] = 'Linear Regression - MHG'
cv_list.append(dict_cv)
#cultivar_cv_scores = pd.DataFrame(cv_list)

#training the regression model
model.fit(x_train, y_train)

#computing permutation feature importance
perm_importance = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
dict_perm = dict(zip(x_train.columns, perm_importance['importances_mean']))
dict_perm['Name'] = 'Linear Regression - MHG'
perm_list.append(dict_perm)

#evaluating the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

#printing the results
coefficients = pd.DataFrame({'Feature': x.columns, 'Coefficient': model.coef_})

#mse = 0.0443
#MAIN COEFF: PH(-0.283), GY (0.265), Seeds per meter (0.179), Season number (0.170)

#WE NEED TO TAKE INTO ACCOUNT BOTH ANALYSIS!!!!

#PART 2.2.2 - GY (std = 0.149)
#data preparation - split the data into training and testing sets
x = cultivars_df.drop(['GY', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
#GY is removed becaused it is the target feature
#Cultivar is removed because it contains only strings
#Density was removed because it is strongly correlated to Seeds (>0.99)
#NGP was removed because it is strongly correlated to NLP (0.825)
y = cultivars_df['GY']  #target feature

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#selecting regression model
model = LinearRegression()

#applying kfold cross-validation
cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
#cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='r2')
cv_scores = -cv_scores

print("Cross-validation MSE scores:", cv_scores)
print("Mean MSE:", cv_scores.mean()) #consistent scores

dict_cv = {f'Split {i+1}': score for i, score in enumerate(cv_scores)}
dict_cv['Mean'] = np.mean(cv_scores)
dict_cv['Name'] = 'Linear Regression - GY'
cv_list.append(dict_cv)
#cultivar_cv_scores = pd.DataFrame(cv_list)

#training the regression model
model.fit(x_train, y_train)

#computing permutation feature importance
perm_importance = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
dict_perm = dict(zip(x_train.columns, perm_importance['importances_mean']))
dict_perm['Name'] = 'Linear Regression - GY'
perm_list.append(dict_perm)

#evaluating the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

#printing the results
coefficients = pd.DataFrame({'Feature': x.columns, 'Coefficient': model.coef_})

#mse = 0.0180
#MAIN COEFF: Maturation Group (0.306), NS (0.200), MHG (-0.111), IFP (0.084)

#PART 2.3 Randomm Forest
cultivars_df = pd.read_csv('data/data_unified.csv')

#PART 2.3.1 - MHG (std = 0.221)
x = cultivars_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y = cultivars_df['MHG']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_scores = -cv_scores

print("Cross-validation MSE scores:", cv_scores)
print("Mean MSE:", cv_scores.mean()) #consistent scores

dict_cv = {f'Split {i+1}': score for i, score in enumerate(cv_scores)}
dict_cv['Mean'] = np.mean(cv_scores)
dict_cv['Name'] = 'Random Forest - MHG'
cv_list.append(dict_cv)

model.fit(x_train, y_train)

perm_importance = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
dict_perm = dict(zip(x_train.columns, perm_importance['importances_mean']))
dict_perm['Name'] = 'Random Forest - MHG'
perm_list.append(dict_perm)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse) 

#printing feaeture importances
importances = model.feature_importances_ #Gini feature importance
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

#mse = 0.0132
#MAIN FEATURES: Maturation Group (0.238), GY (0.208), Seeds (0.134), PH (0.091)

#PART 2.3.2 - GY (std = 0.149)
x = cultivars_df.drop(['GY', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y = cultivars_df['GY']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_scores = -cv_scores

print("Cross-validation MSE scores:", cv_scores)
print("Mean MSE:", cv_scores.mean()) #consistent scores

dict_cv = {f'Split {i+1}': score for i, score in enumerate(cv_scores)}
dict_cv['Mean'] = np.mean(cv_scores)
dict_cv['Name'] = 'Random Forest - GY'
cv_list.append(dict_cv)

model.fit(x_train, y_train)

perm_importance = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42)
dict_perm = dict(zip(x_train.columns, perm_importance['importances_mean']))
dict_perm['Name'] = 'Random Forest - GY'
perm_list.append(dict_perm)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse) 

#printing feaeture importances
importances = model.feature_importances_ #Gini feature importance
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

#mse = 0.0132
#MAIN FEATURES: MHG (0.200), Maturation Group (0.197), NS (0.144), Seeds (0.110)
cultivar_cv_scores = pd.DataFrame(cv_list)
cultivar_perm_importance = pd.DataFrame(perm_list)


#PART 3: New Cultivar Generation. MHG prediction (ex 15 - 18)

#PART 3.1: Creating new synthetic cultivar (MHG also generated)
cultivars_df = pd.read_csv('data/data_unified.csv')
cultivar_df_season1 = cultivars_df.iloc[:160, 1:]
cultivar_df_season2 = cultivars_df.iloc[160:, 1:]

buffer_season1 = []
buffer_season2 = []
for _, group in cultivars_df.iloc[:160, 2:].groupby('Repetition'):
    buffer_season1.append(group)
for _, group in cultivars_df.iloc[160:, 2:].groupby('Repetition'):
    buffer_season2.append(group)

data = {'Season': [0]*4 + [1]*4, 'Cultivar': ['VOIS_ML2024']*8}
new_cultivar_ls1 = []
new_cultivar_ls2 = []
for temp in buffer_season1:
    mean_value = temp.mean().to_dict()
    new_cultivar_ls1.append(mean_value)
for temp in buffer_season2:
    mean_value = temp.mean().to_dict()
    new_cultivar_ls2.append(mean_value)
fanel = new_cultivar_ls1 + new_cultivar_ls2
new_cultivar_df = pd.concat([pd.DataFrame(data), pd.DataFrame(new_cultivar_ls1 + new_cultivar_ls2)], axis = 1)

new_cultivar_df.to_csv('data/new_cultivar.csv', index=False)

#PART 3.2: Verifying new cultivar using clusters (k-means)
#NOTE: a new file was created, including all the previous cultivars and the new one - data_full.csv 
full_cultivar_df = pd.read_csv('data/data_full.csv')
#x = full_cultivar_df.drop(['Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
x = full_cultivar_df.drop(['Cultivar', 'Density per meter/linear', 'NGP', 'Season', 'Repetition'], axis = 1) 
#Repetion and season removed because the cluster seems to "respect" only those 2 variables 
#EX: Cluster 0 = season 1 + repetition 0.67 OR 1
#EX: Cluster 1 = season 0 + repetition 0 OR 0.333
#EX: Cluster 2 = season 0 + repetition 0.67 OR 1
#EX: Cluster 3 = season 1 + repetition 0 OR 0.33

k_values = range(1, 15)
wcss = [] #list to store the within-cluster sum of squares
for k in k_values:
    model = KMeans(n_clusters=k)
    model.fit(x)
    wcss.append(model.inertia_)
    
# Plot the Elbow Curve
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

model = KMeans(n_clusters = 4) #k_optimum = 4 or 10
model.fit(x)
#results good for 4

clusters = model.predict(x)
_, label_counts = np.unique(clusters, return_counts=True)

full_cultivar_df['Cluster'] = clusters

grouped_fc_df = full_cultivar_df.groupby('Cluster')
combined_fc_df = pd.concat([group for _, group in grouped_fc_df])

#models lists: multiple regression (unoptimized), decision tree (unoptimized), 
#kmeans (4 clusters), kmeans (10 clusters - mayber)
#random forest (optimized), Gradient Boosting Machine (maybe)



#PART 3.3: MHG prediction for new cultivar
#PART 3.3.1: RANDOM FOREST REGRESSOR - ALL FEATURES (NON-REDUNDANT/ STRING)
cultivars_df = pd.read_csv('data/data_unified.csv')
statistic = cultivars_df.describe()

x = cultivars_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y = cultivars_df['MHG']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)#0.013

initial_params = model.get_params()
print("Initial Hyperparameters:")
print("n_estimators:", initial_params['n_estimators']) #100
print("max_depth:", initial_params['max_depth']) #None
print("min_samples_split:", initial_params['min_samples_split']) #2
print("min_samples_leaf:", initial_params['min_samples_leaf']) #1

#Optimizing model
hyperParam_list = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20],       # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],    # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]       # Minimum number of samples required at each leaf node
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=hyperParam_list,
                           scoring='neg_mean_squared_error',
                           cv=5,  # 5-fold cross-validation, as in data analysis part
                           verbose=1,
                           n_jobs=-1)  # Use all available CPU cores

#perform grid search
grid_search.fit(x_train, y_train)

#best parameters found using grid search
print("Best Parameters:", grid_search.best_params_)

#best model found
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)#0.013

#predicting MHG for new cultivar and compairing with actual results
new_cultivar_df = pd.read_csv('data/new_cultivar.csv')
statistic = new_cultivar_df.describe()
x_new_test = new_cultivar_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y_new_test = new_cultivar_df['MHG']

y_new_pred = best_model.predict(x_new_test)

mse = mean_squared_error(y_new_test, y_new_pred)
print("Mean Squared Error:", mse)#0.029

#PART 3.3.2: RANDOM FOREST REGRESSOR - Determinant Features (3 or 5???)
#----------------------------------------------------------------------


#PART 3.3.3: Gradient Boosting Machine - ALL FEATURES
cultivars_df = pd.read_csv('data/data_unified.csv')
#one_hot_encoding = pd.get_dummies(cultivars_df['Cultivar'])
#encoded_data = pd.concat([cultivars_df, one_hot_encoding], axis=1)
statistic = cultivars_df.describe()

x = cultivars_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y = cultivars_df['MHG']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R^2 Score:", score)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)#0.0135


#Optimizing model - 2430 models (5 folds for 486 candidates)
hyperParam_list = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.05, 0.1, 0.2],  # Learning rate
    'max_depth': [3, 4, 5],  # Maximum depth of the individual regression estimators
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'subsample': [0.8, 1.0]  # Subsample ratio of the training instances
}

grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                           param_grid=hyperParam_list,
                           scoring='neg_mean_squared_error',
                           cv=5,  # 5-fold cross-validation, as in data analysis part
                           verbose=1,
                           n_jobs=-1)  # Use all available CPU cores

#perform grid search
grid_search.fit(x_train, y_train)

#best parameters found using grid search
print("Best Parameters:", grid_search.best_params_)

#best model found
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)#0.0124

#predicting MHG for new cultivar and compairing with actual results
new_cultivar_df = pd.read_csv('data/new_cultivar.csv')
statistic = new_cultivar_df.describe()
x_new_test = new_cultivar_df.drop(['MHG', 'Cultivar', 'Density per meter/linear', 'NGP'], axis = 1) 
y_new_test = new_cultivar_df['MHG']

y_new_pred = best_model.predict(x_new_test)

mse = mean_squared_error(y_new_test, y_new_pred)
print("Mean Squared Error:", mse)#0.0371



#TEST - ONE HOT ENCODER
cultivars_df = pd.read_csv('data/data_unified.csv')
statistic = cultivars_df.describe()

#x = cultivars_df.drop(['MHG', 'Density per meter/linear', 'NGP'], axis = 1) 
x = cultivars_df.drop(['MHG'], axis = 1) 
y = cultivars_df['MHG']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the column transformer to apply one-hot encoding to 'Cultivar' column
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Cultivar'])  # Assuming 'Cultivar' is the column containing cultivar names
    ],
    remainder='passthrough'  # Keep other numerical features unchanged
)



# Define the pipeline with preprocessor and Gradient Boosting Regressor model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())  # You can customize hyperparameters here
])

transformed_data = model_pipeline.named_steps['preprocessor'].transform(x_train)
transformed_df = pd.DataFrame(transformed_data)

model_pipeline.fit(x_train, y_train)

score = model_pipeline.score(x_test, y_test)
print("R^2 Score:", score)

y_pred = model_pipeline.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)#0.006579875204994996



#TEST 2
cultivars_df = pd.read_csv('data/data_unified.csv')
statistic = cultivars_df.describe()

# Initialize the OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

# Fit and transform the 'Cultivar' column
one_hot_encoded_column = one_hot_encoder.fit_transform(cultivars_df[['Cultivar']])

# Convert the one-hot encoded column into a DataFrame
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_column, columns=one_hot_encoder.get_feature_names_out(['Cultivar']))

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df_encoded = pd.concat([cultivars_df.drop(columns=['Cultivar']), one_hot_encoded_df], axis=1)

# Display the DataFrame with one-hot encoding applied
print(df_encoded)


#test 3
cultivars_unified_df = pd.read_csv('data/data_test.csv')
#numeric_columns = cultivars_df.drop(['Cultivar'], axis = 1) 
numeric_columns = cultivars_unified_df[['Maturation group', 'Seeds per meter/linear', 'Test']]

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





