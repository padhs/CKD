import pandas as pd
import os
import numpy as np
import seaborn as sns
import plotly.express as pltex
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as pyplt

pyplt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 25)

# information about the dataset
# df = pd.read_csv('./datasets/new_kidney_disease.csv')
df = pd.read_csv('./datasets/chronic_kidney_disease_full.csv')

print(df.shape)
# should return (400, 25)
print(df.describe())

print(df.info())
# shows dtype -- everything is object type

print(df.isnull().sum())
# will print the total no. of null values in each column

# renaming the columns to understand them better
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
              'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
              'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume',
              'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
              'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia', 'class']

columns_to_be_converted = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'blood_glucose_random',
                         'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',
                         'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']

# converting object dtype to numerical data
for cols in df:
    for data in columns_to_be_converted:
        if df[cols].name == data:
            df[cols] = pd.to_numeric(df[cols], errors='coerce')
        else:
            df[cols] = df[cols]


print(df.info())
# float- 14, object - 11
print(df.head())
# the '?' is represented as NaN (not a number) in numerical columns
print(df.describe())

# extracting categorical and numerical columns:
categorical_columns = [column for column in df.columns if df[column].dtype == object]
numerical_columns = [column for column in df.columns if df[column].dtype != object]

