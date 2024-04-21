import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# information about the dataset
# df = pd.read_csv('./datasets/new_kidney_disease.csv')
df = pd.read_csv('./datasets/chronic_kidney_disease_full.csv')
print(df.head())

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

# Data Pre-processing -->

# converting object dtype to numerical data & replacing '?' values with NaN (Not a Number)
for cols in df:
    for data in columns_to_be_converted:
        df[cols].replace(to_replace={'?': np.nan}, inplace=True)
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

# looking for unique variables in any categorical columns
for column in categorical_columns:
    print(f'The column:{column} has {df[column].unique()} values\n')

# diabetes_mellitus has '\tno' and '\tyes' values
# coronary_artery_disease has '\tno' values
# class has 'ckd\t' values

# replacing these values with yes, no and ckd for column respectively:a
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'].replace(to_replace={'\tno': 'no'}, inplace=True)
df['class'].replace(to_replace={'ckd\t': 'ckd'}, inplace=True)

# checking for null values
print(df.isna().sum().sort_values(ascending=False))

print(df[numerical_columns].isnull().sum().sort_values(ascending=False))
print(df[categorical_columns].isnull().sum().sort_values(ascending=False))
print(f"These are the numerical columns: {df[numerical_columns]}\n")

# filling null values, we will use 2 methods:

# random sampling for higher null-value columns
# mean/mode sampling for lower null values


def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample


mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[numerical_columns] = mean_imputer.fit_transform(df[numerical_columns])

for column in categorical_columns:
    if column == 'red_blood_cells' and column == 'pus_cells':
        random_value_imputation(column)

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[categorical_columns] = mode_imputer.fit_transform(df[categorical_columns])

# confirm all null values are filled
print(df.isna().sum())

# Feature Encoding:  (converting categorical columns to numeical columns)
for column in categorical_columns:
    print(f"{column} has {df[column].nunique()} categories\n")

# every column has 2 categories--> use LabelEncoder:
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

print(df.head())
print(df.info())
# categorical values are now replaced with 0/1 values.

# rounding off decimals
for column in df.columns:
    df[column] = df[column].round(2)

print(df.head())

df.to_csv('./datasets/kidney_disease_ready_round.csv', header=True, index=False)
