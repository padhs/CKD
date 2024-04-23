import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/chronic_kidney_disease_EDA.csv')

# Exploratory Data Analysis -->
print(df.head())
print(df.info())
print(df.isnull().sum())


