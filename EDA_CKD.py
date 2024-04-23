import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

import matplotlib.style as style

style.use('fivethirtyeight')

df = pd.read_csv('./datasets/kidney_disease_ready_round.csv')

# Exploratory Data Analysis -->
print(df.head())
print(df.info())
print(df.isnull().sum())

# Feature distributions -->
plt.figure(figsize=(20, 15))
plotnumber = 1

for column in df.columns:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)

    plotnumber += 1

plt.tight_layout()
plt.show()

# heatmaps of features-->
plt.figure(figsize=(28, 20), dpi=120)
sns.heatmap(df.corr(), annot=True, linewidths=5, linecolor='lightgrey',)
plt.tight_layout()
plt.show()


def violin_plot(col):
    fig = px.violin(df, y=col, x='class', box=True, template='plotly_dark')
    return fig.show()


def kde_plot(col):
    grid = sns.FacetGrid(df, hue='class', aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    return plt.show()


def scatter_plot(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color='class', template='plotly_dark')
    return fig.show()


# From Pearson's correlation matrix (correlation hypothesis)--->
# specific_gravity, hemoglobin, packed_cell_volume, blood_glucose_random, blood_urea, red_blood_cells, pus_cell, sodium, red_blood_cell_count has positive correlation with class

corr_cols = (
    'specific_gravity',
    'hemoglobin',
    'packed_cell_volume',
    'blood_glucose_random',
    'blood_urea',
    'red_blood_cells',
    'pus_cell',
    'sodium',
    'red_blood_cell_count'
    'pedal_edema'
    'anemia')

for feature in corr_cols:
    violin_plot(feature)
    kde_plot(feature)
    for feature2 in corr_cols:
        scatter_plot(feature, feature2)

