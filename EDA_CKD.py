import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import scipy.stats
import plotly.io as pio

import matplotlib.style as style


style.use('fivethirtyeight')

df = pd.read_csv('./datasets/kidney_disease_ready_round.csv')

# Exploratory Data Analysis -->
print(df.head())
print(df.info())
print(df.isnull().sum())

# Feature distributions -->
plt.figure(figsize=(20, 15))
plot_number = 1

for column in df.columns:
    if plot_number <= 14:
        ax = plt.subplot(3, 5, plot_number)
        sns.distplot(df[column])
        plt.xlabel(column)

    plot_number += 1

plt.tight_layout()
plt.savefig('./datasets/figures/distribution_plot.png')
plt.show()

# heatmaps of features-->
plt.figure(figsize=(28, 20), dpi=120)
sns.heatmap(df.corr(), annot=True, linewidths=5, linecolor='lightgrey',)
plt.tight_layout()
plt.savefig('./datasets/figures/heatmap_plot.png')
plt.show()


def violin_plot(col):
    fig = px.violin(df, y=col, x='class', box=True, template='plotly_dark')
    pio.write_image(fig, './datasets/fig_folder/violin_plot_' + col + '.png')
    return fig.show()


def kde_plot(col):
    grid = sns.FacetGrid(df, hue='class', aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    grid.savefig('./datasets/fig_folder/kde_plot_' + col + '.png')
    return plt.show()


def scatter_plot(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color='class', template='plotly_dark')
    pio.write_image(fig, './datasets/fig_folder/scatter_plot_' + col1 + '_' + col2 + '_' + '.png')
    return fig.show()


# From Pearson's correlation matrix (correlation hypothesis)--->
'''
    specific_gravity, 
    hemoglobin, 
    packed_cell_volume, 
    blood_glucose_random, 
    blood_urea, 
    red_blood_cells, 
    pus_cell, sodium, 
    red_blood_cell_count has positive correlation with class
'''


corr_cols = (
    'specific_gravity',
    'hemoglobin',
    'packed_cell_volume',
    'blood_glucose_random',
    'blood_urea',
    'red_blood_cells',
    'pus_cell',
    'sodium',
    'red_blood_cell_count',
    'pedal_edema',
    'anemia')

for feature in corr_cols:
    violin_plot(feature)
    kde_plot(feature)
    for feature2 in corr_cols:
        scatter_plot(feature, feature2)

# Correlating each feature with class-->

# age v class
plt.figure(figsize=(70, 25))
plt.legend(loc='upper left')
g = sns.countplot(data=df, x='age', hue='class')
g.legend(title='Chronic kidney disease patient?', loc='center left', bbox_to_anchor=(0.1, 0.5), ncol=1)
g.tick_params(labelsize=20)
plt.setp(g.get_legend().get_texts(), fontsize='32')
plt.setp(g.get_legend().get_title(), fontsize='42')
g.axes.set_title('Graph of age vs number of patients with chronic kidney disease', fontsize=50)
g.set_xlabel('Count', fontsize=40)
g.set_ylabel("Age", fontsize=40)
plt.savefig('./datasets/figures/age_vs_count_count_plot.png')
plt.show()


age_corr = ['age', 'class']
age_corr1 = df[age_corr]
age_corr_y = age_corr1[age_corr1['class'] == 1].groupby(['age']).size().reset_index(name='count')
age_corr_y.corr()

sns.regplot(data=age_corr_y, x='age', y='count').set_title(
    "Correlation graph for Age v chronic kidney disease patient", fontsize=12)
plt.savefig('./datasets/figures/age_vs_count_reg_plot.png')
plt.show()

# red_blood_cell v class-->
# (chi-square test):
cont_rbc = pd.crosstab(df['red_blood_cells'], df['class'])
cont_rbc_chi_test = scipy.stats.chi2_contingency(cont_rbc)
print(cont_rbc)
print(cont_rbc_chi_test)

# pus_cells v class-->
cont_pc = pd.crosstab(df['pus_cell'], df['class'])
cont_pc_chi_test = scipy.stats.chi2_contingency(cont_pc)
print(cont_pc)
print(cont_pc_chi_test)

# blood_glucose_random v class-->
bgr_corr = ['blood_glucose_random', 'class']
bgr_corr1 = df[bgr_corr]
bgr_corr1.blood_glucose_random = bgr_corr1.blood_glucose_random.round(-1)

bgr_corr_y = bgr_corr1[bgr_corr1['class'] == 1].groupby(['blood_glucose_random']).size().reset_index(name='count')
bgr_corr_y.corr()

sns.regplot(data=bgr_corr_y, x='blood_glucose_random', y='count').set_title(
    "Correlation graph for blood glucose vs chronic kidney disease patient", fontsize=12)
plt.savefig('./datasets/figures/bgr_vs_class_patient.png')
plt.show()

bgr_corr_n = bgr_corr1[bgr_corr1['class'] == 0].groupby(['blood_glucose_random']).size().reset_index(name='count')
bgr_corr_n.corr()

sns.regplot(data=bgr_corr_n, x='blood_glucose_random', y='count').set_title(
    "Correlation graph for blood glucose vs healthy patient", fontsize=12)
plt.savefig('./datasets/figures/bgr_vs_class_healthy.png')
plt.show()

# blood_urea v class-->
bu_corr = ['blood_urea', 'class']
bu_corr1 = df[bu_corr]
bu_corr1.blood_urea = df.blood_urea.round(-1)
bu_corr_y = bu_corr1[bu_corr1['class'] == 1].groupby(['blood_urea']).size().reset_index(name='count')
bu_corr_y.corr()

sns.regplot(data=bu_corr_y, x='blood_urea', y='count').set_title(
    'Correlation graph for blood urea vs CKD patient', fontsize=12)
plt.savefig('./datasets/figures/bu_vs_class_patient.png')
plt.show()

bu_corr_n = bu_corr1[bu_corr1['class'] == 0].groupby(['blood_urea']).size().reset_index(name='count')
bu_corr_n.corr()

sns.regplot(data=bu_corr_n, x='blood_urea', y='count').set_title(
    'Correlation graph for blood urea vs healthy patient', fontsize=12)
plt.savefig('./datasets/figures/bu_vs_class_healthy.png')
plt.show()

# sodium v class -->
sod_corr = ['sodium', 'class']
sod_corr1 = df[sod_corr]
sod_corr_y = sod_corr1[sod_corr1['class'] == 1].groupby(['sodium']).size().reset_index(name='count')
sod_corr_y.corr()

sns.regplot(data=sod_corr_y, x='sodium', y='count').set_title(
    'Correlation graph for blood sodium vs CKD patient', fontsize=12)
plt.savefig('./datasets/figures/sod_vs_class_patient.png')
plt.show()

sod_corr_n = sod_corr1[sod_corr1['class'] == 0].groupby(['sodium']).size().reset_index(name='count')
sod_corr_n.corr()

sns.regplot(data=sod_corr_n, x='sodium', y='count').set_title(
    'Correlation graph for blood sodium vs healthy patient', fontsize=12)
plt.savefig('./datasets/figures/sod_vs_class_healthy.png')
plt.show()

# pedal edema v class-->
cont_pedal_edema = pd.crosstab(df["pedal_edema"], df["class"])
cont_pedal_edema_pc_chi_test = scipy.stats.chi2_contingency(cont_pedal_edema)
print(cont_pedal_edema)
print(cont_pedal_edema_pc_chi_test)

# Anemia v class-->
cont_anemia = pd.crosstab(df["anemia"], df['class'])
cont_anemia_pc_chi_test = scipy.stats.chi2_contingency(cont_anemia)
print(cont_anemia)
print(cont_anemia_pc_chi_test)

# serum_creatinine v class-->
sc_corr = ['serum_creatinine', 'class']
sc_corr1 = df[sc_corr]
sc_corr1.serum_creatinine = sc_corr1.serum_creatinine.round(1)
sc_corr_y = sc_corr1[sc_corr1['class'] == 1].groupby(['serum_creatinine']).size().reset_index(name='count')
sc_corr_y.corr()

sns.regplot(data=sc_corr_y, x='serum_creatinine', y='count').set_title(
    'Correlation graph for serum creatinine vs CKD patient', fontsize=12)
plt.savefig('./datasets/figures/serum_creatinine_vs_class_patient.png')
plt.show()

sc_corr_n = sc_corr1[sc_corr1['class'] == 0].groupby(['serum_creatinine']).size().reset_index(name='count')
sc_corr_n.corr()

sns.regplot(data=sc_corr_n, x='serum_creatinine', y='count').set_title(
    'Correlation graph for serum creatinine vs CKD patient', fontsize=12)
plt.savefig('./datasets/figures/serum_creatinine_vs_class_healthy.png')
plt.show()

# diabetes v class-->
cont_diabetes = pd.crosstab(df["diabetes_mellitus"], df["class"])
cont_diabetes_chi2_test = scipy.stats.chi2_contingency(cont_diabetes)
print(cont_diabetes)
print(cont_diabetes_chi2_test)

# coronary_artery_disease v class-->
cont_cad = pd.crosstab(df["coronary_artery_disease"], df["class"])
cont_cad_chi2_test = scipy.stats.chi2_contingency(cont_cad)
print(cont_cad)
print(cont_cad_chi2_test)

# hypertension v class -->
cont_hypertension = pd.crosstab(df['hypertension'], df["class"])
cont_hypertension_chi2_test = scipy.stats.chi2_contingency(cont_hypertension)
print(cont_hypertension)
print(cont_hypertension_chi2_test)
