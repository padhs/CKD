

data = data.ArffToCSV()
data.load_arff_file('./datasets/chronic_kidney_disease_full.arff')

data.set_column_data_types({'age'})

df = df.convert_dtypes()
print(df.info())

for cols in df:
    if df[cols].dtype == 'object':
        df[cols] = pd.to_numeric(df[cols], errors='coerce')
    else:
        df[cols] = df[cols]

print(df.info())
print(df.head())


df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['blood_pressure'] = pd.to_numeric(df['blood_pressure'], errors='coerce')
df['specific_gravity'] = pd.to_numeric(df['specific_gravity'], errors='coerce')
df['albumin'] = pd.to_numeric(df['albumin'], errors='coerce')
df['sugar'] = pd.to_numeric(df['sugar'], errors='coerce')
df['blood_glucose_random'] = pd.to_numeric(df['blood_glucose_random'], errors='coerce')
df['blood_urea'] = pd.to_numeric(df['blood_urea'], errors='coerce')
df['serum_creatinine'] = pd.to_numeric(df['serum_creatinine'], errors='coerce')
df['sodium'] = pd.to_numeric(df['sodium'], errors='coerce')
df['potassium'] = pd.to_numeric(df['potassium'], errors='coerce')
df['hemoglobin'] = pd.to_numeric(df['hemoglobin'], errors='coerce')
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')






for column in numerical_columns:
    random_value_imputation(column)

# random value imputation for categorical columns because they have high null counts
for column in categorical_columns:
    if column != 'red_blood_cells' and column != 'pus_cells':
        impute_mode(column)
    else:
        random_value_imputation(column)


for column in numerical_columns:
    df[column] = mean_imputer.fit_transform(df[column])

for column in categorical_columns:
    if column != 'red_blood_cells' and column != 'pus_cells':
        [df[column]] = mode_imputer.fit_transform([df[column]])
    else:
        [df[column]] = mean_imputer.fit_transform([df[column]])






