import warnings
import inline

warnings.filterwarnings("ignore")
# because python tells us to standardize data first, but we don't need to for these models

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import plotly.express as px

plt.style.use('fivethirtyeight')

df = pd.read_csv('./datasets/kidney_disease_ready_round.csv')

# information about the dataset
print(df.head())
print(df.info())

# Model building-->
x_independents = df.drop(columns=['class'])
y_dependents = df['class']

# splitting the dataset into training and test sets-->
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
    x_independents, y_dependents, test_size=0.3, random_state=42)


# Logistic Regression -->
lr = LogisticRegression()
lr.fit(x_data_train, y_data_train)
LR_training_accuracy = accuracy_score(y_data_train, lr.predict(x_data_train))*100
LR_test_accuracy = accuracy_score(y_data_test, lr.predict(x_data_test))*100
LR_confusion_matrix = confusion_matrix(y_data_test, lr.predict(x_data_test))
LR_classification_report = classification_report(y_data_test, lr.predict(x_data_test))
print(f"LR_training_accuracy: {LR_training_accuracy}")
print(f"LR_test_accuracy: {LR_test_accuracy}")
print(f"LR_confusion_matrix:\n{LR_confusion_matrix}")
print(f"LR_classification_report:\n{LR_classification_report}")


# KNN -->
knn = KNeighborsClassifier()
knn.fit(x_data_train, y_data_train)
KNN_training_accuracy = accuracy_score(y_data_train, knn.predict(x_data_train))*100
KNN_test_accuracy = accuracy_score(y_data_test, knn.predict(x_data_test))*100
KNN_confusion_matrix = confusion_matrix(y_data_test, knn.predict(x_data_test))
KNN_classification_report = classification_report(y_data_test, knn.predict(x_data_test))
print(f"KNN_training_accuracy: {KNN_training_accuracy}")
print(f"KNN_test_accuracy: {KNN_test_accuracy}")
print(f"KNN_confusion_matrix:\n{KNN_confusion_matrix}")
print(f"KNN_classification_report:\n{KNN_classification_report}")


# Decision Tree Classifier -->
dtc = DecisionTreeClassifier()
dtc.fit(x_data_train, y_data_train)
DTC_training_accuracy = accuracy_score(y_data_train, dtc.predict(x_data_train))*100
DTC_test_accuracy = accuracy_score(y_data_test, dtc.predict(x_data_test))*100
DTC_confusion_matrix = confusion_matrix(y_data_test, dtc.predict(x_data_test))
DTC_classification_report = classification_report(y_data_test, dtc.predict(x_data_test))
print(f"DTC_training_accuracy: {DTC_training_accuracy}")
print(f"DTC_test_accuracy: {DTC_test_accuracy}")
print(f"DTC_confusion_matrix:\n{DTC_confusion_matrix}")
print(f"DTC_classification_report:\n{DTC_classification_report}")

# hyperparameter tuning of DTC
grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'splitter': ['best', 'random'],
    'min_samples_leaf': [1, 2, 3, 5, 7],
    'min_samples_split': [1, 2, 3, 5, 7],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search_dtc = GridSearchCV(dtc, grid_param, cv=5, scoring=None, n_jobs=-1, verbose=1)
grid_search_dtc.fit(x_data_train, y_data_train)

# best parameters and best score
print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_score_)

# best estimator
dtc = grid_search_dtc.best_estimator_
print('New Scores:\n')
print(f"DTC_training_accuracy: {DTC_training_accuracy}")
print(f"DTC_test_accuracy: {DTC_test_accuracy}")
print(f"DTC_confusion_matrix:\n{DTC_confusion_matrix}")
print(f"DTC_classification_report:\n{DTC_classification_report}")


# Random Forest Classifier -->
rfc = RandomForestClassifier(
    criterion='entropy',
    max_depth=11,
    max_features=0.5,
    min_samples_leaf=2,
    min_samples_split=3,
    n_estimators=130)
rfc.fit(x_data_train, y_data_train)
RFC_training_accuracy = accuracy_score(y_data_train, rfc.predict(x_data_train))*100
RFC_test_accuracy = accuracy_score(y_data_test, rfc.predict(x_data_test))*100
RFC_confusion_matrix = confusion_matrix(y_data_test, rfc.predict(x_data_test))
RFC_classification_report = classification_report(y_data_test, rfc.predict(x_data_test))
print(f"RFC_training_accuracy: {RFC_training_accuracy}")
print(f"RFC_test_accuracy: {RFC_test_accuracy}")
print(f"RFC_confusion_matrix:\n{RFC_confusion_matrix}")
print(f"RFC_classification_report:\n{RFC_classification_report}")


# Ada Boost Classifier
ada = AdaBoostClassifier(base_estimator=dtc)
ada.fit(x_data_train, y_data_train)
ADA_training_accuracy = accuracy_score(y_data_train, ada.predict(x_data_train))*100
ADA_test_accuracy = accuracy_score(y_data_test, ada.predict(x_data_test))*100
ADA_confusion_matrix = confusion_matrix(y_data_test, ada.predict(x_data_test))
ADA_classification_report = classification_report(y_data_test, ada.predict(x_data_test))
print(f"ADA_training_accuracy: {ADA_training_accuracy}")
print(f"ADA_test_accuracy: {ADA_test_accuracy}")
print(f"ADA_confusion_matrix:\n{ADA_confusion_matrix}")
print(f"ADA_classification_report:\n{ADA_classification_report}")


# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(x_data_train, y_data_train)
GBC_training_accuracy = accuracy_score(y_data_train, gbc.predict(x_data_train))*100
GBC_test_accuracy = accuracy_score(y_data_test, gbc.predict(x_data_test))*100
GBC_confusion_matrix = confusion_matrix(y_data_test, gbc.predict(x_data_test))
GBC_classification_report = classification_report(y_data_test, gbc.predict(x_data_test))
print(f"GBC_training_accuracy: {GBC_training_accuracy}")
print(f"GBC_test_accuracy: {GBC_test_accuracy}")
print(f"GBC_confusion_matrix:\n{GBC_confusion_matrix}")
print(f"GBC_classification_report:\n{GBC_classification_report}")


# Stochastic Gradient Boosting(SGB)
sgb = GradientBoostingClassifier(
    max_depth=4,
    subsample=0.90,
    max_features=0.75,
    n_estimators=200
)
sgb.fit(x_data_train, y_data_train)
SGB_training_accuracy = accuracy_score(y_data_train, sgb.predict(x_data_train))*100
SGB_test_accuracy = accuracy_score(y_data_test, sgb.predict(x_data_test))*100
SGB_confusion_matrix = confusion_matrix(y_data_test, sgb.predict(x_data_test))
SGB_classification_report = classification_report(y_data_test, sgb.predict(x_data_test))
print(f"SGB_training_accuracy: {SGB_training_accuracy}")
print(f"SGB_test_accuracy: {SGB_test_accuracy}")
print(f"SGB_confusion_matrix:\n{SGB_confusion_matrix}")
print(f"SGB_classification_report:\n{SGB_classification_report}")


# xgboost -->
xgb = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.5,
    max_depth=5,
    n_estimators=150
)
xgb.fit(x_data_train, y_data_train)
XGB_training_accuracy = accuracy_score(y_data_train, xgb.predict(x_data_train))*100
XGB_test_accuracy = accuracy_score(y_data_test, xgb.predict(x_data_test))*100
XGB_confusion_matrix = confusion_matrix(y_data_test, xgb.predict(x_data_test))
XGB_classification_report = classification_report(y_data_test, xgb.predict(x_data_test))
print(f"XGB_training_accuracy: {XGB_training_accuracy}")
print(f"XGB_test_accuracy: {XGB_test_accuracy}")
print(f"XGB_confusion_matrix:\n{XGB_confusion_matrix}")
print(f"XGB_classification_report:\n{XGB_classification_report}")


# cat boost classifier -->
cat = CatBoostClassifier(
    iterations=10,
    learning_rate=0.5)
cat.fit(x_data_train, y_data_train)
CAT_training_accuracy = accuracy_score(y_data_train, cat.predict(x_data_train))*100
CAT_test_accuracy = accuracy_score(y_data_test, cat.predict(x_data_test))*100
CAT_confusion_matrix = confusion_matrix(y_data_test, cat.predict(x_data_test))
CAT_classification_report = classification_report(y_data_test, cat.predict(x_data_test))
print(f"CAT_training_accuracy: {CAT_training_accuracy}")
print(f"CAT_test_accuracy: {CAT_test_accuracy}")
print(f"CAT_confusion_matrix:\n{CAT_confusion_matrix}")
print(f"CAT_classification_report:\n{CAT_classification_report}")


# Extra Trees Classifier
etc = ExtraTreesClassifier()
etc.fit(x_data_train, y_data_train)
ETC_training_accuracy = accuracy_score(y_data_train, etc.predict(x_data_train))*100
ETC_test_accuracy = accuracy_score(y_data_test, etc.predict(x_data_test))*100
ETC_confusion_matrix = confusion_matrix(y_data_test, etc.predict(x_data_test))
ETC_classification_report = classification_report(y_data_test, etc.predict(x_data_test))
print(f"ETC_training_accuracy: {ETC_training_accuracy}")
print(f"ETC_test_accuracy: {ETC_test_accuracy}")
print(f"ETC_confusion_matrix:\n{ETC_confusion_matrix}")
print(f"ETC_classification_report:\n{ETC_classification_report}")


# lightbgm classifier--->
lgbm = LGBMClassifier(
    learning_rate=1
)
lgbm.fit(x_data_train, y_data_train)
lgbm_training_accuracy = accuracy_score(y_data_train, lgbm.predict(x_data_train))*100
lgbm_test_accuracy = accuracy_score(y_data_test, lgbm.predict(x_data_test))*100
lgbm_confusion_matrix = confusion_matrix(y_data_test, lgbm.predict(x_data_test))
lgbm_classificaiton_report = classification_report(y_data_test, lgbm.predict(x_data_test))
print(f"lgbm_training_accuracy: {lgbm_training_accuracy}")
print(f"lgbm_test_accuracy: {lgbm_test_accuracy}")
print(f"lgbm_confusion_matrix:\n{lgbm_confusion_matrix}")
print(f"lgbm_classification_report:\n{lgbm_classificaiton_report}")


# Models comparison -->

models = pd.DataFrame({
    'Model': ['Logistic Regression',
              'KNN',
              'Decision Tree Classifier',
              'Random Forest Classifer',
              'Ada Boost Classifier',
              'Gradient Boost Classifier',
              'Stochastic Gradient Boosting',
              'Xgboost',
              'Cat Boost',
              'Extra Trees Classifier',
              'LightBGM Classifier'],
    'Training_accuracy': [LR_training_accuracy,
                          KNN_training_accuracy,
                          DTC_training_accuracy,
                          RFC_training_accuracy,
                          ADA_training_accuracy,
                          GBC_training_accuracy,
                          SGB_training_accuracy,
                          XGB_training_accuracy,
                          CAT_training_accuracy,
                          ETC_training_accuracy,
                          lgbm_training_accuracy],
    'Test_accuracy': [LR_test_accuracy,
                      KNN_test_accuracy,
                      DTC_test_accuracy,
                      RFC_test_accuracy,
                      ADA_test_accuracy,
                      GBC_test_accuracy,
                      SGB_test_accuracy,
                      XGB_test_accuracy,
                      CAT_test_accuracy,
                      ETC_test_accuracy,
                      lgbm_test_accuracy]
})
models.sort_values(by='Test_accuracy', ascending=False)
print(models.describe())

models.to_csv('./datasets/model_score.csv', header=True, index=False)
