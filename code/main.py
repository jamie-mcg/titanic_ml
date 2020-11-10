import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from preprocessing import *

if __name__ == "__main__":
    print("\n=============================\n")
    print("TITANIC ML STARTER PROBLEM")

    # First we read in the training data
    train_data = pd.read_csv("./data/train.csv")

    show_data = input("\nShow an example of this data? (y/n) ")
    
    if show_data == "y" or show_data == "yes":
        print("\n=============================\n")
        print("Showing the top 5 data inputs ... \n")
        print(train_data.head())

    num_attribs = ["Fare", "Pclass"]
    cat_attribs = ["Sex", "Pclass", "Name", "Embarked", "Age", "Parch", "SibSp"]

    cat_pipeline = Pipeline([
        ('add_titles', AddTitles()),
        ('freq_imputer', MostFrequentImputer()),
        ('med_imputer', MedianImputer_age()),
        ('add_fam', AddFamily()),
        ('age_bras', AgeBracket()),
        ('one_hot', OneHotEncoder()),
    ])

    num_pipeline = Pipeline([
        ('med_imputer', MedianImputer_Fare()),
        ('drop_class', DropCol()),
        ('scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    train_data_copy = train_data.copy()

    X_train_full = train_data_copy.drop("Survived", axis=1)
    y_train_full = train_data_copy["Survived"]

    forest = RandomForestClassifier(random_state=42, n_estimators=10)

    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [None, 30, 40],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [10, 20, 100]
    }

    rand_search = RandomizedSearchCV(estimator=forest, n_iter=40, cv=3, param_distributions=param_grid)

    X_train_full_prepared = full_pipeline.fit_transform(X_train_full)

    print("\n=============================\n")
    print("Running Randomized Search for hyperparameters ...")

    rand_search.fit(X_train_full_prepared, y_train_full)

    print("Found best hyperparameters as: \n")
    print(rand_search.best_params_)
    print(f"\nWith a best classification accuracy  of {rand_search.best_score_*100:.2f}%")
    print("\n=============================\n")

    rf_best = RandomForestClassifier(**rand_search.best_params_, random_state=42)

    X_test_full = pd.read_csv("./data/test.csv")
    y_test_full = pd.read_csv("./data/test_truth.csv")["Survived"]

    X_test_full_prepared = full_pipeline.transform(X_test_full)

    print("\n=============================\n")
    print("Begining training with best hyperparameters ...")

    rf_best.fit(X_train_full_prepared, y_train_full)

    print("\nFinished training!")
    print("\n=============================\n")
    print("Results:")
    print(f"Training acuuracy: {rf_best.score(X_train_full_prepared, y_train_full)*100:.2f}%")
    print(f"Test accuracy: {rf_best.score(X_test_full_prepared, y_test_full)*100:.2f}%")
    print("\n=============================\n")

    

