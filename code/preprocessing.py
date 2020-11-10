from sklearn.base import BaseEstimator, TransformerMixin

class AddTitles(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.name = "Name"
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        titles = []
        for i in X[self.name].values:
            last, first = i.split(",")
            title, *name = first.split(".")
            if title not in [" Mr", " Miss", " Mrs", " Master"]:
                titles.append("Other")
            else:
                titles.append(title)
#         original = X.columns
        X = X.assign(Name=titles)
        X.rename(columns={"Name": "Title"}, inplace=True)
        return X


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = "Embarked"
        
    def fit(self, X, y=None):
        self.most_frequent = X[self.col].value_counts().index[0]
        return self
    
    def transform(self, X, y=None):
        X[self.col] = X[self.col].fillna(self.most_frequent)
        return X


class MedianImputer_age(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.groupby_col = "Title"
        self.fill_col = "Age"
        self.median_val = []
        
    def fit(self, X, y=None):
        self.median_val = X[[self.groupby_col, self.fill_col]].groupby([self.groupby_col]).median()
        return self
    
    def transform(self, X, y=None):
        for index, i in self.median_val.iterrows():
            X[X[self.groupby_col] == index] = X[X[self.groupby_col] == index].fillna({self.fill_col: i[0]})
        return X


class AddFamily(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["Family"] = X["Parch"] + X["SibSp"]
        X = X.drop(["Parch", "SibSp"], axis=1)
        X["Family"] = X["Family"].replace([4, 5, 6, 7, 10], 3)
        return X


class AgeBracket(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for index, age in X["Age"].items():
            if 0 <= age <= 5:
                X.loc[index, "Age"] = "0-5"
            elif 5 < age <= 10:
                X.loc[index, "Age"] = "5-10"
            elif 10 < age <=15:
                X.loc[index, "Age"] = "10-15"
            elif 15 < age <=20:
                X.loc[index, "Age"] = "15-20"
            elif 20 < age <= 30:
                X.loc[index, "Age"] = "20-30"
            elif 30 < age <= 40:
                X.loc[index, "Age"] = "30-40"
            elif 40 < age <= 60:
                X.loc[index, "Age"] = "40-60"
            else:
                X.loc[index, "Age"] = "60+"
        return X


class MedianImputer_Fare(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.groupby_col = "Pclass"
        self.fill_col = "Fare"
        self.median_val = []
        
    def fit(self, X, y=None):
        self.median_val = X[[self.groupby_col, self.fill_col]].groupby([self.groupby_col]).median()
        return self
    
    def transform(self, X, y=None):
        for index, i in self.median_val.iterrows():
            X[X[self.groupby_col] == index] = X[X[self.groupby_col] == index].fillna({self.fill_col: i[0]})
        return X


class DropCol(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col_ = "Pclass"
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.drop([self.col_], axis=1)