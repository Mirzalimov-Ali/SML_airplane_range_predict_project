import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class Preprocessing:
    def __init__(self, df):
        self.df = df.copy()
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
    
    def fillMissingValues(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
        return self
    
    def encode(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                else:
                    self.df[col] = self.encoder.fit_transform(self.df[col])
        return self
    
    def scale(self):
        num_col = self.df.select_dtypes(include=['float64', 'int64']).columns.drop('Range_(km)')
        self.df[num_col] = self.scaler.fit_transform(self.df[num_col])
        return self
    
    def logTransformation(self):
        skewness = self.df.skew()
        features_log = skewness[skewness >= 0.5].index.tolist()
        
        for col in features_log:
            if (self.df[col] > 0).all():
                self.df[col] = np.log1p(self.df[col])
        return self
    
    def getDataset(self):
        return self.df