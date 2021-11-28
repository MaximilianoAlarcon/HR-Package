from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Transform_data(BaseEstimator, TransformerMixin):
    def __init__(self, min_capa_verde_transformer, max_capa_verde_transformer, mean_capa_verde_transformer, std_capa_verde_transformer):
        self.min_capa_verde_transformer = min_capa_verde_transformer
        self.max_capa_verde_transformer = max_capa_verde_transformer
        self.mean_capa_verde_transformer = mean_capa_verde_transformer
        self.std_capa_verde_transformer = std_capa_verde_transformer
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primero realizamos la cópia del DataFrame 'X' de entrada
        
        
        def powertransformer(dataframe,col,transformers):
            c = dataframe[col].copy()
            c = np.array(c).reshape(-1,1)
            c = transformers["paso1"].transform(c)
            c = transformers["paso2"].transform(c)
            c = transformers["paso3"].transform(c)
            c = np.float32(c)
            dataframe[col] = c
            
        def powertransformer_sqrt(dataframe,col,transformers):
            c = dataframe[col].copy()
            c = np.sqrt(c)
            c = np.array(c).reshape(-1,1)
            c = transformers["paso1"].transform(c)
            c = transformers["paso2"].transform(c)
            c = transformers["paso3"].transform(c)
            c = np.float32(c)
            dataframe[col] = c
            
        def powertransformer_log(dataframe,col,transformers):
            c = dataframe[col].copy()
            c = np.log(c)
            c = np.float32(c)
            c = np.array(c).reshape(-1,1)
            c = transformers["paso1"].transform(c)
            c = transformers["paso2"].transform(c)
            c = transformers["paso3"].transform(c)
            c = np.float32(c)
            dataframe[col] = c        
        
        
        data = X.copy()
        powertransformer(data,"min_capa_verde",self.min_capa_verde_transformer)
        powertransformer_log(data,"max_capa_verde",self.max_capa_verde_transformer)
        powertransformer(data,"mean_capa_verde",self.mean_capa_verde_transformer)
        powertransformer_log(data,"std_capa_verde",self.std_capa_verde_transformer)
        return data
