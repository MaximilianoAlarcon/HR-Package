from sklearn.base import BaseEstimator, TransformerMixin

class Poly_data(BaseEstimator, TransformerMixin):
    def __init__(self, polynomialfeatures_transformer, standardscaler_poly_data):
        self.polynomialfeatures_transformer = polynomialfeatures_transformer
        self.standardscaler_poly_data = standardscaler_poly_data
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):      
        data = X.copy()
        data = self.polynomialfeatures_transformer.transform(data)
        data = self.standardscaler_poly_data.transform(data)
        return data
