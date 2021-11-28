from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

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
        model = tf.keras.models.load_model('my_custom_sklearn_transforms/deep_learning_model.h5py')
        data = model.predict(data)
        return data