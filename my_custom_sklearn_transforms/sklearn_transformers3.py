from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf

class Model(BaseEstimator, RegressorMixin):
    def __init__(self, ruta_modelo):
        self.ruta_modelo = ruta_modelo
    
    def fit(self, X, y=None):
        return self

    def predict(self, X):      
        data = X.copy()
        model = tf.keras.models.load_model(self.ruta_modelo)
        data = model.predict(data)
        return data
