from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

class Model(BaseEstimator, RegressorMixin):
    def __init__(self, ruta_modelo):
        self.ruta_modelo = ruta_modelo
    
    def fit(self, X, y=None):
        return self

    def predict(self, X):      
        data = X.copy()
        #model = tf.keras.models.load_model(self.ruta_modelo)
        model = Sequential()
        model.add(Dense(32,kernel_initializer='normal',input_dim = 70,activation="relu"))
        model.add(Dense(1,activation = "linear")) 
        model.compile(optimizer = "adam",loss="mse",metrics=['mse'])
        model.load_weights(self.ruta_modelo)
        data = model.predict(data)
        return data
