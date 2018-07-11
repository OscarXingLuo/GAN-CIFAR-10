import tensorflow as tf
import numpy as np
import matplotlib as plt

class Network():

	def D_Model(self, filters, kernel_size ):
		self.model = tf.keras.Sequential()

		self.model.add(tf.keras.layers.Dense(input_size=(?,3072))) #input layer
		self.model.add(tf.keras.layers.Conv2D(filters['first_layer'], kernel_size['first_layer'])) #Conv layer
		self.model.add(tf.keras.layers.Conv2D(filters['second_layer'], kernel_size['second_layer'])) #Conv layer
		self.model.add(tf.keras.layers.Conv2D(filters['third_layer'], kernel_size['third_layer'])) #Conv layer
		self.model.add(tf.keras.layers.MaxPooling2D()) #Pool
		self.model.add(tf.keras.layers.Flatten()) #flatten convolutions 
		self.model.add(tf.keras.layers.Dense(10)) #output layer

	def G_Model(self):

		self.model = tf.keras.Sequential()

		self.model.add(tf.keras.layers.Dense(input_size=1))
		self.model.add(tf.keras.layers.Dense())

	def compile_model(self, compile_settings: dict):
        self.model.compile(
                optimizer=compile_settings['optimizer'],
                loss=compile_settings['loss'],
                metrics=compile_settings['metrics']
                )
    
    def fit(self, X_train, y_train, fit_settings: dict):
        
        self.model.fit(
                x = X_train,
                y = y_train,
                validation_data = fit_settings['validation_data'],
                batch_size = fit_settings['batch_size'],
                epochs = fit_settings['epochs'],
                verbose = fit_settings['verbose']
            )
        
    def predict(self, X, predict_settings: dict):
        y = self.model.predict(X)
        return y