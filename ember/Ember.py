import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.base import TransformerMixin
from sklearn import preprocessing

class Ember(TransformerMixin):

    def __init__(self, categorical_columns=[], embedding_output_targets=[], embedding_size = 10, loss='mean_squared_error', epochs=200):

        early_stopping_callback = EarlyStopping(monitor='loss', min_delta=1E-6, patience=10, verbose=0, mode='auto')

        cache_dir = os.path.join(os.getcwd(), '.cache', 'ember_models')

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.callbacks = [early_stopping_callback]

        self.embedding_size = embedding_size
        self.loss = loss
        self.epochs = epochs

        self.categorical_columns = categorical_columns
        self.output_targets = embedding_output_targets

        self.encodings = {}
        self.models = {}
        self.embeddings = {}


    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        if y is not None and self.output_targets is not None:
            if y.name in self.output_targets:
                self.output_targets.remove(y.name)

        for categorical_column in self.categorical_columns:
            self._encode(X[categorical_column], categorical_column)

        self._train_dnns(X, y)

        return self    

    def transform(self, X, *_):
        assert isinstance(X, pd.DataFrame)

        for column_name in self.categorical_columns:
            x_encoded = X[column_name].apply(lambda x: self.encodings[column_name][x])
                        
            weight_matrix = x_encoded.apply(lambda x: self.embeddings[column_name][0][x]).values

            embedding_dim = self.embeddings[column_name][0].shape[1]

            weight_feature_names = [''.join([column_name, '_ember_weight_', str(_idx)]) for _idx in range(0, embedding_dim)]

            #df = pd.DataFrame(np.row_stack(weight_matrix))
            #df.columns = weight_feature_names
            

            X[weight_feature_names] = pd.DataFrame(np.row_stack(weight_matrix), index=X.index)

        X.drop(columns=self.categorical_columns, inplace=True, errors='ignore')

        return X

    def _encode(self, x, column_name):

        if len(pd.unique(x)) < 0:
            raise Exception("Categorical variable must have more than one unique value (cardnality > 1)") 

        _classes = {}
        for value_idx, value in enumerate(pd.unique(x)):
            _classes[value] = value_idx

        self.encodings[column_name] = _classes

    def _train_dnns(self, X, y):

        for column_name, _classes in self.encodings.items():

            self.models[column_name] = self._build_model(len(_classes), (len(self.output_targets) + 0 if y is None else 1), self.embedding_size)

            x_train, y_train = self._get_training_data(X, y, column_name)

            self.models[column_name].fit(x=x_train, 
                                         y=y_train, 
                                         epochs = self.epochs, 
                                         batch_size = 32,
                                         callbacks = self.callbacks,
                                         verbose=None)    



            self.models[column_name].save(os.path.join(os.getcwd(), '.cache', 'ember_models', column_name + '_model.h5'))

            self.embeddings[column_name] = self.models[column_name].get_layer('embedding').get_weights()

            # clear our tf graph
            tf.keras.backend.clear_session()

        

    def _build_model(self, input_dim, output_length, embedding_size):
        # TODO use auto-keras here for the output layer, must be able to gurantee embedding input layer and label
        model = Sequential()
        model.add(Embedding(input_dim = input_dim, output_dim = embedding_size, input_length = 1, name="embedding"))
        model.add(Dense(units=40, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=10, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=output_length, name="targets"))

        model.compile(loss=self.loss, optimizer = "adam")

        return model       


    def _get_training_data(self, X, y, column_name_to_encode):

        x_encoded = X[column_name_to_encode].apply(lambda x: self.encodings[column_name_to_encode][x])

        x_train = np.reshape(x_encoded.values, (-1, 1))    

        if not self.output_targets:
            y_train = np.reshape(y.values, (-1,1))
        else:
            y_train = np.reshape(pd.concat(X[self.output_targets], y).values, (-1, len(self.output_targets)+1))

        return x_train, y_train


        