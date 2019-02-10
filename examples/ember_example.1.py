
#%%
import pandas as pd
import numpy as np
import os
import sys

#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append('/home/washcycle/Development.local/embedding_utils')
from ember import Ember

#%%
df = pd.read_csv("nyc-east-river-bicycle-crossings.zip")
df["Date"] = pd.to_datetime(df['Date'])
df["day_of_week"] = df["Date"].dt.day_name()


#%%
df.head()

#%%
emb = Ember(['day_of_week'], ['Total', 'Brooklyn Bridge', 'Manhattan Bridge'])
emb.fit(df)
df_new = emb.transform(df)

#%%
df_new.head()

#%%
assert 'day_of_week' not in df_new.columns

#%%
x_train = df_new.drop(columns=['Total', 'Precipitation', 'Date', 'Day', 'Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge'])
y_train = df_new['Total']

#%%
from sklearn import preprocessing
from sklearn.linear_model import Ridge
reg = Ridge(alpha=100)
reg.fit(x_train, y_train)

#%%
reg.coef_

#%%
from sklearn.metrics import r2_score, mean_squared_error
y_pred = reg.predict(x_train)

print(r2_score(y_train, y_pred))
print(mean_squared_error(y_train, y_pred))

#%%
import yellowbrick
res = y_train - y_pred

#%%
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(reg)
visualizer.score(x_train, y_train)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data



#%%
