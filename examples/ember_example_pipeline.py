
#%%
import pandas as pd
import numpy as np
import sys

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm
from sklearn.preprocessing import FunctionTransformer


sys.path.append('/home/washcycle/Development.local/embedding_utils')
sys.path.append('/home/washcycle/Development.local/embedding_utils/tests')
from ember import Ember

import os

print(os.getcwdb())

#%%
df = pd.read_csv("tests/nyc-east-river-bicycle-crossings.zip", index_col=0)
df["Date"] = pd.to_datetime(df['Date'])
df["day_of_week"] = df["Date"].dt.day_name()


#%%
df.head()

#%%
df['Precipitation'] = pd.to_numeric(df['Precipitation'].str.replace('[^0-9|^.]','').str.strip(), errors='coerce').fillna(0)

#%%
y = df['Total']

#%% Function Transformers
def drop_features(X):
    print("here")
    return X.drop(columns=['Total', 'Precipitation', 'Date', 'Day'])

#%%

df.drop(columns=[['Total', 'Precipitation', 'Date', 'Day']], inplace=True, errors='ignore')

#%%
def drop_features(X):
    print("here")
    return X.drop(columns=['Total', 'Precipitation', 'Date', 'Day'])

#%%
clf = svm.SVR(kernel='linear')
emb = Ember(categorical_columns=['day_of_week'], embedding_output_targets=['Total'])
ember_svm = Pipeline([('drop_features', FunctionTransformer(drop_features, validate=False)), ('ember', emb), ('svc', clf)])

ember_svm.set_params(svc__C=.1).fit(df, y)


#%%
from sklearn.metrics import r2_score, mean_squared_error
y_pred = ember_svm.predict(df)

print(r2_score(y, y_pred))
print(mean_squared_error(y, y_pred))

#%%
import yellowbrick
res = y - y_pred

#%%
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(ember_svm)
visualizer.score(df, y)  # Evaluate the model on the test data
visualizer.poof()    

#%%
