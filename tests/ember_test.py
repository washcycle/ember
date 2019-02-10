from ember import Ember
import pandas as pd

# dummy data set for testing
df = pd.read_csv("tests/nyc-east-river-bicycle-crossings.zip")
df["Date"] = pd.to_datetime(df['Date'])
df["day_of_week"] = df["Date"].dt.day_name()

def test_encoder():
    em = Ember(['col1', 'col2'], ['col3'])
    em._encode(['a','a','b','c','a'], 'group')

    # check that the encoding get correctly enumerated and sized
    assert 'group' in em.encodings.keys()
    assert len(em.encodings['group'].keys()) == 3

def test_dnn():
    em = Ember(['day_of_week'], ['Total'])
    em.fit(df)

    # check that at least 1 epoch was trained
    assert len(em.models['day_of_week'].history.epoch) > 0

def test_fit():
    em = Ember(['day_of_week'], ['Total'])
    em.fit(df)
    df_new = em.transform(df)

    # Check if generated feature names are in the new df
    assert set([''.join(['day_of_week', '_ember_weight_', str(_idx)]) for _idx in range(0, 10)]).issubset(set(df_new.columns))
    assert 'day_of_week' not in df_new.columns
   