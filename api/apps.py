from django.apps import AppConfig
from django.conf import settings
import joblib
import os
import pandas as pd

DATA_FILE = os.path.join(settings.DATA, "SQLiV3.csv")
#load the dataset

df = pd.read_csv(DATA_FILE)
#delete unnecessary features
del df["Unnamed: 2"]
del df["Unnamed: 3"]

#drop nan in the dataset
df = df.dropna()

# Dropping the label values, that are different from 0 or 1.
df = df.drop(df[(df['Label'] != '0') & (df['Label'] != '1')].index)


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "my_log.pkl")
    CLEANED_DATA = df
    model = joblib.load(MODEL_FILE)
