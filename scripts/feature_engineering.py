import pandas as pd
import os
from src.feature_engineering import FeatureCreation
from src.preprocessing import Preprocessing

df = pd.read_csv('data/preprocessed/preprocessed_airplane_price_dataset.csv')

# feature creation
feature_creation = FeatureCreation(df)
df = (
    feature_creation.create_Company()
      .create_HMC_per_person()
      .create_Cost_per_km()
      .change_Age()
      .getDataset()
)

# encoding & scaling & transformation
preprocessing = Preprocessing(df)
df = (
    preprocessing.encode()
    .scale()
    .logTransformation()
    .getDataset()
)


# save dataset
output_folder = 'data/engineered'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'engineered_airplane_price_dataset.csv')
df.to_csv(output_path, index=False)

print(df.head(10))