import kagglehub
import pandas as pd

# 1. Baixar o dataset
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")

print("Path to dataset files:", path)

# 2. Carregar o CSV (ajustar o nome se necessário)
csv_path = f"{path}/heart_cleveland_upload.csv" 
df = pd.read_csv(csv_path)

print(df.head())  
print(df.info())     
print(df.describe())

print(df['condition'].value_counts())
print(df.head(10))


