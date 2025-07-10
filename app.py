import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Baixar o dataset
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")

print("Path to dataset files:", path)

# 2. Carregar o CSV (ajustar o nome se necessário)
csv_path = f"{path}/heart_cleveland_upload.csv" 
df = pd.read_csv(csv_path)

print(df.head())  
print(df.info())     
print(df.describe())

#1 passo: Separar entradas e saidas de dados com base no nosso csv.
X = df.drop("condition", axis=1)
y = df["condition"]

#2 passo: criar dados de treino e teste (test_size significa a porcentagem que sera dos dados de teste e o restante de treino. )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))

print(df['condition'].value_counts())

