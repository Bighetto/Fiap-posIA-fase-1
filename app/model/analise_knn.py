import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Lê o CSV
df = pd.read_csv("C:/Users/devpy/PycharmProjects/Fiap-posIA-fase-1/heart_cleveland_upload.csv")
# Separa features (X) e alvo (y)

# Altere 'alvo' para o nome da sua coluna alvo
X = df.drop(columns=["condition"])
y = df["condition"]

# Divide entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria e treina o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Faz predições
y_pred = knn.predict(X_test)

# Avalia acurácia
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.2f}")