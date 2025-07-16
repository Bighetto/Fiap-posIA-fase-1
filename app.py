import pandas as pd
import kagglehub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split,cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Baixar e carregar os dados
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
csv_path = f"{path}/heart_cleveland_upload.csv"
df = pd.read_csv(csv_path)

# Contar quantos têm age < 40
count_menor_que_40 = df[df['age'] < 40].shape[0]
print(f"Quantidade de pessoas com idade menor que 40: {count_menor_que_40}")

# Contar quantos têm age > 70
count_maior_que_70 = df[df['age'] > 70].shape[0]
print(f"Quantidade de pessoas com idade maior que 70: {count_maior_que_70}")
# Removendo dados fracos
df = df[(df['age'] >= 40) & (df['age'] <= 70)]

# 2. Verificar dados
print(df.info())
print(df.isnull().sum())

# 3. Definir X (features) e y (target)
# X = df.drop('condition', axis=1)
X = df[['thal', 'ca', 'oldpeak', 'exang', 'cp']]
y = df['condition']

# 4. Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Treinar modelo de Regressão Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Fazer predições
y_pred = model.predict(X_test_scaled)

#7.1 Validacoes cruzadas
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print("\nF1 médio (cross-validation):", scores.mean())

# 8. Avaliação
print("\n🔍 Acurácia:", accuracy_score(y_test, y_pred))
print("\n📊 Relatório de Classificação:\n", classification_report(y_test, y_pred))

# 9. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão")
plt.show()

# 10. Coeficientes do modelo
coef_df = pd.DataFrame({
    'Variável': X.columns,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', key=np.abs, ascending=False)

print("\nCoeficientes (importância das variáveis):")
print(coef_df)