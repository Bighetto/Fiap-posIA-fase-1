import pandas as pd
import numpy as np
import kagglehub

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

np.random.seed(42)
FEATURES = ["thal", "ca", "oldpeak", "exang", "cp"]
TARGET = "condition"
print("[INFO] Baixando dataset do KaggleHub...")
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
csv_path = f"{path}/heart_cleveland_upload.csv"
df = pd.read_csv(csv_path)
print(f"[OK] Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas.")

# Filtro etário 40–70 (mesmo do app/API)
df = df[(df["age"] >= 40) & (df["age"] <= 70)].copy()

# Seleciona as mesmas 5 features da API
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Shapes ->",
      "X:", X.shape, "| y:", y.shape,
      "| X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Features usadas:", FEATURES)

# Pipeline + Busca de Hiperparamtro
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

param_grid = {
    "clf__n_neighbors": [3, 5, 7, 9, 11],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0,
)

gs.fit(X_train, y_train)

print("\nMelhores hiperparâmetros (CV por F1):", gs.best_params_)
print("Melhor F1 (validação):", round(gs.best_score_, 3))

# Avaliação no teste 
best_knn = gs.best_estimator_
y_pred = best_knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print("\n=== Desempenho no TESTE ===")
print("Accuracy:", round(acc, 3))
print("F1      :", round(f1, 3))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
