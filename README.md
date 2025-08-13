# Fiap-posIA-fase-1

Repositório destinado à atividade **Tech Challenge** da primeira fase do curso **Pós-graduação em Inteligência Artificial para Devs - Turma 6IADT** (FIAP).

O projeto envolve análise de dados e criação de um modelo de Machine Learning para classificação de doenças cardíacas, além da disponibilização de uma API em **FastAPI** para servir as previsões.

---

## 📂 Estrutura de Pastas

```
.
├── app
│   ├── api
│   │   ├── main.py           # Ponto de entrada da API
│   │   ├── predict.py        # Rota para previsões
│   │   ├── schemas.py        # Definição de schemas (Pydantic)
│   ├── __pycache__           # Cache do Python
│
├── model
│   ├── analise_knn.py        # Script de análise e treino do modelo KNN
│   ├── app.py                # Inicialização do modelo para API
│   ├── modelo_pipeline.pkl   # Pipeline treinado (pré-processamento + modelo)
│   ├── modelo.pkl            # Modelo KNN treinado
│   ├── scaler.pkl            # Scaler utilizado no pré-processamento
│
├── analise_data.py           # Script para análise exploratória dos dados
├── heart_cleveland_upload.csv# Dataset utilizado
├── Dockerfile                # Configuração para containerização
├── requirements.txt          # Dependências do projeto
├── README.md                 # Este arquivo
└── TechChallenge_FIAP_Fase_1.ipynb # Notebook com todo o fluxo do projeto
```

---

## ⚙️ Pré-requisitos

- **Python 3.12+**
- **Pip** instalado
- **Docker** (para rodar a aplicação containerizada)

---

## 📦 Instalação e Execução Local

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/Bighetto/Fiap-posIA-fase-1.git
cd Fiap-posIA-fase-1
```

### 2️⃣ Criar e ativar um ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Rodar a API localmente
```bash
uvicorn app.api.main:app --reload
```

Acesse:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)  
- **Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)  

---

## 🐳 Rodar com Docker

### 1️⃣ Build da imagem
```bash
docker build -t meu-fastapi:latest .
```

### 2️⃣ Rodar o container
```bash
docker run --rm -p 8000:8000 meu-fastapi:latest
```

---

## 📊 Funcionalidades

- **Treinamento do modelo KNN** otimizado com *GridSearchCV*
- **Pré-processamento** (normalização, tratamento de dados)
- **Análise Exploratória** com gráficos
- **API REST** para servir previsões do modelo
- **Documentação interativa** via Swagger

---

## 📈 Fluxo do Projeto

1. **Análise dos Dados** (`analise_data.py` / notebook)
2. **Treinamento do Modelo** (`analise_knn.py`)
3. **Exportação do Pipeline Treinado** (`modelo_pipeline.pkl`)
4. **Criação da API** (`main.py`, `predict.py`, `schemas.py`)
5. **Execução local ou via Docker**

---

## ✨ Tecnologias Utilizadas

- **Python**
- **Scikit-learn**
- **Pandas**
- **Matplotlib / Seaborn**
- **FastAPI**
- **Docker**
