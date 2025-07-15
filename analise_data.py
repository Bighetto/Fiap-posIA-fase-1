import pandas as pd
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")

csv_path = f"{path}/heart_cleveland_upload.csv" 
df = pd.read_csv(csv_path)

#Informacoes sobre as colunas e os dados.
# print(df.columns)
# print(df.info())

sns.countplot(data=df, x='condition')
plt.title('Distribuição dos casos (0 = sem doença, 1 = com doença)')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.show()

sns.histplot(data=df, x='age', hue='condition', multiple='stack', bins=15)
plt.title('Distribuição de idade por condição cardíaca')
plt.show()

sns.boxplot(data=df, x='condition', y='chol')
plt.title('Níveis de colesterol por condição')
plt.show()

df['faixa_idade'] = pd.cut(df['age'], bins=[20, 40, 50, 60, 70, 100], labels=['20-40', '41-50', '51-60', '61-70', '71+'])

sns.boxplot(data=df, x='faixa_idade', y='thalach', hue='condition')
plt.title('Frequência cardíaca máxima por faixa etária')
plt.show()