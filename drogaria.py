# -*- coding: utf-8 -*-

#%% Aplicação em Análise Fatorial PCA

#Referencia:
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

# Carregar as bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg

#%% Importando o banco de dados

# Fonte: Fávero & Belfiore (2017, Capítulo 10) 

df = pd.read_excel("PercepcaoDrogaria.xlsx")
print(df)

#%% Matriz de correlaçãoes entre as variáveis

matriz_corr = pg.rcorr(df, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

print(matriz_corr)

#%% Gráfico com a matriz de correlações

fig = plt.figure(figsize=(9,7), facecolor='white')
sns.heatmap(df.corr(), vmin = -1, annot=True,linewidth=0.5,  annot_kws={"fontsize":10, 'rotation':0} ,fmt = '.2f', cmap = 'coolwarm')

#%% Teste de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(df)
print(f'Bartlett statistic: {bartlett}')
print(f'p-value : {p_value}')

#%% Estatística KMO

kmo_all, kmo_model = calculate_kmo(df)
print(f'kmo_model : {kmo_model}')

#%% Definindo a PCA (inicial)

fa = FactorAnalyzer()
fa.fit(df)

#%% Obtendo os Eigenvalues

ev, v = fa.get_eigenvalues()
print(ev)

#%% Critério de Kaiser

## Verificar eigenvalues com valores maiores que 1

print([item for item in ev if item > 1])

#%% Parametrizando a PCA para 2 fatores (autovalores > 1)

fa.set_params(n_factors = 2, method = 'principal', rotation = None)
fa.fit(df)

#%% Eigenvalues, variâncias e variâncias acumulada

eigen_fatores = fa.get_factor_variance()
eigen_fatores

tabela_eigen = pd.DataFrame(eigen_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

cargas_fatores = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatores)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = df.columns
tabela_cargas

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = df.columns
tabela_comunalidades

print(tabela_comunalidades)

#%% Resultado do fator para as observações do dataset

predict_fatores= pd.DataFrame(fa.transform(df))
predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]

# Adicionando ao banco de dados

df = pd.concat([df.reset_index(drop=True), predict_fatores], axis=1)

print(df)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = tabela_cargas.index

print(tabela_scores)

#%% Criando um ranking

df['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    df['Ranking'] = df['Ranking'] + df[tabela_eigen.index[index]]*variancia


#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8))

plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
ax = sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, color='green')

ax.bar_label(ax.containers[0])
plt.xlabel("Componentes principais", fontsize=14)
plt.ylabel("Porcentagem de variância explicada (%)", fontsize=14)
plt.show()


#%% Obtendo o índice da tabela de cargas fatoriais

tabela_cargas = tabela_cargas.reset_index()

#%% Plotando no gráfico

plt.figure(figsize=(16,10))
plt.scatter(tabela_cargas["Fator 1"], tabela_cargas["Fator 2"])

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val':val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(x = tabela_cargas["Fator 1"],
            y = tabela_cargas["Fator 2"],
            val=tabela_cargas["index"],
            ax = plt.gca()) 

plt.xlabel("PC 1", fontsize=14)
plt.ylabel("PC 2", fontsize=14)

plt.axhline(y = 0, color = 'gray', linestyle = '--')
plt.axvline(x = 0, color = 'gray', linestyle = '--')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()

#%% Podemos observar o ranking formado

print(df.sort_values(by=["Ranking"],ascending=True))

#%% Gráfico interativo

import plotly.express as px
import plotly.io as pio

pio.renderers.default='browser'

fig = px.scatter(tabela_cargas, x='Fator 1', y='Fator 2', text="index")

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
)

fig.update_yaxes(range=[-1, 1], row=1, col=1)
fig.update_xaxes(range=[-1, 1], row=1, col=1)

fig.show()
