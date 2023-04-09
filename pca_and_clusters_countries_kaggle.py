# -*- coding: utf-8 -*-

#%% Análise Fatorial PCA + Cluster
#Referencias
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

# Carregando os pacotes necessários

import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%% Importando o banco de dados

# Fonte: https://www.kaggle.com/datasets/vipulgohel/clustering-pca-assignment?resource=download&select=Country-data.csv

dados_paises = pd.read_csv("Países PCA Cluster.csv")
print(dados_paises)

#%% Informações sobre as variáveis

print(dados_paises.info())
print(dados_paises.describe())

#%% Separando somente as variáveis quantitativas do banco de dados

paises_pca = dados_paises[["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]]
print(paises_pca)

#%% Matriz de correlaçãoes entre as variáveis

matriz_corr = pg.rcorr(paises_pca, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(matriz_corr)

#%% Outra maneira de plotar as mesmas informações

corr = paises_pca.corr()

f, ax = plt.subplots(figsize=(11, 9))

mask = np.triu(np.ones_like(corr, dtype=bool))

cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)

sns.heatmap(dados_paises.corr(), 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin = -.25,
            center=0,
            square=True, 
            linewidths=.5,
            annot = True,
            fmt='.3f', 
            annot_kws={'size': 16},
            cbar_kws={"shrink": .75})

plt.title('Matriz de correlação')
plt.tight_layout()
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 14)
ax.set_ylim(len(corr))

plt.show()

#%% Teste de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(paises_pca)

print(f'Bartlett statistic: {bartlett}')

print(f'p-value : {p_value}')


#%% Estatística KMO

kmo_all, kmo_model = calculate_kmo(paises_pca)

print(f'kmo_model : {kmo_model}')


#%% Definindo a PCA (procedimento preliminar)

fa = FactorAnalyzer()
fa.fit(paises_pca)


#%% Obtendo os Eigenvalues (autovalores)

ev, v = fa.get_eigenvalues()

print(ev)

#%% Critério de Kaiser

# Verificar autovalores com valores maiores que 1
# Existem 3 componentes acima de 1

#%% Parametrizando a PCA para 3 fatores (autovalores > 1)

fa.set_params(n_factors = 3, method = 'principal', rotation = None)
fa.fit(paises_pca)


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
tabela_cargas.index = paises_pca.columns
tabela_cargas

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = paises_pca.columns
tabela_comunalidades

print(tabela_comunalidades)

#%% Resultados dos fatores para as observações do dataset (predict)

predict_fatores= pd.DataFrame(fa.transform(paises_pca))
predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]

print(predict_fatores)

# Adicionando ao dataset original

dados_paises = pd.concat([dados_paises.reset_index(drop=True), predict_fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = paises_pca.columns
tabela_scores

print(tabela_scores)

#%% Correlação entre os fatores

# A seguir, verifica-se que a correlação entre os fatores é zero (ortogonais)

corr_fator = pg.rcorr(dados_paises[['Fator 1','Fator 2', 'Fator 3']], method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(corr_fator)

#%% Gráfico das cargas fatoriais e suas variâncias nos componentes principais

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

tabela_cargas_chart = tabela_cargas.reset_index()

plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=30)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='black', ls='--')
plt.axvline(x=0, color='black', ls='--')
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
plt.xlabel(f"PC 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.ylabel(f"PC 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.show()


#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8))

plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, color='green')
plt.xlabel("Componentes principais", fontsize=14)
plt.ylabel("Porcentagem de variância explicada", fontsize=14)
plt.show()

#%% Análise de Cluster (utilizando os fatores)

# Importando os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import scipy.stats as stats
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np


#%% Selecionado apenas variáveis métricas (os fatores)

paises_cluster = dados_paises[["Fator 1", "Fator 2", "Fator 3"]]

#%% Cluster Hierárquico Aglomerativo

# Gerando o dendrograma

plt.figure(figsize=(16,8))

dendrogram = sch.dendrogram(sch.linkage(paises_cluster, method = 'complete', metric = 'euclidean'), labels = list(dados_paises.country))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 3, color = 'red', linestyle = '--')
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

#%% Gerando a variável com a indicação do cluster no dataset

## Deve ser realizada a parametrização:
    ## Número de clusters
    ## Medida de distância
    ## Método de encadeamento
    
## A medida de distância e o método de encadeamento são mantidos

cluster_comp = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(paises_cluster)

# Retorna uma lista de valores com o cluster de cada observação

print(indica_cluster_comp, "\n")

dados_paises['cluster_comp'] = indica_cluster_comp

print(dados_paises)

#%% Cluster Não Hierárquico K-means

# Considerando que identificamos 10 possíveis clusters na análise hierárquica

kmeans = KMeans(n_clusters = 10, init = 'random').fit(paises_cluster)

#%% Para identificarmos os clusters gerados

kmeans_clusters = kmeans.labels_

print(kmeans_clusters)

dados_paises['cluster_kmeans'] = kmeans_clusters

print(dados_paises)

#%% Identificando as coordenadas centróides dos clusters finais

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = paises_cluster.columns
cent_finais.index.name = 'cluster'
print(cent_finais)

#%% Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

pred_y = kmeans.fit_predict(paises_cluster)
sns.scatterplot(x='Fator 1', y='Fator 2', data=dados_paises, hue='cluster_kmeans', palette='viridis', s=60)
plt.scatter(cent_finais['Fator 1'], cent_finais['Fator 2'], s = 40, c = 'red', label = 'Centróides', marker="X")
plt.title('Clusters e centróides', fontsize=16)
plt.xlabel('Fator 1', fontsize=16)
plt.ylabel('Fator 2', fontsize=16)
plt.legend()
plt.show()

#%% Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
## Elaborado com base na "inércia": distância de cada obervação para o centróide de seu cluster
## Quanto mais próximos entre si e do centróide, menor a inércia

inercias = []
K = range(1,paises_cluster.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(paises_cluster)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, inercias, 'bx-')
plt.axhline(y = 10, color = 'red', linestyle = '--')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Inércias', fontsize=16)
plt.title('Método do Elbow', fontsize=16)
plt.show()

# Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

#%% Estatística F

# Análise de variância de um fator:
# As variáveis que mais contribuem para a formação de pelo menos um dos clusters

def teste_f_kmeans(kmeans, dataframe):
    
    variaveis = dataframe.columns

    centroides = pd.DataFrame(kmeans.cluster_centers_)
    centroides.columns = dataframe.columns
    centroides
    
    print("Centróides: \n", centroides ,"\n")

    df = dataframe[variaveis]

    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    dic = dict(zip(unique, counts))

    qnt_clusters = kmeans.n_clusters

    observacoes = len(kmeans.labels_)

    df['cluster'] = kmeans.labels_

    output = []

    for variavel in variaveis:

        dic_var={'variavel':variavel}

        # variabilidade entre os grupos

        variabilidade_entre_grupos = np.sum([dic[index]*np.square(observacao - df[variavel].mean()) for index, observacao in enumerate(centroides[variavel])])/(qnt_clusters - 1)

        dic_var['variabilidade_entre_grupos'] = variabilidade_entre_grupos

        variabilidade_dentro_dos_grupos = 0

        for grupo in unique:

            grupo = df[df.cluster == grupo]

            variabilidade_dentro_dos_grupos += np.sum([np.square(observacao - grupo[variavel].mean()) for observacao in grupo[variavel]])/(observacoes - qnt_clusters)

        dic_var['variabilidade_dentro_dos_grupos'] = variabilidade_dentro_dos_grupos

        dic_var['F'] =  dic_var['variabilidade_entre_grupos']/dic_var['variabilidade_dentro_dos_grupos']
        
        dic_var['sig F'] =  1 - stats.f.cdf(dic_var['F'], qnt_clusters - 1, observacoes - qnt_clusters)

        output.append(dic_var)

    df = pd.DataFrame(output)
    
    print(df)

    return df

# Os valores da estatística F são bastante sensíveis ao tamanho da amostra

output = teste_f_kmeans(kmeans,paises_cluster)

#%% Gráfico 3D dos clusters

import plotly.express as px 
import plotly.io as pio

pio.renderers.default='browser'

fig = px.scatter_3d(dados_paises, 
                    x='Fator 1', 
                    y='Fator 2', 
                    z='Fator 3',
                    color='cluster_kmeans')
fig.show()