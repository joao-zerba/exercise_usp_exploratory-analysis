# -*- coding: utf-8 -*-

#%% Aplicação MCA + Cluster
#Referencias:
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

# Carregar as bibliotecas

import pandas as pd
import prince
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import scipy

#%% Importando o dataset

# Fonte: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

df = pd.read_csv("heart.csv")
df.head()

#%% Informações das variáveis

print(df.info())

#%% Ajustando categorias das variáveis qualitativas

df.loc[df['FastingBS']==0,'Fasting'] = "No"
df.loc[df['FastingBS']==1,'Fasting'] = "Yes"

df.loc[df['HeartDisease']==0,'Disease'] = "No"
df.loc[df['HeartDisease']==1,'Disease'] = "Yes"

df = df.drop(columns=['HeartDisease','FastingBS'])

print(df)

#%% Selecionando as variáveis qualitativas

colunas = df.select_dtypes(['object']).columns

df_quali = df[colunas]

print(df_quali)

#%% Gerando os pares de tabelas de contingência

from itertools import combinations

for item in list(combinations(df_quali.columns, 2)):
    print(item, "\n")
    tabela = pd.crosstab(df_quali[item[0]], df_quali[item[1]])
    
    print(tabela, "\n")
    
    chi2, pvalor, gl, freq_esp = chi2_contingency(tabela)

    print(f"estatística qui²: {chi2}") # estatística qui²
    print(f"p-valor da estatística: {pvalor}") # p-valor da estatística
    print(f"graus de liberdade: {gl} \n") # graus de liberdade

#%% Identificando cada uma das categorias das colunas

for col in df_quali:
    print(col, df_quali[col].unique(), f"Categorias: {len(df_quali[col].unique())}")

#%% Definindo a MCA

# Inicializando a instância do MCA
mca = prince.MCA()

# Rodando o modelo
mca = mca.fit(df_quali)

#%% Coordenadas da categorias

print(mca.column_coordinates(df_quali))

#%% Plotando o mapa perceptual

# ax = mca.plot_coordinates(X=df_quali,
#                          figsize=(16,12),
#                          show_row_points = True,
#                          show_column_points = True,
#                          show_row_labels=False,
#                          column_points_size = 100,
#                          show_column_labels = True)
#TODO plot perceptual map with another lib
#%% Definindo um dataset com as variáveis quantitativas

# Iniciando uma análise de cluster

df_quanti = df.drop(columns=df_quali.columns)
print(df_quanti)

#%% Gerando o ZScore no banco de dados

for item  in df_quanti.columns:
    df_quanti[item] = stats.zscore(df_quanti[item])
    
print(df_quanti)

#%% Dendrograma (single)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti, method = 'single'))
plt.axhline(y = 2.1, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Dendrograma (average)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti, method = 'average'))
plt.axhline(y = 4, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Dendrograma (complete)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti, method = 'complete'))
plt.axhline(y = 8, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Indicando 2 clusters (complete)

cluster_sing = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'complete')
indica_cluster_sing = cluster_sing.fit_predict(df_quanti)

# Adiciona a variável ao dataset original

df['cluster_complete'] = indica_cluster_sing
df['cluster_complete'] = df['cluster_complete'].astype('category')

print(df)

#%% Plotando gráfico de pontos (idade, colesterol e o cluster)

plt.figure(figsize=(15,10))

fig = sns.scatterplot(x='Age', y='Cholesterol', data=df, hue='cluster_complete')
plt.show()

#%% Dataset com variáveis quantitativas (em ZScore)

print(df_quanti)

#%% Método Elbow para identificação do nº de clusters

## Elaborado com base na "inércia": distância de cada obervação para o centróide de seu cluster
## Quanto mais próximos entre si e do centróide, menor a inércia

inercias = []
K = range(1,6)

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_quanti)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))

plt.scatter(2,inercias[1], color = 'red')
plt.plot(K, inercias, 'bx-')
plt.xlabel('Nº Clusters')
plt.ylabel('Inércias')
plt.title('Método de Elbow')
plt.show()

#%% Método K-Means

kmeans = KMeans(n_clusters = 2, init = 'random').fit(df_quanti)

#%% Definindo o Teste F

def f_test(kmeans_model, dataframe):
    
    variaveis = kmeans.feature_names_in_

    centroides = pd.DataFrame(kmeans.cluster_centers_)
    centroides.columns = kmeans.feature_names_in_
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

        media = df[variavel].mean()

        variabilidade_entre_grupos = np.sum([dic[index]*np.square(observacao - df[variavel].mean()) for index, observacao in enumerate(centroides[variavel])])/(qnt_clusters - 1)

        dic_var['variabilidade_entre_grupos'] = variabilidade_entre_grupos

        variabilidade_dentro_dos_grupos = 0

        for grupo in unique:

            grupo = df[df.cluster == grupo]

            variabilidade_dentro_dos_grupos += np.sum([np.square(observacao - grupo[variavel].mean()) for observacao in grupo[variavel]])/(observacoes - qnt_clusters)

        dic_var['variabilidade_dentro_dos_grupos'] = variabilidade_dentro_dos_grupos

        dic_var['F'] =  dic_var['variabilidade_entre_grupos']/dic_var['variabilidade_dentro_dos_grupos']
        
        dic_var['sig F'] =  1 - scipy.stats.f.cdf(dic_var['F'], qnt_clusters - 1, observacoes - qnt_clusters)

        output.append(dic_var)

    return pd.DataFrame(output)

f_test_df = f_test(kmeans,df_quanti)

#%% Analisando os resultados do Teste F

print(f_test_df)

#%% Indicação do cluster no dataset

df['cluster_kmeans'] = kmeans.labels_

#%% Identificação dos centróides

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = df_quanti.columns
print(cent_finais)

#%% Redefinindo o dataset com ZScore

df_quanti['cluster_kmeans'] = df['cluster_kmeans']

#%% Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

sns.scatterplot(x='Age', y='Cholesterol', data=df_quanti, hue='cluster_kmeans', palette='viridis', s=60)
plt.scatter(cent_finais['Age'], cent_finais['Cholesterol'], s = 40, c = 'red', label = 'Centróides', marker="X")
plt.title('Clusters e centróides', fontsize=16)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Cholesterol', fontsize=16)
plt.legend()
plt.show()

#%% Adicionando a variável de cluster ao dataset com variáveis qualitativas

df_quali['cluster'] = df_quanti['cluster_kmeans']
print(df_quali)

#%% Informações sobre as variáveis

print(df_quali.info())

#%% Ajustando formato das variáveis

df_quali['cluster'] = df_quali['cluster'].astype("object")
print(df_quali)

#%% Excluindo coluna "Disease"

df_quali = df_quali.drop(columns=["Disease"])
print(df_quali)

#%% Gerando tabelas de contingência

from itertools import combinations

for item in list(combinations(df_quali.columns, 2)):
    print(item, "\n")
    tabela = pd.crosstab(df_quali[item[0]], df_quali[item[1]])
    
    print(tabela, "\n")
    
    chi2, pvalor, gl, freq_esp = chi2_contingency(tabela)

    print(f"estatística qui²: {chi2}") # estatística qui²
    print(f"p-valor da estatística: {pvalor}") # p-valor da estatística
    print(f"graus de liberdade: {gl} \n") # graus de liberdade


#%% Iniciando a instância do MCA

mca = prince.MCA()
mca = mca.fit(df_quali)

#%% Obtendo os eigenvalues

print(mca.eigenvalues_)

#%% Inércia total

print(mca.total_inertia_)

#%% Obtendo a variância de cada dimensão

#print(mca.explained_inertia_)

#%% Obtendo as coordenadas nas duas dimensões do mapa

print(mca.column_coordinates(df_quali))

#%% Plotando o mapa perceptual

# ax = mca.plot_coordinates(X=df_quali,
#                          figsize=(16,12),
#                          show_row_points = False,
#                          show_column_points = True,
#                          show_row_labels=False,
#                          column_points_size = 100,
#                          show_column_labels = True)
#TODO plot perceptual map with another lib
#%% Redefinindo o dataset

df_inicial = df[['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope','Fasting', 'Disease']]
print(df_inicial)

#%% Iniciando a instância do MCA

mca = prince.MCA()

# Gerando o modelo
mca = mca.fit(df_inicial)

#%% Plotando o mapa perceptual

# mp_mca = mca.plot_coordinates(
#                  X = df_inicial,
#                  figsize = (16, 12),
#                  show_row_points = False,
#                  show_column_points = True,
#                  column_points_size = 100,
#                  show_column_labels = True,
#                  legend_n_cols = 1)
#TODO plot perceptual map with another lib