# -*- coding: utf-8 -*-

#%% Aplicação em MCA + Cluster
#Referencias
# Prof. Helder Prado
# Prof. Wilson Tarantin Jr.

# Carregar as bibliotecas

import pandas as pd
import prince
import scipy
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import scipy

#%% Importar o dataset

df = pd.read_csv("Segmentacao.csv")
print(df)

#%% Informações do dataset

df.info()

#%% Verificar quantidade de valores nulos

print((df.isna().sum()/len(df))*100)

#%% Excluir os valores nulos

df = df.dropna()

print((df.isna().sum()/len(df))*100)

#%% Selecionado somente variáveis qualitativas

df_quali = df.select_dtypes(["object"])
print(df_quali)

#%% Selecionando somente variáveis quantitativas

df_quanti = df.drop(columns=df_quali.columns)
print(df_quanti)

#%% Categorias das variáveis qualitativas

for item in df_quali.columns:
    categorias = df_quali[item].unique()
    print(categorias, f"Categorias: {len(categorias)}")


#%% Elaborando a MCA 

## Utilizando o método da matriz de Burt

mca = prince.MCA()
mca = mca.fit(df_quali)

#%% Obtendo os eigenvalues

print(mca.eigenvalues_)

#%% Inércia total

print(mca.total_inertia_)

#%% Obtendo a variância explicada em cada dimensão

# print(mca.explained_inertia_)

#%% Obtendo as coordenadas das categorias nas duas dimensões do mapa

print(mca.column_coordinates(df_quali))

#%% Obtendo as coordenadas das observações nas duas dimensões do mapa

print(mca.row_coordinates(df_quali))

#%% Plotando o mapa perceptual

# mp_mca = mca.plot_coordinates(
#                  X = df_quali,
#                  figsize = (16, 12),
#                  show_row_points = True,
#                  show_column_points = True,
#                  column_points_size = 100,
#                  show_column_labels = True,
#                  legend_n_cols = 1)
#TODO use another lib to plot perceptual map
#%% Guardando as coordendas das observações

coordenadas = mca.row_coordinates(df_quali)
coordenadas.columns = ["coord_x","coord_y"]

print(coordenadas)

#%% Adicionando as coordenadas ao dataset original

df = pd.concat([df,coordenadas], axis=1)
print(df)

#%% Redefinindo o dataset quantitativo (com as coordenadas)

df_quanti = df.drop(columns=df_quali)
print(df_quanti)

#%% Iniciando a análise de cluster

# Padronizar as variável pelo ZScore

df_quanti_z = df_quanti.copy()

for item  in df_quanti.columns:
    df_quanti_z[item] = stats.zscore(df_quanti[item])
    
print(df_quanti_z)

#%% Gerando o dendrograma (single)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti_z, method = 'single'))
plt.axhline(y = 5.5, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Gerando o dendrograma (average)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti_z, method = 'average'))
plt.axhline(y = 3.5, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Gerando o dendrograma (complete)

plt.figure(figsize=(30,10))
dendrogram = sch.dendrogram(sch.linkage(df_quanti_z, method = 'complete'))
plt.axhline(y = 5.5, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([])
plt.ylabel('Distância Euclidiana')
plt.show()

#%% Criando a variável que indica o cluster

cluster_sing = AgglomerativeClustering(n_clusters = 8, affinity = 'euclidean', linkage = 'complete')
indica_cluster_sing = cluster_sing.fit_predict(df_quanti_z)

df_quanti['cluster_complete'] = indica_cluster_sing
df_quanti['cluster_complete'] = df_quanti['cluster_complete'].astype('category')

print(df_quanti)

#%% Método de Elbow

inercias = []
K = range(1,20)

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_quanti_z)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))

plt.scatter(6,inercias[5], color = 'red', marker="X", s=200)
plt.plot(K, inercias, 'bx-')
plt.xlabel('Nº Clusters')
plt.ylabel('Inércias')
plt.title('Método de Elbow')
plt.show()

#%% Método K-Means

kmeans = KMeans(n_clusters=6, init = 'random').fit(df_quanti_z)

#%% Definindo Teste F

def f_test(kmeans_model, dataframe):
    
    variaveis = dataframe.columns

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

#%% Gerando a tabela com o teste F

f_test_df = f_test(kmeans,df_quanti_z)

#%% Adicionando a indicação de cluster ao dataset

df_quanti['cluster_kmeans'] = kmeans.labels_

#%% Plotando 

import plotly.express as px

import plotly.io as pio
pio.renderers.default = 'browser'

fig = px.scatter(df_quanti, 
                 x='coord_x',  
                 y='coord_y',
                 color='cluster_kmeans')

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
)


fig.show()

#%% Centróides dos clusters

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = df_quanti_z.columns
cent_finais.index.name = 'cluster'
print(cent_finais)

#%% Adicionando a indicação de cluster ao dataset Z

df_quanti_z['cluster_kmeans'] = kmeans.labels_

#%% Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

sns.scatterplot(x='coord_x', y='coord_y', data=df_quanti_z, hue='cluster_kmeans', palette='viridis', s=60, alpha=0.5)
plt.scatter(cent_finais['coord_x'], cent_finais['coord_y'], s = 40, c = 'red', label = 'Centróides', marker="X")
plt.title('Clusters e centróides', fontsize=16)
plt.xlabel('coord_x', fontsize=16)
plt.ylabel('coord_y', fontsize=16)
plt.legend()
plt.show()

#%% Plotando o mapa perceptual

# Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

# mp_mca = mca.plot_coordinates(
#                  X = df_quali,
#                  figsize = (16, 12),
#                  show_row_points = False,
#                  show_column_points = True,
#                  column_points_size = 100,
#                  show_column_labels = True,
#                  legend_n_cols = 1)
#TODO use another lib to plot perceptual map


sns.scatterplot(x='coord_x', y='coord_y', data=df_quanti_z, hue='cluster_kmeans', palette='viridis', s=10, alpha=0.5)
plt.legend()
plt.show()