#!/usr/bin/env python
# coding: utf-8

# Ce notebook présente différentes algorithmes de Machine Learning appliqués à des tweets.
# Il s'agit d'un projet d'étude réalisé dans le cadre du DU Data Analysis de la Sorbonne.

# # 1. Import des données
# * https://mihaelagrigore.medium.com/scraping-historical-tweets-without-a-twitter-developer-account-79a2c61f76ab
# * https://github.com/JustAnotherArchivist/snscrape



# In[ ]:


import re
 

import os
import subprocess

import json
import csv

import uuid

from IPython.display import display_javascript, display_html, display

import pandas as pd
import numpy as np

from datetime import datetime, date, time

import matplotlib.pyplot as plt

# Pour le pré processing
from unidecode import unidecode
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pour la dataviz
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Les bigrammes
from collections import Counter
from nltk.util import ngrams

# Pour la vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.feature_extraction.text import HashingVectorizer

# Pour la modélisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import MinMaxScaler


from sklearn.naive_bayes import GaussianNB

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis 

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# hyperparameter training imports
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans


import emojis

sns.set()

from collections import defaultdict
from sklearn import metrics
from time import time

import simplemma

from nltk.probability import FreqDist

from yellowbrick.cluster import KElbowVisualizer


from nltk.cluster import KMeansClusterer


pd.set_option('display.min_rows', 50)
pd.options.display.max_colwidth = 150

plt.style.use('ggplot')

plt.rcParams['font.size'] = '16'

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter


# In[ ]:


# Choix entre un fichier existant ou une nouvelle recherche
choix_data=int(input('Choix ancien fichier=0, Choix nouvelle recherche=1 : '))


# In[ ]:


if choix_data==1:

    # Scrap d'un nouveau fichier et Sauvegarde sous format json
    json_filename = 'Ukraine-query-tweets.json'

    #Using the OS library to call CLI commands in Python
    os.system(f'snscrape --max-results 500000 --jsonl --progress --since 2022-02-24 twitter-search "#Ukraine lang:fr until:2022-08-18" > {json_filename}')

elif choix_data==0:

    # Utilisation d'un fichier json existant
    json_filename = 'Ukraine-query-tweets-v2.json'
 


# In[ ]:



# Création d'un dataframe df contentant les tweets

df = pd.read_json(json_filename, lines=True)

# Sauvegarde d'une copie
df_copy = df.copy()

# Visualisation de df
df.head()


# # 2. Premières analyses: évolution du nombre de tweets dans le temps

# ## 2.1 Evolution du nombre de tweets dans le temps

# In[ ]:


# Création de df2: 2 colonnes= day + count=nbre de tweets par jour
df['day'] = df['date'].dt.strftime('%D') 
df['week_number'] = df['date'].dt.strftime('%V') 
df2=df.groupby(['day']).size().reset_index(name='counts')
df2.head()


# In[ ]:


# Evolution du nombre de tweets par jour
fig, ax = plt.subplots(figsize=(15,10))
x = df2['day']
y = df2['counts']
plt.plot(x, y)
_ = ax.set_xticks(x[::5])
_ = ax.set_xticklabels(x[::5], rotation=45)
_=ax.tick_params(axis='both', which='major', labelsize=14)
#_ = plt.xlabel('Jour')
_ = plt.ylabel('Nombre de tweets par jour')


# In[ ]:


# Création de df3: 2 colonnes= semaine + count=nbre de tweets par semaine
df3=df.groupby(['week_number']).size().reset_index(name='counts')
df3.head()


# In[ ]:


# Evolution du nombre de tweets par semaine
fig, ax = plt.subplots(figsize=(15,10))
plt.bar(df3['week_number'], df3['counts'])
_ = ax.set_xlabel('Week Number')
_ = ax.set_ylabel('Number of tweets')
_=ax.tick_params(axis='both', which='major', labelsize=14)


# ## 2.2 Nombre de mots
# 

# In[ ]:


document_lengths = np.array(list(map(len, df['content'].str.split(' '))))

print(f"Le nombre moyen de mots par tweet est : {int(np.mean(document_lengths))}.")


print(f"Le nombre minimum de mots par tweet est: {min(document_lengths)}.")
print(f"Le nombre maximum de mots par tweet est: {max(document_lengths)}.")


# In[ ]:


fig, ax = plt.subplots(figsize=(15,6))

_=ax.set_title("Distribution of number of words before preprocessing", fontsize=16)
_=ax.set_xlabel("Number of words")
_=sns.distplot(document_lengths, bins=50, ax=ax)
_=ax.tick_params(axis='both', which='major', labelsize=14)


# ## 2.3. Analyse des emoji pour préparer une analyse de sentiments (non traitée ici)
# * https://emojis.readthedocs.io/en/latest/api.html#module-emojis
# * https://www.kaggle.com/code/infamouscoder/emoji-sentiment-features
# * https://www.kaggle.com/code/infamouscoder/emoji-sentiment-features
# 

# ### 2.3.1. Analyse globale

# In[ ]:


#!pip install emojis


# In[ ]:


def create_column_emoji(my_pd):
    """
    Crée une liste des labels des emojis en francais.

    Paramètres
    ----------

    my_list : liste d'emojis.

    """
    col_emo=[]
    my_set=emojis.get(my_pd)
    #for emo in my_set:
    #    val = emoji.demojize(emo, language='fr').split(':')[1]
    #    col_emo.append(val)
    return list(my_set)
    


# In[ ]:


# Création d'une colonne qui contient tous les émojis des tweets
df['emoji']=df['content'].apply((lambda x : create_column_emoji(x)))


# In[ ]:


df_copy = df.copy()


# In[ ]:


print(f"Le % de tweets comportant des emojis est: {np.ceil(100*df.loc[(df['emoji'].str.len() != 0),:].shape[0]/df.shape[0])}%.")


# In[ ]:


df['emoji'].head()


# In[ ]:


#df_emoji = df[df['emoji'].str.len() != 0]


# Cela n'est pas suffisant pour une prise en compte.

# ### 2.3.2. Preprocessing pour une analyse de sentiments

# In[ ]:


# liste de tous les emojis presents
#res_list = [y for x in df['emoji'] for y in x]


# In[ ]:


#len(res_list)


# In[ ]:


# Liste d'emojis uniques
#emo_list = list(set(res_list))


# In[ ]:


#len(emo_list)


# In[ ]:


df_emoji = df[df['emoji'].str.len() != 0]


# In[ ]:


# Création de nouvelles colonnes correspondant aux sentiments exprimés

df_emoji['positive_emoji'] = 0
df_emoji['neutral_emoji'] = 0
df_emoji['negative_emoji'] = 0

positive_emoji = ['❤️','❤','😍','♥️','😊','💕','👍','😂','🙌','🤑','💖','✨','😊','🎉','💞','😝','😈','😃','😁','😎','😘','💓','😉','😬','😄','😀','😜','💗','😌','😆','😛','😻','🙋','❣️','🙂','😇','💝','😏','😋','🤗','🙆','🤓','😚','😙','😸','😼','😺','😽']
neutral_emoji = ['🙏','💜','💙','👽','💛','💟','💚','😅','🙃','💩','😳','🙄','😑','🙇','🙎','😐','😶']
negative_emoji = ['💥','💘','😭','😱','👎','😫','😨','😢','💀','🤔','👻','😓','💦','😤','😩','😴','💔','😒','😪','😈','😣','😮','😡','😕','😔','😠','😷','😥','😞','😲','😰','🙀','😖','😧','😟','😹','😵','😶','😯','🤒','🤕','😾','💤']


# In[ ]:


for idx, text in enumerate(df_emoji['emoji']):
    
    for emoj in text:
        
        if emoj in positive_emoji:
            df_emoji['positive_emoji'].iloc[idx] += 1
        elif emoj in negative_emoji:
            df_emoji['negative_emoji'].iloc[idx] += 1
        else:
            df_emoji['neutral_emoji'].iloc[idx] += 1


# In[ ]:


print(f" Voici le bilan de l'analyse de sentiments: \n")
print(f"Score positif: {df_emoji['positive_emoji'].value_counts()} " )
print(f"Score négatif: {df_emoji['negative_emoji'].value_counts()} " )
print(f"Score neutre:  {df_emoji['neutral_emoji'].value_counts()} " )


# ## 2.4. Suppression des hastags, url, arobase

# In[ ]:


df.head()


# In[ ]:


#remove urls, hashtags, arobase (et emoji)
def remove_urls(text):
    return re.sub(r'http\S+',' ', text)

def remove_hashtags(text):
    return re.sub(r'#\S+',' ', text)

def remove_arobase(text):
    return re.sub(r'@\S+',' ', text)

def remove_emojis(text):
    string = ' '.join([word for word in text if word not in emo_list])
    return string 


# In[ ]:


df['content_clean'] = df['content'].apply(remove_hashtags)
df['content_clean'] = df['content_clean'].apply(remove_urls)
df['content_clean'] = df['content_clean'].apply(remove_arobase)


# In[ ]:


df[['content','content_clean']].head()


# # 3. Listes de stop words

# ## 3.1. Module ntk: french_stopwords_list

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')


# In[ ]:



# Création d'une liste de stopword en français à partir de nltk
french_stopwords_list = stopwords.words('french')

# Suppression des accents
french_stopwords_list=[unidecode(x) for x in french_stopwords_list]

# ajout ukraine, guerre et annee
french_stopwords_list.append('ukraine')
french_stopwords_list.append('guerre')
french_stopwords_list.append('annee')

#print(f"Ci-dessous, la liste des stopwords en français de nltk :\n{french_stopwords_list}")


# ## 3.2 Module stopwords: french_stopwords_list_2

# In[ ]:


get_ipython().system('pip install stop-words')


# In[ ]:


from stop_words import get_stop_words


# In[ ]:


french_stopwords_list_2 = get_stop_words('french')
#print(f"Ci-dessous, la liste des stopwords en français de stop_words :\n{french_stopwords_list_2}")


# In[ ]:


# Suppression des accents

french_stopwords_list_2=[unidecode(x) for x in french_stopwords_list_2]
#print(f"Ci-dessous, la liste des stopwords SANS ACCENT de stop_words :\n{french_stopwords_list_2}")


# ## 3.3 Fusion des 2 listes: french_stopwords_list_3
# 

# In[ ]:


french_stopwords_list_3 = set(french_stopwords_list + french_stopwords_list_2)
french_stopwords_list_3 = sorted (french_stopwords_list_3 )

print(f"Ci-dessous, la liste complète des stopwords en français :\n{french_stopwords_list_3}")


# # 4. Prétraitement (suite) 

# ## 4.1 Prétraitement: Nettoyage complet = suppression des stopwords, etc..
# 

# In[ ]:


# Création d'une fonction pour supprimer les stop words
def no_stop_word(string, stopWords):

    """
    Supprime les stop words d'un texte.

    Paramètres
    ----------

    string : chaine de caractère.

    stopWords : liste de mots à exclure. 
    """
    
    string = ' '.join([word for word in string.split() if word not in stopWords])
    return string

# Création de la fonction de NETTOYAGE COMPLET
def final_cleaner(pandasSeries, stopWords):
    
    """
    Stemmatise une Series Pandas de documents 

    Paramètres
    ----------
    
    pandasSeries : Une Series Pandas

    stemmer : Stemmer de NLTK
    
    stopWords : Une liste de stopWords
    """
    
    print("#### Nettoyage en cours ####") 
    
    # confirmation que chaque article est bien de type str
    pandasSeries = pandasSeries.apply(str)
        
    # Passage en minuscule
    print("... Passage en minuscule") 
    pandasSeries = pandasSeries.apply(lambda x : x.lower())
    
    # Suppression des accents
    print("... Suppression des accents") 
    pandasSeries = pandasSeries.apply(unidecode)
    
    # Détection du champs année
    print("... Détection du champs année") 
    pandasSeries = pandasSeries.apply(lambda x : re.sub(r'[0-9]{4}', 'annee', x))
    
    # Suppression http
    #print("... Suppression http") 
    #pandasSeries = pandasSeries.apply(lambda x : re.sub(r'https://', ' ', x))
    
    # Suppression des caractères spéciaux et numériques
    print("... Suppression des caractères spéciaux et numériques") 
    pandasSeries = pandasSeries.apply(lambda x :re.sub(r"[^a-z]+", ' ', x))
    
    # Suppression des stop words
    print("... Suppression des stop words") 
    #stopWords = [unidecode(sw) for sw in stopWords]
    pandasSeries = pandasSeries.apply(lambda x : no_stop_word(x, stopWords))
 
    print("#### Nettoyage OK! ####")

    return pandasSeries


# In[ ]:


# Application du nettoyage final
df['content_clean_final'] = final_cleaner(df['content_clean'], french_stopwords_list_3)


# ## 4.2 STEMMATISATION et LEMMATISATION

# ### 4.2.1 STEMMATISATION 

# In[ ]:


# Création d'une fonction de STEMMATISATION

def stemmatise_text(text,stemmer):

    """
    Stemmatise un texte : Ramène les mots d'un texte à leur racine (peut créer des mots qui n'existe pas).

    Paramètres
    ----------

    text : Chaine de caractères.

    stemmer : Stemmer de NLTK.
    """

    return " ".join([stemmer.stem(word) for word in text.split()])


# In[ ]:


# On initialise un stemmer NLTK
stemmer = SnowballStemmer('french')


# Application de la fonction stemmatise_text
df['content_stem'] = df['content_clean_final'].apply(lambda x : stemmatise_text(x,stemmer))

df[['content_clean_final', 'content_stem']].head()


# ### 4.2.2 LEMMATISATION
# 

# In[ ]:


# Création d'une fonction de LEMMATISATION

def lemmatise_text(text):

    """
   Lemmatise un texte 
    Paramètres
    ----------

    text : Chaine de caractères.

    lemmer : lemmer de simplema
    """

    return " ".join([simplemma.lemmatize(word, lang='fr') for word in text.split()])

# Application de la fonction lemmatise_text
df['content_lem'] = df['content_clean_final'].apply(lambda x : lemmatise_text(x))

df[['content_stem', 'content_lem']].head()


# ### 4.2.3 Filtrage des tweets qui contiennent trop peu de mots

# In[ ]:


# Taille avant filtrage
df.shape


# In[ ]:


# Filtrage 
df = df[df['content_lem'].str.len() >= 3]
# Taille après filtrage
df.shape


# ## 4.3. Visualisation des MOTS LES PLUS UTILISES après traitement

# In[ ]:


# Création d'une variable contenant le nombre de "mots" de chaque article
df['nb_words_lem'] = df['content_lem'].apply(lambda x: len(x.split()))

# Affichage du dataframe df
#df.head()

# Répartition des tweets en fonction du nombre de mots
#plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(15,10))
plt.hist(df['nb_words_lem'], bins=30, color='b', edgecolor='k')
_ = ax.set_xlabel('Number of words after preprocessing')
_ = ax.set_ylabel('Number of tweets')
_=ax.tick_params(axis='both', which='major', labelsize=14)


# In[ ]:


df['nb_words_lem'].value_counts()


# In[ ]:


df = df[df['nb_words_lem'] >= 2]


# In[ ]:


df[df['nb_words_lem']==3]


# In[ ]:


# Import the wordcloud library
import wordcloud 

# Join the different processed tweets together.
long_string = ' '.join(df['content_stem'])

# Create a WordCloud object
wc = wordcloud.WordCloud(width=400,
                      height=330,
                      max_words=50,
                      colormap='tab20c',
                      collocations=True)

# Generate a word cloud
wc.generate(long_string)

# Visualize the word cloud
plt.figure(figsize=(10,8))
plt.imshow(wc)
plt.axis('off')
plt.title('Words Clouds', fontsize=13)
plt.show()


# In[ ]:


# Fonction pour afficher les mots les plus utilisés 
def print_words(df , col, nb_words):
    
    """
   print les max_words mots les plus fréquemment utilisés par cluster

    Paramètres
    ----------
    
    df : DataFrame Pandas
    
    col : La série de df à analyser (après pré-processing)

    nb_words : nombre de mots

    """


        
    data = df[col]
        
    long_string = ' '.join(data)
        
    my_counts =  Counter(re.findall('\w+', long_string))
  
    most_occur = my_counts.most_common(nb_words)
  
    #print(f"Top {nb_words} :\n {most_occur}.")
    
    return most_occur


# In[ ]:


top_list=print_words(df , 'content_lem', 20)
df_top = pd.DataFrame(top_list, columns= ['Word','Count'])
df_top.head(20)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
plt.bar(df_top['Word'], df_top['Count'])
_ = ax.set_xticklabels(df_top['Word'], rotation=45)
_ = ax.set_xlabel('Mot')
_ = ax.set_ylabel('Nombre d\'occurences')
_=ax.tick_params(axis='both', which='major', labelsize=14)


# ## 4.4. Analyse des hashtags

# In[ ]:


# Nombre de tweets pour lesquels il n'y a pas de hastag
df['hashtags'].isnull().sum()


# In[ ]:


# Suppression de ces tweets qui sont peu nombreux
df.dropna(subset=['hashtags'], inplace= True)


# In[ ]:


# Fonction qui transforme les hashtags en une liste de mots
def list_hashtags(list_txt):
    if len(list_txt)==1:
        return list_txt[0]
    else: 
        return ' '.join(list_txt)


# In[ ]:


#clean_hashtags([df['hashtags'][0]])


# In[ ]:


df['hashtags_clean'] = df['hashtags'].apply(lambda x : list_hashtags(x))


# In[ ]:


df[['hashtags', 'hashtags_clean']].head(10)


# In[ ]:


# Join the different processed tweets together.
long_string_hashtags = ' '.join(df['hashtags_clean'])

# Create a WordCloud object
wc = wordcloud.WordCloud(width=400,
                      height=330,
                      max_words=50,
                      colormap='tab20c',
                      collocations=True)

# Generate a word cloud
wc.generate(long_string)

# Visualize the word cloud
plt.figure(figsize=(10,8))
plt.imshow(wc)
plt.axis('off')
plt.title('Words Clouds Hashtags', fontsize=13)
plt.show()


# In[ ]:


top_list_hashtags=print_words(df , 'hashtags_clean', 20)
df_top_hashtags = pd.DataFrame(top_list, columns= ['Word','Count'])
df_top_hashtags.head(20)


# L'analyse des mots les plus utilisés dans les hashtags et les tweets donne à peu près la même chose...

# # 5. Clustering

# In[ ]:


# Suppression des tweets qui contiennent moins de 3 mots après lemmatisation


# ## 5.1 Kmeans

# In[ ]:


# Vectoriseur TFIDF: on ignore les mots présents dans plus de 50% des tweets ou dans moins de 1000 tweets
# On ne prend en compte que les bigrams et lres trigrams.
tfidf = TfidfVectorizer(max_df=0.5,
                        min_df=2000,
                        ngram_range=(1,3))
                       


# In[ ]:


# Vectoriseur COUNT
#count = CountVectorizer(min_df=1000,
                        #max_df=0.5,
                        #ngram_range=(1, 3), # sélection bigrammes) 


# In[ ]:


tfidf_matrix = tfidf.fit_transform([x for x in df["content_lem"]])

print(tfidf_matrix.shape)


# In[ ]:


#count_matrix = count.fit_transform([x for x in df["content_lem"]])


# In[ ]:


#print(count_matrix.shape)


# In[ ]:


list_words_tfidf = tfidf.get_feature_names()
#type(list_words_tfidf)


# https://kavita-ganesan.com/hashingvectorizer-vs-countvectorizer/#.YzVS_YRBwuW
# 
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py

# In[ ]:


# Create a WordCloud object
wc = wordcloud.WordCloud(width=400,
                      height=330,
                      max_words=50,
                      colormap='tab20c',
                      collocations=True)

# Generate a word cloud
wc.generate(' '.join(list_words_tfidf))

# Visualize the word cloud
plt.figure(figsize=(10,8))
plt.imshow(wc)
plt.axis('off')
plt.title('Words Clouds TFIDF', fontsize=16)
plt.show()


# In[ ]:


Sum_of_squared_distances =[]
K = range(1,30,2)
for k in K:
    km =KMeans(n_clusters =k)
    km =km.fit(tfidf_matrix)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()


# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py

# In[ ]:


# Application de kmeans puis création d'une colonne cluster
km = KMeans(
        n_clusters=10,
        max_iter=100,
        n_init=1,
        random_state=5)


# Fit the k-means object with tfidf_matrix or count_matrix
km.fit_transform(tfidf_matrix)

clusters = km.labels_.tolist()

# Create a column cluster to denote the generated cluster for each tweet
df["cluster"] = clusters

# Display number of tweets per cluster 
df['cluster'].value_counts() 


# In[ ]:



def print_words_clusters(df , col, col_clus,  nb_cluster, nb_words):
    
    """
   print les max_words mots les plus fréquemment utilisés par cluster

    Paramètres
    ----------
    
    df : DataFrame Pandas
    
    col : La série de df à analyser (sur lesquelles les clusters ont été calculés)

    nb_cluster : nombre de clusters à prendre en compte
    
    nb_words : nombre de mots

    """

    for i in range(nb_cluster):
        
        data = df[df[col_clus] == i][col]
        
        long_string = ' '.join(data)
        
        my_counts =  Counter(re.findall('\w+', long_string))
  
        most_occur = my_counts.most_common(nb_words)
  
        print(f"Top {nb_words} du cluster n = {i+1} :\n {most_occur}.")
    
    


# In[ ]:


# Analyse des différents types de tweets suivant les clusters:
# Join the different processed titles together.

def plot_words_clusters(df , col, col_clus,  nb_cluster, max_words):
    
    """
   Trace un words_clouds pour chaque cluster 

    Paramètres
    ----------
    
    df : DataFrame Pandas
    
    col : La série de df à analyser (sur lesquelles les clusters ont été calculés)

    nb_cluster : nombre de clusters à prendre en compte
    
    max_words : nombre max de mots

    """

    for i in range(nb_cluster):
        
        data = df[df[col_clus] == i][col]
        
        long_string = ' '.join(data)

        # Create a WordCloud object
        wc = wordcloud.WordCloud(width=400,
                                height=330,
                                max_words=max_words,
                                colormap='tab20c',
                                collocations=True)

        # Generate a word cloud
        wc.generate(long_string)

        # Visualize the word cloud
        plt.figure(figsize=(10,8))
        plt.imshow(wc)
        plt.axis('off')
        plt.title(f'Words Clouds pour le cluster n={i+1}', fontsize=13)
        plt.show()


# In[ ]:


plot_words_clusters(df,"content_lem", "cluster", 10, 30)


# ## 5.3 Kmeans après réduction de dimension

# ### 5.2.1. Kmeans avec réduction de dimension: avec TFIDF

# In[ ]:


# Réduction de dimension:

lsa = make_pipeline(TruncatedSVD(n_components=50), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(tfidf_matrix)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")


# In[ ]:


#Recherche du nombre de clusters optimum: méthode du coude

Sum_of_squared_distances =[]
K = range(1,40)
for k in K:
    km =KMeans(n_clusters =k)
    km =km.fit(X_lsa)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


# Application de kmeans après réduction de dimension

km = KMeans(
        n_clusters=10,
        max_iter=100,
        n_init=1,
        random_state=1)


# Fit the k-means object with tfidf_matrix or count_matrix
km.fit_transform(X_lsa)

clusters = km.labels_.tolist()

# Create a column cluster to denote the generated cluster for each tweet
df["cluster_lsa"] = clusters

# Display number of tweets per cluster 
df['cluster_lsa'].value_counts() 


# In[ ]:


plot_words_clusters(df,"content_lem", "cluster_lsa", 10, 30)


# # 6. Recherche de topics: LDA

# ## 6.1 LDA: First method (gensim+doc2bow)

# Sources:
# 
# https://www.kaggle.com/code/vukglisovic/classification-combining-lda-and-word2vec
# 
# https://towardsdatascience.com/lda-topic-modeling-with-tweets-deff37c0e131
# 
# https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know
# 
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

# In[ ]:


#Analyse statistique
df.describe()


# Certains tweets ne contiennent aucun mot après traitement: on les supprime.

# In[ ]:


df_non_null = df[df['nb_words_lem']>0]


# In[ ]:


all_words = [text.split() for text in df_non_null['content_lem']]
all_words = [y for x in all_words for y in x]
all_words_unique= list(set(all_words))


# In[ ]:


#all_words


# In[ ]:



print(f"Le corpus comporte {len(all_words_unique)} mots différents.")


# In[ ]:


#vocab = sorted(all_words_unique)
#print(vocab)


# In[ ]:


# Mots les plus fréquemment utilisés
word_freq = FreqDist(all_words)

#word_freq.most_common(30)


# In[ ]:


#retrieve word and count from FreqDist tuples

most_common_count = [x[1] for x in word_freq.most_common(30)]
most_common_word = [x[0] for x in word_freq.most_common(30)]

#create dictionary mapping of word count
top_30_dictionary = dict(zip(most_common_word, most_common_count))


# In[ ]:


wordcloud = WordCloud(colormap = 'Accent', background_color = 'black').generate_from_frequencies(top_30_dictionary)

#plot with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('top_30_cloud.png')

plt.show()


# In[ ]:


def token_text(text):
    tokens = text.split()
    return tokens


# In[ ]:


df_non_null['stem_tokens']=df_non_null['content_stem'].apply(lambda x : token_text(x))


# In[ ]:


col_tokens = df_non_null['stem_tokens']


# In[ ]:


col_tokens.head()


# In[ ]:


dictionary = Dictionary(documents=df_non_null['stem_tokens'].values)


# In[ ]:


print("Le dictionnaire comporte {} mots.".format(len(dictionary.values())))


# In[ ]:


dictionary.filter_extremes(no_above=0.75, no_below=1000)

dictionary.compactify()  # Reindexes the remaining words after filtering
print("Après suppression des extrêmes, il reste {} mots.".format(len(dictionary.values())))


# In[ ]:


print(f"Voici les identifiants des mots de dictionnary après suppression outliers:\n\n {dictionary.token2id}.")


# In[ ]:


#Bag of words
tweets_bow = [dictionary.doc2bow(tweet) for tweet in df_non_null['stem_tokens']]


# In[ ]:


#The output will contain a vector for each tweet, in the form of (word id, frequency of word occurrence in document)
# Les 3 premiers tweets:
tweets_bow[0:3]


# In[ ]:


# LDA: 5 topics

k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = dictionary,
                      random_state = 1,
                      passes=10)

tweets_lda.show_topics()


# Source:
# https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know

# In[ ]:


#gensimvis.prepare(tweets_lda, tweets_bow, dictionary)


# In[ ]:


# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=tweets_lda, texts=df_non_null['stem_tokens'], dictionary=dictionary, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

