
# coding: utf-8

# # Súper Pipeline
# ## **RANDOM ENGINEER PIPELINE**
# 
# 
# 
# ---
# 
# 
# 

# ## Preparación

# ### Importar librerías

# In[ ]:


# Agregar variables geográficas
# Optimizar clases para generar pipelines más limpios
# Agregar hiperparámetro para definir si se usa grid o randomsearch


# In[2]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, IsolationForest


# In[3]:


from nltk import word_tokenize          
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
class StemTokenizer_sno(object):
    def __init__(self):
        self.sno = SnowballStemmer('spanish')
    def __call__(self, doc):
        return [self.sno.stem(t) for t in word_tokenize(doc)]


# from nltk.corpus import stopwords
# 
# import gensim
# from gensim import corpora, models
# from gensim.utils import simple_preprocess
# 
# def preprocesamiento(docs):
#     
#     # Generamos un listado de tokens por documento
#     stop = stopwords.words('spanish')
#     tokens = [[word for word in gensim.utils.simple_preprocess(str(doc), deacc=True) if word not in stop] for doc in docs]
#     
#     # Construimos los modelos de bigramas y trigramas
#     bigram = gensim.models.Phrases(tokens, min_count=2, threshold=50)
#     trigram = gensim.models.Phrases(bigram[tokens], threshold=50)
#     bigram_mod = gensim.models.phrases.Phraser(bigram)
#     trigram_mod = gensim.models.phrases.Phraser(trigram)
#     trigrams = [trigram_mod[bigram_mod[t]] for t in tokens]
#     
#     # Generamos diccionario y corpus
#     dictionary = corpora.Dictionary(trigrams)
#     corpus = [dictionary.doc2bow(n) for n in trigrams]
#     
#     return dictionary, corpus

# In[4]:


from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X[self.columns])
        else:
            raise TypeError("Este Transformador solo funciona en DF de Pandas")
    
    def fit(self, X, *_):
        return self


# In[5]:


#Optimizar la creación de la serie
class TextMixer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, discriminate=True, empty_placeholder='EMPTY'):
        self.columns = columns
        self.discriminate = discriminate
        self.empty_placeholder = empty_placeholder
        
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            if self.discriminate == True:
                for i in self.columns:
                    X.loc[:, i] = X.loc[:, i].fillna('EMPTY_%s' % (X[i].name))
            else:
                for i in self.columns:
                    X.loc[:, i] = X.loc[:, i].fillna('EMPTY')
            X['columna_de_texto_random'] = X[self.columns].apply(lambda x: ' '.join(x), axis=1)
            return X['columna_de_texto_random']
        else:
            raise TypeError("Este Transformador solo funciona en DF de Pandas")
    
    def fit(self, X, *_):
        return self


# In[10]:


# Más elegante, ver si funciona.
'''class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X[self.columns], columns = self.columns)
        else:
            raise TypeError("Este Transformador solo funciona en DF de Pandas")
    
    def fit(self, X, *_):
        self.dummies_cols_ = pd.get_dummies(X[self.columns], columns = self.columns).columns'''
        return self

# In[11]:


class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.dict_dummies = {}
        
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            if self.dict_dummies == {}:
                return pd.get_dummies(X[self.columns])
            else:
                dummy_df = pd.DataFrame()
                for col in self.dict_dummies.keys():
                    for dummy in self.dict_dummies[col]:
                        b = pd.Series(X[col]).apply(lambda x: 1 if x==dummy else 0)
                        dummy_df[dummy] = b
                return dummy_df
        else:
            raise TypeError("Este Transformador solo funciona en DF de Pandas")
    
    def fit(self, X, *_):
        for col in self.columns:
            a = pd.get_dummies(pd.Series(X[col]), drop_first=True).columns.tolist()
            self.dict_dummies[col] = a
        return self


# # Pipeline

# In[36]:


from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer


# In[ ]:


#Columnas de texto. AGREGAR MEZCLAR DESCRIPCIONES
cs_text = SeriesSelector('nueva_descripcion')
vect = TfidfVectorizer(encoding='utf-8', stop_words=stopwords.words('spanish'), min_df=1, max_df=1.0,
                                         tokenizer=StemTokenizer_sno(), ngram_range=(1, 2))

pipe_text = make_pipeline(cs_text, vect)


# In[ ]:


#Columnas Dummies
dummies_cols = ['tipodepropiedad', 'ciudad'] #Columnas a convertir en dummy
pipe_dummies = GetDummiesTransformer(dummies_cols)


# In[ ]:


#Columnas Numéricas
num_cols = [] #Idem dummies

train_df[num_cols] = train_df[num_cols].str.extract(r'(\d*\.?\d*)', expand=False).astype(float)

#Si no anda con todas hacer de a una
'''for col in num_cols:
    train_df[col] = train_df[col].str.extract(r'(\d*\.?\d*)', expand=False).astype(float)
'''


# In[ ]:


#Columnas numéricas

select_col = ColumnSelector(num_cols)
impute=Imputer('mean')
pipe_num=make_pipeline(select_col, impute)


# In[ ]:


preprocessing_pipe = make_union(pipe_num, pipe_dummies, pipe_desc, n_jobs=-1)


# In[ ]:


#corremos absolutamente todo y ya tenemos nuestro df preprocesado
# preprocessing_pipe.fit_transform(train_df)


# In[ ]:


# Celda para probar tiempo del vectorizer
# import time
# start = time.time()

# pipe_desc.fit_transform(train)

# elapsed_time = (time.time() - start) / 60

# print(str(elapsed_time)+' minutos')


# In[ ]:


model = lgb.LGBMRegressor()


# In[ ]:


from tempfile import mkdtemp
cachedir = mkdtemp()

# Pipe que va desde preprocesamiento hasta predicción
final_pipe = make_pipeline(preprocessing_pipe, model, memory=cachedir)


# In[37]:


final_pipe.steps


# In[ ]:


params_clasif = [{'lgbmregressor': [RandomForestRegressor()], 'lgbmregressor__n_estimators': [10,50,100,400], 
           'lgbmregressor__max_depth': [1,2,3,4,5,6,7],'lgbmregressor__max_leaf_nodes' : [5,10,15,20,30],
           'lgbmregressor__min_samples_leaf' : [1,2,3,4,5,10,15,20],
           'lgbmregressor__min_samples_split' : [5,6,7,8,9,10,20]},    
        {'lgbmregressor': [lgb.LGBMRegressor()], 'lgbmregressor__n_estimators':[100, 500, 1000] , 
        'lgbmregressor__max_depth': [5, 10, 15], 
        'lgbmregressor__min_samples_split': [5, 10, 15],
        'lgbmregressor__min_samples_leaf': [5, 10, 15],
        'lgbmregressor__learning_rate':[0.001, 0.001, 0.1, 1.0]},
        {'lgbmregressor': [ExtraTreesRegressor()], 'lgbmregressor__n_estimators': [10,50,100,400],
         'lgbmregressor__max_depth': [1,2,3,4,5,6,7],'lgbmregressor__max_leaf_nodes' : [5,10,15,20,30],
         'lgbmregressor__min_samples_leaf' : [1,2,3,4,5,10,15,20],
         'lgbmregressor__min_samples_split' : [5,6,7,8,9,10,20]}]


# In[ ]:


params_regr = [{'catboostclassifier': [GaussianNB()]}, 
          {'catboostclassifier': [SVC()], 'catboostclassifier__C': [1, 10, 100, 1000],
           'catboostclassifier__gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
           'catboostclassifier__kernel': ['rbf']},
        {'catboostclassifier': [KNeighborsClassifier()], 'catboostclassifier__n_neighbors': range(1,15),
        'catboostclassifier__weights' : ['uniform', 'distance']},
        {'catboostclassifier': [LogisticRegression()], 'catboostclassifier__C':[0.1, 1, 10,100]},
        {'catboostclassifier':[DecisionTreeClassifier()], 'catboostclassifier__max_depth': [1,2,3,4,5,6,7],
         'catboostclassifier__max_leaf_nodes' : [5,10,15,20,30],'catboostclassifier__min_samples_leaf' : [1,2,3,4,5,10,15,20],
         'catboostclassifier__min_samples_split' : [5,6,7,8,9,10,20]},
        {'catboostclassifier': [RandomForestClassifier()], 'catboostclassifier__n_estimators': [100,400,600],
         'catboostclassifier__max_depth': [1,2,3,4,5,6,7],'catboostclassifier__max_leaf_nodes' : [5,10,15,20,30],
         'catboostclassifier__min_samples_leaf' : [1,2,3,4,5,10,15,20],'catboostclassifier__min_samples_split' : [5,6,7,8,9,10,20]},    
        {'catboostclassifier': [LGBMClassifier()], 'catboostclassifier__n_estimators':[100, 500, 1000] , 
        'catboostclassifier__max_depth': [5, 10, 15], 
        'catboostclassifier__min_samples_split': [5, 10, 15],
        'catboostclassifier__min_samples_leaf': [5, 10, 15],
        'catboostclassifier__learning_rate':[0.001, 0.001, 0.1, 1.0]},
        {'catboostclassifier': [ExtraTreesClassifier()], 'catboostclassifier__n_estimators': [100,400,600], 
         'catboostclassifier__max_depth': [1,2,3,4,5,6,7],'catboostclassifier__max_leaf_nodes' : [5,10,15,20,30],
         'catboostclassifier__min_samples_leaf' : [1,2,3,4,5,10,15,20],'catboostclassifier__min_samples_split' : [5,6,7,8,9,10,20]}]


# In[ ]:


# Hace un gridsearch y determina la mejor combinación de modelos para cada uno de los posibles regresores
grids = []
for i in range(len(params)):
    gs = GridSearchCV(estimator=final_pipe, param_grid=params[i], scoring='r2', cv=5, n_jobs=-1)
    fit = gs.fit(train_df, train_df.precioAjustadoPorInflacion)
    grids.append(fit)

params_dict = []
for i in grids:
    a = {'score':i.best_score_, 'model': i.best_estimator_, 'parameters': i.best_params_}
    params_dict.append(a)
model_df = pd.DataFrame(params_dict)

model = model_df.loc[model_df['score'].idxmax, 'model']
model_df

