import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gensim
from gensim.models import word2vec  
from sklearn.manifold import TSNE  
import re  
import nltk  
from nltk.corpus import stopwords  
from nltk.stem.porter import *  
stemmer = PorterStemmer()  
from nltk.stem import PorterStemmer
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark.sql.functions as sf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkContext, SparkConf, HiveContext
from pyspark.sql.functions import col, asc, desc
from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans 
from pyspark.ml.evaluation import ClusteringEvaluator 
from pyspark.ml.feature import StandardScaler 
class Model_Clasters:
    """
    Class for creating w0rd2vec model of a text subset in your data and clasterisation. You can choose clasterisation method.
    Available clasterisation methods:
    ///
    Class uses pyspark.
    For choosing number of clasters the silhouette method is used.
    You can see which word belongs to which claster by list under it. 0 is for red claster, 1 is for orange and etc.
    """
    def __init__(self,filename:str,sub:str):
        self.data =  pd.DataFrame(pd.read_csv(filename,delimiter = ','))
        self.new_values = []
        self.accordance = {}
        self.data = self.data.dropna(subset=sub)
        proc = [self.rewiew_to(text) for text in self.data[sub]]
        res = self.str_to_lst(proc)
        self.model = word2vec.Word2Vec(res, window=5, min_count=40, workers=4)
        
        labeles = []
        tokens = []
        for word in self.model.wv.key_to_index:
            tokens.append(self.model.wv[word])
            labeles.append(word)
        tokens = np.array(tokens)
        tsne_model = TSNE(perplexity=40, n_components=2,init='pca', max_iter=2500)
        self.new_values = tsne_model.fit_transform(tokens)
        for l,t in zip(labeles,self.new_values):
            self.accordance[l] = t
            
        self.ac = pd.DataFrame.from_dict(self.accordance,orient="index")
        
        conf = SparkConf().setAppName('spark_dlab_app') 
        spark = SparkSession.builder.config(conf = conf).enableHiveSupport().getOrCreate()
        
        nv = pd.DataFrame(self.new_values)
        nv.to_csv("Vectors.csv",header = False,index = False)
        nd  = spark.read.csv("Vectors.csv", sep=",",inferSchema=True)
        
        sc = SparkContext.getOrCreate()
        sqlContext = SQLContext(sc)
        vec_assembler = VectorAssembler(inputCols = nd.columns, 
                                        outputCol='features') 
        self.final_data = vec_assembler.transform(nd) 
        self.final_data.select('features')
        scaler = StandardScaler(inputCol="features",  
                        outputCol="scaledFeatures",  
                        withStd=True,  
                        withMean=False) 
  
        scalerModel = scaler.fit(self.final_data) 
        self.final_data = scalerModel.transform(self.final_data) 
        pf_final = self.final_data.toPandas()
        self.final_data.select('scaledFeatures')
        s = self.silhouette()
        self.Clasterisation(s)        

    def rewiew_to(self,raw):
        letters_only = re.sub("[^a-zA-Z]", " ",raw)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        m_words = [w for w in words if not w in stops]
        lemma = nltk.wordnet.WordNetLemmatizer()
        singles = [lemma.lemmatize(word) for word in m_words]
        return (" ".join(singles))   
        
    def str_to_lst(self,text):
        res = []
        for elem in text:
            res.append([w for w in elem.split()])
        return res
        
    def show(self):
        x = []
        y = []
        for value in self.new_values:
            x.append(value[0])
            y.append(value[1])
        plt.figure(figsize=(8,8))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labeles[i],xy = (x[i],y[i]),xytext = (5,2),
            textcoords= 'offset points',ha = 'right',va = 'bottom')
        plt.show()
       
    def silhouette(self,claster_method = ""):
        silhouette_score=[] 
        evaluator = ClusteringEvaluator(predictionCol='prediction', 
                                        featuresCol='scaledFeatures',  
                                        metricName='silhouette',  
                                        distanceMeasure='squaredEuclidean') 
        s = []
        for i in range(2,10): 
            kmeans=KMeans(featuresCol='scaledFeatures', k=i) 
            model=kmeans.fit(self.final_data) 
            predictions=model.transform(self.final_data) 
            score=evaluator.evaluate(predictions) 
            s.append([i,score])
            silhouette_score.append(score) 
            print('Silhouette Score for k =',i,'is',score)
        s = (min(s[4:],key = lambda s: s[1]))[0]
        plt.plot(range(2,10),silhouette_score) 
        plt.xlabel('k') 
        plt.ylabel('silhouette score') 
        plt.title('Silhouette Score') 
        plt.show()
        return s
    def Clasterisation(self,claster_num,claster_method = ""):
        kmeans = KMeans(featuresCol='scaledFeatures',k=claster_num) 
        model = kmeans.fit(self.final_data)
        predictions = model.transform(self.final_data)
        x = self.final_data.select("_c0").toPandas()
        y = self.final_data.select("_c1").toPandas()
        x1 = list(x["_c0"])
        y1 = list(y["_c1"])
        pred = predictions.select("prediction").toPandas()
        pred = list(pred["prediction"])
        words = list(self.accordance.keys())
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        for i,word in zip(range(130,len(x1)-150),words):
            ax.annotate(word,(x1[i],y1[i]),xytext = (x1[i]+0.3,y1[i]+0.3))
        
        plt.scatter(x1, y1, c = pred, cmap="rainbow")
        plt.show()
        self.ac["Predictions"] = pred
        clas = list(zip(words,pred))
        clas = sorted(clas,key = lambda clas:clas[1])
        centers = model.clusterCenters() 
        print("Cluster Centers: ") 
        for center in centers: 
            print(center)
        print(clas)