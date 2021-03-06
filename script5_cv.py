
# coding: utf-8

#this script carries out a cross validation on the model of Logistic regression
#with advanced Feature Extraction & Transformation: Stemming & Cleaning, Stopword Removal
# and Feature Selection with bi-grams

# Necessary imports, either for the script to operate 
# on the cluster or for further improvements

from pyspark import SparkContext
import loadFiles as lf
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
#cross validation
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SQLContext 
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.linalg import Vectors

from pyspark import SparkContext
import loadFiles as lf
import numpy as np
import nltk
from pyspark.sql import SQLContext 

#create Spark context and SQL context
sc = SparkContext(appName="Simple App")
sqlContext = SQLContext(sc)

# In[1]:

import loadFilesPartial as lfp
from time import time

print "Start loading all data to a dataframe"
t0 = time()

data,Y=lf.loadLabeled("./data/train")
labeledData = zip(data,[y.item() for y in Y])
df = sc.parallelize(labeledData,numSlices=16).toDF(['review','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[3]:

from pyspark.ml.feature import NGram
import itertools
import nltk


print "Start preprocessing all data"
t0 = time()

def preProcess(doc):
    clean = doc[0].replace("<br /><br />"," ")
    tok = nltk.tokenize.wordpunct_tokenize(clean)
    low = [word.lower() for word in tok]
    return low,doc[1]

bigram = NGram(inputCol="words", outputCol="bigrams")

dfPre=df.map(preProcess).toDF(['words','label']).cache()
dfTrain, dfValid = bigram.transform(dfPre).randomSplit([0.8,0.2])
dfTrain.cache()
dfValid.cache()

lists=dfTrain.map(lambda r : r.bigrams).collect()
dictBigrams=list(set(itertools.chain(*lists)))
dictionaryBigrams={}
for i,word in enumerate(dictBigrams):
	dictionaryBigrams[word]=i
    
dict_broad=sc.broadcast(dictionaryBigrams)
revDict_broad=sc.broadcast(dictBigrams)

tt = time() - t0
print "Data preprocessed in {} second".format(round(tt,3))


# In[5]:

from pyspark.mllib.linalg import SparseVector
from functools import partial

print "Feature creation on training set"
t0 = time()

def vectorizeBi(row,dico):
    vector_dict={}
    for w in row.bigrams:
        if w in dico:
            vector_dict[dico[w]]=1
    return (SparseVector(len(dico),vector_dict),row.label)

dfBigram=dfTrain.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(['features','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))



# In[18]:

from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


print "Fitting the classifier on bigram features"
t0 = time()

string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
lr = LogisticRegression(featuresCol='bigramVectors',labelCol='target_indexed',maxIter=30, regParam=0.01)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')

string_indexer_model = string_indexer.fit(dfBigram)
dfTrainIndexed = string_indexer_model.transform(dfBigram).cache()

lr = LogisticRegression()   #choose the model
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()   
#la grille est construite pour trouver le meilleur parametre 'alpha' pour le terme de regularisation du modele: c'est un 'elastic Net'
#max.iter vaut 30 par defaut, on pourrait changer sa valeur
#on va donc essayer 30 valeur entre 0 et 1
#alpha=0 c'est une regularisation L2, 
#alpha=1, c'est une regularisation L1
print "Cross validation debut"

evaluator = BinaryClassificationEvaluator() #choose the evaluator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator) #perform the cross validation and keeps the best value of maxIter
cvModel = cv.fit(dfBigram)   #train the model on the whole training set
#resultat=evaluator.evaluate(cvModel.transform(dfTest))  #compute the percentage of success on test set
#print "Pourcentage de bonne classification(0-1): ",resultat

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[19]:

print "Testing precision of the model"
t0 = time()

dfValidSelect=dfValid.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(['features','label']).cache()
#dfValidIndexed = string_indexer_model.transform(dfValidSelect).cache()
df_valid_pred = cvModel.transform(dfValidSelect).cache()
res=evaluator.evaluate(df_valid_pred)
print res

tt = time() - t0
print "Done in {} second".format(round(tt,3))







