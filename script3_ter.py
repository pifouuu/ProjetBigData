
# coding: utf-8

#this script generates the file classifications.txt based on the model of Logistic regression
#with advanced Feature Extraction & Transformation: Stemming & Cleaning,
# and Feature Selection with the Khi-deux test on bigrams


# Necessary imports, either for the script to operate 
# on the cluster or for further improvements

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

data,Y=lfp.loadLabeled("./data/train",100)
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

allFeatures=dfTrain.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(['bigramVectors','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[6]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics

print "Starting chi square test"
t0 = time()

labeledPoints = allFeatures.map(lambda row : LabeledPoint(row.label, row.bigramVectors)).cache()
chi = Statistics.chiSqTest(labeledPoints)

tt = time() - t0
print "Done in {} second".format(round(tt,3))


## In[20]:
#
#import pandas as pd
#pd.set_option('display.max_colwidth', 30)
#
#records = [(result.statistic, result.pValue) for result in chi]
#index=[revDict_broad.value[i] for i in range(len(revDict_broad.value))]
#chi_df = pd.DataFrame(data=records, index=index, columns=["Statistic","p-value"])
#
#chi_df.sort_values("p-value")


# In[17]:

print "Feature selection"
t0 = time()

biSelect = [revDict_broad.value[i] for i,bigram in enumerate(chi) if bigram.pValue <=0.5]
dictSelect = {}
for i,bigram in enumerate(biSelect):
    dictSelect[bigram]=i
dictSel_broad = sc.broadcast(dictSelect)

dfTrainSelect=dfTrain.map(partial(vectorizeBi,dico=dictSel_broad.value)).toDF(['selectedFeatures','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[18]:

from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


print "Fitting the classifier on selected features"
t0 = time()

string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
lr = LogisticRegression(featuresCol='selectedFeatures',labelCol='target_indexed',maxIter=30, regParam=0.01)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')

string_indexer_model = string_indexer.fit(dfTrainSelect)
dfTrainIndexed = string_indexer_model.transform(dfTrainSelect).cache()
lrModel = lr.fit(dfTrainIndexed)

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[19]:

print "Testing precision of the model"
t0 = time()

dfValidSelect=dfValid.map(partial(vectorizeBi,dico=dictSel_broad.value)).toDF(['selectedFeatures','label']).cache()
dfValidIndexed = string_indexer_model.transform(dfValidSelect).cache()
df_valid_pred = lrModel.transform(dfValidIndexed).cache()
res=evaluator.evaluate(df_valid_pred)
print res

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[12]:

import loadFiles as lf
print "Start loading  and preprocessing test data "
t0 = time()

test,names=lf.loadUknown('./data/test')
text_name=zip(test,names)
dfTest = sc.parallelize(text_name).toDF(['review','label']).cache()

dfTestPre=dfTest.map(preProcess).toDF(['words','label']).cache()
bigram = NGram(inputCol="words", outputCol="bigrams")
dfTestBi = bigram.transform(dfTestPre).cache()
finalDfSelect = dfTestBi.map(partial(vectorizeBi,dico=dictSel_broad.value)).toDF(['selectedFeatures','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[14]:

print "Classifying test data"
t0 = time()
list_predictions = lrModel.transform(finalDfSelect).collect()
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[16]:

output=file('./classifications_script3.txt','w')
for x in list_predictions:
	output.write('{}\t{}\n'.format(x.label, x.prediction))
output.close()
print "File classifications_script3.txt is written"






