
# coding: utf-8

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



# In[305]:

import loadFilesPartial as lfp
data,Y=lfp.loadLabeled("./data/train",10)
labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)


# In[306]:

def preProcess(doc):
    clean = doc.replace("<br /><br />"," ")
    return clean.lower()


# In[307]:

from time import time


# In[308]:

print "Start preprocessing all data"
t0 = time()
rdd = labeledRdd.map(lambda doc : (preProcess(doc[0]),doc[1]))
rdd.take(1)
tt = time() - t0
print "Data preprocessed in {} second".format(round(tt,3))


# In[313]:

print "Create dataframe"
t0 = time()
df = sqlContext.createDataFrame(rdd, ['review', 'label'])
print "Showing first example : "
print
print df.first()
tt = time() - t0
print
print "Dataframe created in {} second".format(round(tt,3))


# In[314]:

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='review', outputCol='words')
dfTok = tokenizer.transform(df)


# In[315]:

from pyspark.ml.feature import NGram
bigram = NGram(inputCol="words", outputCol="bigrams")
dfBigram = bigram.transform(dfTok)


# In[317]:

print "Start tokenizing, computing bigrams and splitting between test and train"
t0 = time()
dfTrain, dfTest = dfBigram.randomSplit([0.8,0.2])
dfTrain.take(1)
dfTest.take(1)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[318]:

print "Creation of the bigram dictionary"
t0 = time()
import itertools
lists=dfTrain.map(lambda r : r.bigrams).collect()
dictBigrams=list(set(itertools.chain(*lists)))
dictionaryBigrams={}
for i,word in enumerate(dictBigrams):
	dictionaryBigrams[word]=i
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[319]:

print "Broadcasting the dictionary and the reverse dictionary"
t0 = time()
dict_broad=sc.broadcast(dictionaryBigrams)
revDict_broad = sc.broadcast(dictBigrams)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[320]:

from pyspark.mllib.linalg import SparseVector
def vectorizeBi(row,dico):
    vector_dict={}
    for w in row.bigrams:
        if w in dico:
            vector_dict[dico[w]]=1
    return (row.label,SparseVector(len(dico),vector_dict))


# In[321]:

from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField,DoubleType

schema = StructType([StructField('label',DoubleType(),True),StructField('bigramVectors',VectorUDT(),True)])


# In[322]:

from functools import partial
print "Converting bigrams to sparse vectors in a dataframe for the train set"
t0 = time()
features=dfTrain.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(schema)
features.take(1)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[323]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
print "Computing the chi vector"
t0 = time()
labeledPoints = features.map(lambda row : LabeledPoint(row.label, row.bigramVectors))
chi = Statistics.chiSqTest(labeledPoints)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[324]:

print "Starting bigram selection,broadcasting the newly created bigram dictionary"
t0 = time()
biSelect = [revDict_broad.value[i] for i,bigram in enumerate(chi) if bigram.pValue <=0.3]
dictSelect = {}
for i,bigram in enumerate(biSelect):
    dictSelect[bigram]=i
dictSel_broad = sc.broadcast(dictSelect)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[326]:

print "Creating sparse vectors for all data based on this new dictionary"
t0 = time()
dfTrainSelect=dfTrain.map(partial(vectorizeBi,dico=dictSel_broad.value)).toDF(schema)
dfTestSelect=dfTest.map(partial(vectorizeBi,dico=dictSel_broad.value)).toDF(schema)
dfTrainSelect.take(1)
dfTestSelect.take(1)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[328]:

from pyspark.ml.feature import StringIndexer
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
string_indexer_model = string_indexer.fit(dfTrainSelect)
dfTrainIndexed = string_indexer_model.transform(dfTrainSelect)


# In[329]:

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol='bigramVectors', labelCol='target_indexed', maxDepth=10)


# In[330]:

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')


# In[331]:

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
grid=(ParamGridBuilder()
     .baseOn([evaluator.metricName,'precision'])
     .addGrid(dt.maxDepth, [10,20])
     .build())
cv = CrossValidator(estimator=dt, estimatorParamMaps=grid,evaluator=evaluator)


# In[332]:

print "Fitting the decision tree on selected features"
t0 = time()
cv_model = cv.fit(dfTrainIndexed)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[302]:

pr
dfTestIndexed = string_indexer_model.transform(dfTestSelect)
df_test_pred = cv_model.transform(dfTestIndexed)
res=evaluator.evaluate(df_test_pred)
print res 




