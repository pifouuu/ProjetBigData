
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

# In[1]:

import loadFilesPartial as lfp
data,Y=lfp.loadLabeled("./data/train",10)
labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)


# In[2]:

import nltk
def preProcess(doc):
    clean = doc[0].replace("<br /><br />"," ")
    tok = nltk.tokenize.wordpunct_tokenize(clean)
    low = [word.lower() for word in tok]
    return low,doc[1]


# In[3]:

from time import time


# In[4]:

print "Start preprocessing train data"
t0 = time()
rdd = labeledRdd.map(preProcess)
rdd.take(1)
tt = time() - t0
print "Train data preprocessed in {} second".format(round(tt,3))



# In[5]:

print "Create dataframe"
t0 = time()
df = sqlContext.createDataFrame(rdd,['words', 'label'])
df.show()
tt = time() - t0
print "Dataframe created in {} second".format(round(tt,3))


# In[6]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
import transformers as tr
posTagger = tr.NLTKPosTagger(
    inputCol="words", outputCol="tags")
print "Compute tags"
t0 = time()
dfTags = posTagger.transform(df)
dfTags.show()
tt = time() - t0
print "Tags computed in {} second".format(round(tt,3))


# In[7]:

from pyspark.ml.feature import NGram
trigram = NGram(n=3,inputCol="tags", outputCol="tagTrigrams")
t0 = time()
dfTriAux = trigram.transform(dfTags)
trigram.setInputCol("words")
trigram.setOutputCol("wordTrigrams")
dfTri = trigram.transform(dfTriAux)
dfTri.show()
tt = time() - t0
print "Trigrams created in {} second".format(round(tt,3))


# In[8]:

dfTrain, dfTest = dfTri.randomSplit([0.8,0.2])


# In[9]:

import itertools
lists=dfTrain.map(lambda r : r.words).collect()
dictUnigrams=list(set(itertools.chain(*lists)))
dictionaryUni={}
for i,word in enumerate(dictUnigrams):
	dictionaryUni[word]=i


# In[10]:

authorizedTrigrams = set(['NOUN VERB ADJ'])
auth_broad = sc.broadcast(authorizedTrigrams)


# In[11]:

def retrieveTrigrams(row,auth):
    tagTrigrams = row.tagTrigrams
    wordTrigrams = row.wordTrigrams
    selectedTri = []
    for i, trigram in enumerate(wordTrigrams):
        if tagTrigrams[i] in auth:
            selectedTri.append(trigram)
    return selectedTri


# In[12]:

from functools import partial
print "Creation of the trigram dictionary"
t0 = time()
import itertools
lists=dfTrain.map(partial(retrieveTrigrams,auth=auth_broad.value)).collect()
dictTrigrams=list(set(itertools.chain(*lists)))
dictionaryTrigrams={}
for i,word in enumerate(dictTrigrams):
	dictionaryTrigrams[word]=i
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[13]:

dictTriRev_broad = sc.broadcast(dictTrigrams)
dictTri_broad = sc.broadcast(dictionaryTrigrams)
dictRev_broad = sc.broadcast(dictUnigrams)
dict_broad = sc.broadcast(dictionaryUni)


# In[14]:

from pyspark.mllib.linalg import SparseVector
def vectorize(row,dicoUni,dicoTri):
    vector_dict={}
    length = len(dicoUni)
    for w in row.words:
        if w in dicoUni:
            vector_dict[dicoUni[w]]=1
    for tri in row.wordTrigrams:
        if tri in dicoTri:
            vector_dict[dicoTri[tri]+length]=1
    return (row.label,SparseVector(length+len(dicoTri),vector_dict))


# In[15]:

from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField,DoubleType,ArrayType,StringType
t = ArrayType(StringType())
schema = StructType([StructField('label',DoubleType(),True),                     StructField('featureVectors',VectorUDT(),True)])


# In[16]:

print "Creating feature vectors"
t0 = time()
dfTrainVec=dfTrain.map(partial(vectorize,dicoUni=dict_broad.value,dicoTri=dictTri_broad.value)).toDF(schema)
dfTestVec=dfTest.map(partial(vectorize,dicoUni=dict_broad.value,dicoTri=dictTri_broad.value)).toDF(schema)
tt = time() - t0
print "Dataframe created in {} second".format(round(tt,3))


# In[19]:

print "Indexing labels"
t0 = time()
from pyspark.ml.feature import StringIndexer
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
string_indexer_model = string_indexer.fit(dfTrainVec)
dfTrainIdx = string_indexer_model.transform(dfTrainVec)
dfTrainIdx.take(1)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[20]:

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol='featureVectors', labelCol='target_indexed', maxDepth=10)


# In[21]:

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')


# In[22]:

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
grid=(ParamGridBuilder()
     .baseOn([evaluator.metricName,'precision'])
     .addGrid(dt.maxDepth, [10,20])
     .build())
cv = CrossValidator(estimator=dt, estimatorParamMaps=grid,evaluator=evaluator)


# In[ ]:

print "Fitting the decision tree"
t0 = time()
cv_model = cv.fit(dfTrainIdx)
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[113]:

print "Testing the decision tree on test data after indexing the latter"
t0 = time()
dfTestIdx = string_indexer_model.transform(dfTestVec)
df_test_pred = cv_model.transform(dfTestIdx)
res=evaluator.evaluate(df_test_pred)
print res 
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[116]:

get_ipython().system(u'jupyter nbconvert --to script script3.ipynb')

