
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

# In[19]:

import loadFilesPartial as lfp
from time import time

print "Start loading all data to a dataframe"
t0 = time()

data,Y=lfp.loadLabeled("./data/train",100)
labeledData = zip(data,[y.item() for y in Y])
df = sc.parallelize(labeledData).toDF(['review','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[20]:

from pyspark.ml.feature import NGram
import itertools
import nltk
from functools import partial
from pyspark.sql.types import StructType, StructField,DoubleType,ArrayType,StringType


print "Start preprocessing all data"
t0 = time()

def preProcess(doc):
    clean = doc.review.replace("<br /><br />"," ")
    tok = nltk.tokenize.wordpunct_tokenize(clean)
    tags = nltk.pos_tag(tok,tagset='universal')
    low = [word.lower() for word in tok]
    return low,zip(*tags)[1],doc.label

schema = StructType([StructField('words',ArrayType(StringType()),True), StructField('tags',ArrayType(StringType()),True), StructField('label',DoubleType())])

dfPre=df.map(preProcess).toDF(schema).cache()
trigram = NGram(n=3,inputCol="tags", outputCol="tagTrigrams")
dfTriAux = trigram.transform(dfPre).cache()
trigram.setInputCol("words")
trigram.setOutputCol("wordTrigrams")
dfTri = trigram.transform(dfTriAux).cache()

dfTrain, dfValid = dfTri.randomSplit([0.8,0.2])


lists=dfTrain.map(lambda r : r.words).collect()
dictUnigrams=list(set(itertools.chain(*lists)))
dictionaryUni={}
for i,word in enumerate(dictUnigrams):
	dictionaryUni[word]=i

dict_broad = sc.broadcast(dictionaryUni)

authorizedTrigrams = set(['NOUN VERB ADJ'])
auth_broad = sc.broadcast(authorizedTrigrams)

def retrieveTrigrams(row,auth):
    tagTrigrams = row.tagTrigrams
    wordTrigrams = row.wordTrigrams
    selectedTri = []
    for i, trigram in enumerate(wordTrigrams):
        if tagTrigrams[i] in auth:
            selectedTri.append(trigram)
    return selectedTri

lists=dfTrain.map(partial(retrieveTrigrams,auth=auth_broad.value)).collect()
dictTrigrams=list(set(itertools.chain(*lists)))
dictionaryTrigrams={}
for i,word in enumerate(dictTrigrams):
	dictionaryTrigrams[word]=i

dictTri_broad = sc.broadcast(dictionaryTrigrams)


tt = time() - t0
print "Data preprocessed in {} second".format(round(tt,3))


# In[21]:

from pyspark.mllib.linalg import SparseVector
from functools import partial

print "Feature creation on training set"
t0 = time()

def vectorize(row,dicoUni,dicoTri):
    vector_dict={}
    length = len(dicoUni)
    for w in row.words:
        if w in dicoUni:
            vector_dict[dicoUni[w]]=1
    for tri in row.wordTrigrams:
        if tri in dicoTri:
            vector_dict[dicoTri[tri]+length]=1
    return (SparseVector(length+len(dicoTri),vector_dict),row.label)

dfTrainSelect=dfTrain.map(partial(vectorize, dicoUni=dict_broad.value, dicoTri=dictTri_broad.value)).toDF(['selectedFeatures','label']).cache()
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[22]:

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


# In[25]:

print "Testing precision of the model"
t0 = time()

dfValidSelect=dfValid.map(partial(vectorize, dicoUni=dict_broad.value, dicoTri=dictTri_broad.value)).toDF(['selectedFeatures','label']).cache()
dfValidIndexed = string_indexer_model.transform(dfValidSelect)
df_valid_pred = lrModel.transform(dfValidIndexed).cache()
res=evaluator.evaluate(df_valid_pred)
print res

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[28]:

import loadFiles as lf
print "Start loading  and preprocessing test data "
t0 = time()

test,names=lf.loadUknown('./data/test')
text_name=zip(test[1:2000],names[1:2000])
dfTest = sc.parallelize(text_name).toDF(['review','label']).cache()

schema2 = StructType([StructField('words',ArrayType(StringType()),True), StructField('tags',ArrayType(StringType()),True), StructField('label',StringType())])

dfTestPre=dfTest.map(preProcess).toDF(schema2).cache()
trigram = NGram(n=3,inputCol="words", outputCol="wordTrigrams")
dfTestTri = trigram.transform(dfTestPre).cache()
dfTestSelect=dfTestTri.map(partial(vectorize, dicoUni=dict_broad.value, dicoTri=dictTri_broad.value)).toDF(['selectedFeatures','label']).cache()

tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[29]:

dfTestSelect.show()


# In[14]:

print "Classifying test data"
t0 = time()
list_predictions = lrModel.transform(dfTestSelect).collect()
tt = time() - t0
print "Done in {} second".format(round(tt,3))


# In[ ]:

output=file('./classifications_script3.txt','w')
for x in list_predictions:
	output.write('{}\t{}\n'.format(x.label, x.prediction))
output.close()
print "File classifications_script3.txt is written"


# In[25]:

get_ipython().system(u'jupyter nbconvert --to script script4.ipynb --output script4_bis')


# In[ ]:



