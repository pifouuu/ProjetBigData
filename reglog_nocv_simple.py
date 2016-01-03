# coding: utf-8

#import packages
from pyspark import SparkContext
import loadFiles as lf
import numpy as np
import nltk
import loadFilesPartial as lfp
import transformers as tr
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SQLContext 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#create Sparkcontext
sc = SparkContext(appName="Simple App")

def Predict(name_text,dictionary,model):
	words=name_text[1].strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text[0], model.predict(SparseVector(len(dictionary),vector_dict)))

data,Y=lf.loadLabeled("./data/train")

print len(data)

labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)


from pyspark.sql import SQLContext 
def preProcess(doc):
    clean = doc.replace("<br /><br />"," ")
    return clean.lower()
rdd = labeledRdd.map(lambda doc : (preProcess(doc[0]),doc[1]))

sqlContext = SQLContext(sc)

df = sqlContext.createDataFrame(rdd, ['review', 'label'])
dfTrain, dfTest = df.randomSplit([0.8,0.2])

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='review', outputCol='words')
dfTrainTok = tokenizer.transform(dfTrain)

import itertools
lists=dfTrainTok.map(lambda r : r.review).collect()
dictWords=set(itertools.chain(*lists))
dictionaryWords={}
for i,word in enumerate(dictWords):
	dictionaryWords[word]=i

from pyspark.mllib.linalg import SparseVector
def vectorize(row,dico):
    vector_dict={}
    for w in row.words:
        if w in dico:
            vector_dict[dico[w]]=1
    return (row.label,SparseVector(len(dico),vector_dict))


from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField,DoubleType

schema = StructType([StructField('label',DoubleType(),True),StructField('Vectors',VectorUDT(),True)])


features=dfTrainTok.map(partial(vectorize,dico=dict_broad.value)).toDF(schema)

print "Features created"

from pyspark.ml.feature import StringIndexer

string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
string_indexer_model = string_indexer.fit(features)
featIndexed = string_indexer_model.transform(features)

print "labels indexed"

lr = LogisticRegression(featuresCol='Vectors', labelCol=string_indexer.getOutputCol())

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')

lr_model = lr.fit(featIndexed)

dfTestTok = tokenizer.transform(dfTest)
featuresTest=dfTestTok.map(partial(vectorize,dico=dict_broad.value)).toDF(schema)
testIndexed = string_indexer_model.transform(featuresTest)

df_test_pred = lr_model.transform(testIndexed)

res=evaluator.evaluate(df_test_pred)

print res

#test,names=lf.loadUknown('./data/test')
#name_text=zip(names,test)
##for each doc :(name,text):
##apply the model on the vector representation of the text
##return the name and the class
#predictions=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()

#output=file('./classifications.txt','w')
#for x in predictions:
#	output.write('%s\t%d\n'%x)
#output.close()
