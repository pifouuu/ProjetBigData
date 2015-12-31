
# coding: utf-8

import loadFilesPartial as lfp
from pyspark import SparkContext

#create Sparkcontext
sc = SparkContext(appName="Simple App")
data,Y=lfp.loadLabeled("./data/train",10)
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

print "Clean train and test set created"


from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='review', outputCol='words')
dfTrainTok = tokenizer.transform(dfTrain)

print "Tokens computed"


from pyspark.ml.feature import NGram
bigram = NGram(inputCol="words", outputCol="bigrams")
dfBigram = bigram.transform(dfTrainTok)

print "Bigrams computed"


import itertools
lists=dfBigram.map(lambda r : r.bigrams).collect()
dictBigrams=set(itertools.chain(*lists))
dictionaryBigrams={}
for i,word in enumerate(dictBigrams):
	dictionaryBigrams[word]=i

print "Dictionary created"


dict_broad=sc.broadcast(dictionaryBigrams)


from pyspark.mllib.linalg import SparseVector
def vectorizeBi(row,dico):
    vector_dict={}
    for w in row.bigrams:
        if w in dico:
            vector_dict[dico[w]]=1
    return (row.label,SparseVector(len(dico),vector_dict))


from pyspark.mllib.linalg import VectorUDT
from pyspark.sql.types import StructType, StructField,DoubleType

schema = StructType([StructField('label',DoubleType(),True),StructField('bigramVectors',VectorUDT(),True)])


features=dfBigram.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(schema)

print "Features from bigrams created"

from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier

string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
string_indexer_model = string_indexer.fit(features)
featIndexed = string_indexer_model.transform(features)

print "labels indexed"


dt = DecisionTreeClassifier(featuresCol='bigramVectors', labelCol=string_indexer.getOutputCol(), maxDepth=10)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')


from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
grid=(ParamGridBuilder()
     .baseOn([evaluator.metricName,'precision'])
     .addGrid(dt.maxDepth, [10,20])
     .build())
cv = CrossValidator(estimator=dt, estimatorParamMaps=grid,evaluator=evaluator)

from time import time
print "Start fitting"
t0 = time()
cv_model = cv.fit(featIndexed)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))

print "Start preprocessing test data"
t0 = time()

dfTestTok = tokenizer.transform(dfTest)
dfTestBigram = bigram.transform(dfTestTok)
featuresTest=dfTestBigram.map(partial(vectorizeBi,dico=dict_broad.value)).toDF(schema)
testIndexed = string_indexer_model.transform(featuresTest)

tt = time() - t0
print "Test data preprocessed in {} seconds".format(round(tt,3))

print "Start classifying test data"
t0 = time()
df_test_pred = cv_model.transform(testIndexed)
tt = time() - t0
print "Test data classified in {} seconds".format(round(tt,3))

res=evaluator.evaluate(df_test_pred)


print "Test data score : {}".format(round(res,3))






