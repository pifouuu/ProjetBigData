# coding: utf-8

#import packages
from pyspark import SparkContext
import loadFiles as lf
import numpy as np
import nltk
import loadFilesPartial as lfp
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SQLContext 
from pyspark.ml.feature import Tokenizer



#create Sparkcontext
sc = SparkContext(appName="Simple App")

data,Y=lfp.loadLabeled("./data/train",1000)
labeledData = zip(data,[y.item() for y in Y])

# CHANGE NUMBER OF PARTITIONS ?
# labeledRdd = sc.parallelize(labeledData, 16)
labeledRdd = sc.parallelize(labeledData)

def cleanLower(doc):
    return doc.replace("<br /><br />"," ").lower()
rdd = labeledRdd.map(lambda doc : (cleanLower(doc[0]),doc[1]))

print "Text is cleaned"


sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(rdd, ['review', 'label'])
dfTrain, dfTest = df.randomSplit([0.8,0.2])

print "Random split is done"


tokenizer = Tokenizer(inputCol='review', outputCol='reviews_words')
hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='reviews_tf')
idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="reviews_tfidf")
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
dt = DecisionTreeClassifier(featuresCol=idf.getOutputCol(), labelCol=string_indexer.getOutputCol(), maxDepth=10)

pipeline = Pipeline(stages=[tokenizer,
                            hashing_tf,
                            idf,
                            string_indexer,
                            dt])

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')

grid=(ParamGridBuilder()
     .baseOn([evaluator.metricName,'precision'])
     .addGrid(dt.maxDepth, [10,20])
     .build())
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,evaluator=evaluator)

print "Grid is build"
print "CV Estimator is defined"



cv_model = cv.fit(dfTrain)

print "Model is fitted"

df_test_pred = cv_model.transform(dfTest)

print "Labels are predicted"

print evaluator.evaluate(df_test_pred)












