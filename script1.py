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

#create Sparkcontext
sc = SparkContext(appName="Simple App")

data,Y=lf.loadLabeled("./data/train")
labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)

def cleanLower(doc):
    return doc.replace("<br /><br />"," ").lower()
rdd = labeledRdd.map(lambda doc : (cleanLower(doc[0]),doc[1]))

sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(rdd, ['review', 'label'])
dfTrain, dfTest = df.randomSplit([0.8,0.2])

tokenizerNoSw = tr.NLTKWordPunctTokenizer(
    inputCol="review", outputCol="wordsNoSw",  
    stopwords=set(nltk.corpus.stopwords.words('english')))
hashing_tf = HashingTF(inputCol=tokenizerNoSw.getOutputCol(), outputCol='reviews_tf')
idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="reviews_tfidf")
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
dt = DecisionTreeClassifier(featuresCol=idf.getOutputCol(), labelCol=string_indexer.getOutputCol(), maxDepth=10)

pipeline = Pipeline(stages=[tokenizerNoSw,
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

cv_model = cv.fit(dfTrain)
df_test_pred = cv_model.transform(dfTest)
evaluator.evaluate(df_test_pred)












