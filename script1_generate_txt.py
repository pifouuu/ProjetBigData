# coding: utf-8

#this script generates the file classifications.txt based on the model of Decision Tree Classifier
#with advanced Feature Extraction & Transformation: Stemming & Cleaning, Stopword Removal
# and Feature Selection using TF-IDF

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


#create Sparkcontext
sc = SparkContext(appName="Simple App")

data,Y=lf.loadLabeled("./data/train")
labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)
def cleanLower(doc):
    return doc.replace("<br /><br />"," ").lower()
rdd = labeledRdd.map(lambda doc : (cleanLower(doc[0]),doc[1]))

print "Text is cleaned"

sqlContext = SQLContext(sc)
dfTrain = sqlContext.createDataFrame(rdd, ['review', 'label'])

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


model = pipeline.fit(dfTrain)

print "The model is fitted"

#import test set
test,names=lf.loadUknown('./data/test')
text_name=zip(test,names)
UnlabeledRdd = sc.parallelize(text_name)
def cleanLower2(doc):
    return doc.replace("<br /><br />"," ").lower()
Unlabeledrdd = UnlabeledRdd.map(lambda doc : (cleanLower2(doc[0]),doc[1]))

print "Test Text is cleaned"

dfTest = sqlContext.createDataFrame(Unlabeledrdd , ['review', 'name'])
# Make predictions.
predictions = model.transform(dfTest)
list_predictions=predictions.collect()

print "Predictions are made"


output=file('./classifications_script1.txt','w')
for x in list_predictions:
	output.write('{}\t{}\n'.format(x.name, x.prediction))
output.close()
print "File classifications_script1.txt is written"
