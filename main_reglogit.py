#import packages
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

sc = SparkContext(appName="Simple App")

# my modulus
import loadFilesPartial as lfp


#*****************************************************************
#************************Two useful functions*********************
#******************************************************************
def createBinaryLabeledPoint(doc_class,dictionary):
	words=doc_class[0].strip().split(' ')
	#create a binary vector for the document with all the words that appear (0:does not appear,1:appears)
	#we can set in a dictionary only the indexes of the words that appear
	#and we can use that to build a SparseVector
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return LabeledPoint(doc_class[1], SparseVector(len(dictionary),vector_dict))
def Predict(name_text,dictionary,model):
	words=name_text[1].strip().split(' ')
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text[0], model.predict(SparseVector(len(dictionary),vector_dict)))


#*****************************************************************
#********Feature Extraction & Transformation*********************
#******************************************************************
#load data
data,Y=lf.loadLabeled("./data/train")
#data,Y=lfp.loadLabeled("./data/train",1000)
print len(data)
dataRDD=sc.parallelize(data,numSlices=16)

#map data to a binary matrix
#1. get the dictionary of the data
#The dictionary of each document is a list of UNIQUE(set) words 
lists=dataRDD.map(lambda x:list(set(x.strip().split(' ')))).collect()
all=[]
#combine all dictionaries together (fastest solution for Python)
for l in lists:
	all.extend(l)
dict=set(all)
print len(dict)
#it is faster to know the position of the word if we put it as values in a dictionary
dictionary={}
for i,word in enumerate(dict):
	dictionary[word]=i
#we need the dictionary to be available AS A WHOLE throughout the cluster
dict_broad=sc.broadcast(dictionary)
#build labelled Points from data
data_class=zip(data,Y)#if a=[1,2,3] & b=['a','b','c'] then zip(a,b)=[(1,'a'),(2, 'b'), (3, 'c')]
dcRDD=sc.parallelize(data_class,numSlices=16)
#get the labelled points
labeledRDD=dcRDD.map(partial(createBinaryLabeledPoint,dictionary=dict_broad.value))


#****************************************************************
#*********************CROSS VALIDATION: 80%/20%******************
#*******************Model: logistic regression*******************
#*****************************************************************

#create a data frame from an RDD -> features must be Vectors.sparse from pyspark.mllib.linalg
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(labeledRDD, ['features','label'])
dfTrain, dfTest = df.randomSplit([0.8,0.2])
dfTrain.show()
#choose estimator and grid
lr = LogisticRegression()	#choose the model
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()	
#the grid is built to find the best paramter 'alpha' for the regularization of the model. It is an elastic net
#alpha=0, for a L2 regularization, 
#alpha=1, for a L1 regularization
print "Start Cross validation"

evaluator = BinaryClassificationEvaluator()	#choose the evaluator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator) #perform the cross validation and keeps the best value of maxIter
cvModel = cv.fit(dfTrain)	#train the model on the whole training set
resultat=evaluator.evaluate(cvModel.transform(dfTest))	#compute the percentage of success on test set
print "Percentage of correct predicted labels (0-1): ",resultat
