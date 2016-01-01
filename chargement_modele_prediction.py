# Les valeurs a modifier sont :
# sauvegarde_MYMODEL --> remplacer par le fichier ou le modele a ete stocke
# MY_SCRIPT --> remplacer par le nom de votre script d'entrainement

from pyspark import SparkContext
import loadFiles as lf
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
sc = SparkContext(appName="Simple App")

import pickle
import MY_SCRIPT

import 

# load trained model
input=file('./sauvegarde_MYMODEL','r')
fitted_model = pickle.load(input)
input.close()

# broadcast model
mb=sc.broadcast(fitted_model)

# load test data
text,names=lf.loadUknown('./data/test')

# put test data into a clean DataFrame format
#
# note : on aurait pu utiliser le .toDf ...
#
name_text=zip(names,text)
unlabeledRdd = sc.parallelize(name_text)
test_rdd = unlabeledRdd.map(lambda doc : (cleanLower(doc[0]),doc[1]))
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(test_rdd, ['id', 'review'])

# make prediction on test data
predictions=fitted_model.tranform(df)

# write in classifications.txt
# NOTE : this assumes that the fitted model has a methode transform() which appends the prediction
# as the last column of the dataframe
output=file('./classifications.txt','w')
for i in range(predictions.count()):
	output.write('%s\t%d\n'% (predictions.collect()[i][0], predictions.collect()[i][-1]))
output.close()
