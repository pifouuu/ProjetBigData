
# coding: utf-8

#import packages
from pyspark import SparkContext
import loadFiles as lf
import numpy as np
import nltk
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import keyword_only
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import SQLContext

#create Sparkcontext
sc = SparkContext(appName="Simple App")

#Load the database for training
import loadFilesPartial as lfp
data,Y=lf.loadLabeled("./data/train")
print len(data)


#**********************************************************************
#----------------------Pre-processing data-----------------------------
#**********************************************************************
print "Pre-processing data"
# Le item est nécessaire pour convertir au format Double (python natif) 
# car les dataframes semblent avoir des problèmes avec les types numpy
labeledData = zip(data,[y.item() for y in Y])
labeledRdd = sc.parallelize(labeledData)


#from BeautifulSoup import BeautifulSoup as bfs
# Définition d'une fonction de nettoyage : on remplace les balises par des espaces
# à l'aide du parser de hmtl, et on met tout en minuscules.
#def cleanLower(doc):
#    clean = bfs(doc).get_text(separator=' ')
#   return clean.lower()

#Unfortunately Bs4 is not available on the X-cluster: removing html tags
#is going to be more empirical!!

# Définition d'une fonction de nettoyage : on remplace les balises par des espaces
# à l'aide du parser de hmtl, et on met tout en minuscules. (sans utiliser BeautifulSoup)
def cleanLower(doc):
    return doc.replace("<br />","").lower()
# On applique le nettoyage à la RDD
cleanRdd = labeledRdd.map(lambda doc : (cleanLower(doc[0]),doc[1]))



# On va utiliser la librairie spark ML donc on travaille avec des dataframes
# EN THEORIE, c'est plus facile. C'est un mensonge.
sqlContext = SQLContext(sc) #for the X-cluster
df = sqlContext.createDataFrame(cleanRdd, ['review', 'label'])

dfTrain, dfTest = df.randomSplit([0.8,0.2])
print "Training set (for the cross-validation): "
dfTrain.show()
print "Test set (for the cross-validation): "
dfTest.show()


#Tokenizing
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='review', outputCol='words')
dfTrainTok = tokenizer.transform(dfTrain)

# La ponctuation n'a pas l'air gérée très correctement...
print "La ponctuation n'a pas l'air gérée très correctement..."
dfTrainTok.select('words').take(1)


# J'ai écrit cette fonction au cas où on souhaiterait absolument passer par des 
# pipelines, et intégrer les stopword dans la structure d'un transformer devait 
# passer par ce code grosso modo. Au final, on se passera surement des pipelines
# car leurs possibilités sont limitées sur spark ml. Très limitées. C'est de la 
# merde en fait. Voila.


#----------------------------------------------------------------------
#----------------------Removing Stop words-----------------------------
#----------------------------------------------------------------------
print "Removing Stop words"
import nltk
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import keyword_only
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

class NLTKWordPunctTokenizer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, stopwords=None):
        super(NLTKWordPunctTokenizer, self).__init__()
        self.stopwords = Param(self, "stopwords", "")
        self._setDefault(stopwords=set())
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, stopwords=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
    def setStopwords(self, value):
        self._paramMap[self.stopwords] = value
        return self
    def getStopwords(self):
        return self.getOrDefault(self.stopwords)
    def _transform(self, dataset):
        stopwords = self.getStopwords()
        def f(s):
            tokens = nltk.tokenize.wordpunct_tokenize(s)
            return [t for t in tokens if t.lower() not in stopwords]
        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


#remove Stop-words
# Par contre cette fonction marche bien
tokenizerNoSw = NLTKWordPunctTokenizer(inputCol="review", outputCol="wordsNoSw",stopwords=set(nltk.corpus.stopwords.words('english')))
dfTrainTokNoSw = tokenizerNoSw.transform(dfTrainTok)
print "Training set without stop words: "
dfTrainTokNoSw.show()
# La ponctuation est bien gérée
print "La ponctuation est bien gérée :"
dfTrainTokNoSw.select('wordsNoSw').take(1)



#----------------------------------------------------------------------
#----------------------------------Tagging-----------------------------
#----------------------------------------------------------------------
from pyspark.sql.types import StructType, StructField

class NLTKPosTagger(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, tagset=None):
        super(NLTKPosTagger, self).__init__()
        self.tagset = Param(self, "tagset", "")
        self._setDefault(tagset='universal')
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, tagset=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)
    def setTagset(self, value):
        self._paramMap[self.tagset] = value
        return self
    def getTagset(self):
        return self.getOrDefault(self.tagset)
    def _transform(self, dataset):
        tagset = self.getTagset()
        def f(s):
            return nltk.pos_tag(s,tagset=tagset)
        fields = (StructField('word',StringType(),True),StructField('tag',StringType(),True))
        t = ArrayType(StructType(fields))
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


posTagger = NLTKPosTagger(inputCol="words", outputCol="tagWords")
dfTagged = posTagger.transform(dfTrainTok)
#dfTagged.show()



#----------------------------------------------------------------------
#------------------------------Bigrams---------------------------------
#----------------------------------------------------------------------
from pyspark.ml.feature import NGram
bigram = NGram(inputCol="words", outputCol="bigrams")
dfBigram = bigram.transform(dfTrainTokNoSw)
print "DataFrame des Bigram: "
dfBigram.show()




#**********************************************************************
#------------------------Feature selection-----------------------------
#**********************************************************************

# Pour la suite on a le choix entre l'encodage utilisé par le prof (le mot y est ou n'y est pas)
# ou la version en apparence plus informative du tfidf. En vrai, le tfidf peut être trompeur 
# donc je construis quand même les dictionnaires d'unigrammes et de bigrammes pour pouvoir 
# calculer les sparse vectors du prof.

import itertools
lists=dfBigram.map(lambda r : r.words).collect()
dictUnigrams=set(itertools.chain(*lists))
lists2=dfBigram.map(lambda r : r.bigrams).collect()
dictBigrams=set(itertools.chain(*lists2))


dictionaryUni={}
for i,word in enumerate(dictUnigrams):
	dictionaryUni[word]=i
dictionaryBigrams={}
for i,word in enumerate(dictBigrams):
	dictionaryBigrams[word]=i


'!' in dictionaryUni


# Voila qui est mieux...
lists3=dfBigram.map(lambda r : r.wordsNoSw).collect()
dict3=set(itertools.chain(*lists3))
dictionary3 = {}
for i,word in enumerate(dict3):
    dictionary3[word]=i
'!' in dictionary3

from string import punctuation
import re
r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))


#for k in dict3:
#    m = r.search(k)
#    if m:
#        print m.group()

# Fonction calculant le sparse vector correspondant à un ensemble de tokens
from pyspark.mllib.linalg import SparseVector
def vectorizeUni(tokens):
    vector_dict={}
    for w in tokens:
        vector_dict[dictionaryUni[w]]=1
    return SparseVector(len(dictionaryUni),vector_dict)

def vectorizeBi(tokens):
    vector_dict={}
    for w in tokens:
        vector_dict[dictionaryBigrams[w]]=1
    return SparseVector(len(dictionaryBigrams),vector_dict)


# In[52]:

# La ca devient le bordel, j'en ai chié pour arriver à appliquer une fonction à toute une colonne 
# d'une dataframe. Contrairement à pandas, y a pas de fonction "apply", il faut recourir à des 
# UserDefinedFunctions, et penser que le type sparseVector ne sera pas reconnu par la dataframe, qui
# n'est compatible qu'avec un nombre restreint de types

# EDIT : en fait je m'en suis pas rendu compte, mais cette manip je l'avais déjà faite pour la surcharge des
# tokenizer et postagger... les cinq dernières lignes à la fin avec udf et tout



from pyspark.sql.functions import UserDefinedFunction
from pyspark.mllib.linalg import VectorUDT
udfVectorizeUni=UserDefinedFunction(lambda x : vectorizeUni(x),VectorUDT())

# Une dataframe est un objet immutable, donc pas la peine d'essayer de modifier une colonne,
# à la place on crée une deuxième dataframe où on ajoute la colonne qu'on veut.
dfVect = dfBigram.withColumn("words", udfVectorizeUni("words"))
# On a bien remplacé ici du coup les mots par les vecteurs sparse
print "DataFrame(1-gram): On a bien remplacé ici du coup les mots par les vecteurs sparse"
dfVect.show()


udfVectorizeBi=UserDefinedFunction(lambda x : vectorizeBi(x),VectorUDT())
dfVect2 = dfVect.withColumn("bigrams", udfVectorizeBi("bigrams"))
print "DataFrame(bi-gram): On a bien remplacé ici du coup les mots par les vecteurs sparse"
dfVect2.show()

# Pour les opérations de traitement du langage, il est d'usage de normaliser (L2)
# les vecteurs de features : c'est ce qui marche le mieux apparemment.
from pyspark.ml.feature import Normalizer
normalizerUni = Normalizer(inputCol='words',outputCol='normWords',p=2.0)
normalizerBi = Normalizer(inputCol="bigrams",outputCol='normBigrams',p=2.0)
dfNorm = normalizerUni.transform(dfVect2)
dfNorm2 = normalizerBi.transform(dfNorm)
print "DataFrame(bi-gram): normalisé"
dfNorm2.select('words','normWords').show()
# La différence n'apparait pas dans la table puisqu'on n'a la place de visualiser que les indices des élements 
# non nuls et pas leur valeur
# On passe au TFIDF
# Evidemment en choisissant la bonne dataframe parmi celle du dessus, on peut appliquer ces calculs
# à n'importz quelle colonne (bigrammes, avec stop words ou sans...)
from pyspark.ml.feature import HashingTF
htf = HashingTF(inputCol='words',outputCol='wordsTF',numFeatures=10000)
dfTrainTF = htf.transform(dfTrainTokNoSw)
# INverse doc frequency
from pyspark.ml.feature import IDF
idf = IDF(inputCol=htf.getOutputCol(),outputCol="wordsTFIDF")
idfModel = idf.fit(dfTrainTF)
dfTrainTFIDF = idfModel.transform(dfTrainTF)
dfTrainTFIDF.select('review','wordsTF','wordsTFIDF').show()

# Je sais que cette étape m'a été utile une fois, la ça a pas trop l'air
from pyspark.ml.feature import StringIndexer
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
string_indexer_model = string_indexer.fit(dfTrainTFIDF)
dfTrainFinal = string_indexer_model.transform(dfTrainTFIDF)
dfTrainFinal.select('review','label','target_indexed').show()



#**********************************************************************
#-----------Training the model for prediction--------------------------
#**********************************************************************


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol=idf.getOutputCol(),labelCol=string_indexer.getOutputCol())
dt_model = dt.fit(dfTrainFinal)



# On applique le même à notre ensemble de test ridicule.
# En théorie le pipeline permet d'automatiser tout ça mais bon, on s'en servira probablement pas

# EDIT : en fait c'est plutot facile de créer des transformers à partir de chaque étape, donc peut 
# être que les pipelines c'est faisables. A voir
df_test_words = tokenizer.transform(dfTest)
df_test_tf = htf.transform(df_test_words)
df_test_tfidf = idfModel.transform(df_test_tf)
df_test_final = string_indexer_model.transform(df_test_tfidf)
# Les prédictions
df_test_pred = dt_model.transform(df_test_final)
df_test_pred.select('review', 'target_indexed', 'prediction', 'probability').show(5)

# Je fais un pipeline très basique
from pyspark.ml import Pipeline


# Instanciate all the Estimators and Transformers necessary
tokenizer = Tokenizer(inputCol='review', outputCol='reviews_words')
hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='reviews_tf', numFeatures=10000)
idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="reviews_tfidf")
string_indexer = StringIndexer(inputCol='label', outputCol='target_indexed')
dt = DecisionTreeClassifier(featuresCol=idf.getOutputCol(), labelCol=string_indexer.getOutputCol(), maxDepth=10)


# Instanciate a Pipeline
pipeline = Pipeline(stages=[tokenizer,hashing_tf,idf,string_indexer,dt])
pipeline_model = pipeline.fit(dfTrain)
df_test_pred = pipeline_model.transform(dfTest)
df_test_pred.select('review', 'target_indexed', 'prediction', 'probability').show()


# Un outil automatique pour calculer le taux de bonne classif.
# La encore pas très utile en vrai
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target_indexed', metricName='precision')
evaluator.evaluate(df_test_pred)



#**********************************************************************
#-----------Cross Validation--------------------------
#**********************************************************************

# La cross validation et le test des différents paramètres du classifieurs c'est pas trop dur sur spark ML, 
# c'est en fait la seule raison pour laquelle cette librairie me paraissait mieux... avec du recul j'aurais 
# perdu moins de temps à recoder moi-même la cross valid et la grid search...

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

grid=(ParamGridBuilder().baseOn([evaluator.metricName,'precision']).addGrid(dt.maxDepth, [10,20]).build())
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,evaluator=evaluator)
cv_model = cv.fit(dfTrain)
df_test_pred = cv_model.transform(dfTest)
resultat=evaluator.evaluate(df_test_pred)
print "Pourcentage de bonne classification(0-1): ",resultat

