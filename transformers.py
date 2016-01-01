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
            tagged = nltk.pos_tag(s,tagset=tagset)
            return zip(*tagged)[1]

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

