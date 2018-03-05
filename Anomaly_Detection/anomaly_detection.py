
# coding: utf-8

# In[1]:


# anomaly_detection.py
import findspark
findspark.init()

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator

appName = 'anomaly_detection'
conf = SparkConf().setAppName(appName)
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName(appName).getOrCreate()


# In[161]:


class AnomalyDetection():
    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]),                 (1, ["http", "udf", 0.5]),                 (2, ["http", "tcp", 0.5]),                 (3, ["ftp", "icmp", 0.1]),                 (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = spark.createDataFrame(data, schema)

    def readData(self, filename, is_reading_toy_data=False):
        if not is_reading_toy_data:
            self.rawDF = spark.read.parquet(filename).cache()
        else:
            self.readToyData()

    def cat2Num(self, df, indices):
        """ 
            Input: $df represents a DataFrame with two columns: "id" and "rawFeatures"
                   $indices represents which dimensions in $rawFeatures are categorical features, 
                    e.g., indices = [0, 1] denotes that the first two dimensions are categorical features.
        
            Output: Return a new DataFrame that adds the "features" column into the input $df
        
            Comments: The difference between "features" and "rawFeatures" is that 
            the latter transforms all categorical features in the former into numerical features 
            using one-hot key representation
        """
        
        
        features = df
        for i in indices:
            extract_cat = udf(lambda x: x[i], StringType())
            features = features.withColumn("cat_{}".format(i), extract_cat(features['rawFeatures'])) 
        for i in indices:
            distinct = list(features.select('cat_{}'.format(i)).distinct().toPandas()['cat_{}'.format(i)])
            one_hot_encode = udf(lambda x: [float(x == value) for value in distinct], ArrayType(FloatType()))
            features = features.withColumn('cat_he_{}'.format(i), 
                                           one_hot_encode(features['cat_{}'.format(i)]))
            features = features.drop('cat_{}'.format(i))
            
        def insert_feature_udf(raw_feature, new_element, index):
            raw_feature[index] = ', '.join(str(i) for i in new_element)
            return raw_feature
        features = features.withColumn('features', features['rawFeatures'])
        for i in indices:
            insert_feature = udf(lambda x, y: insert_feature_udf(x, y, i), ArrayType(StringType()))
            features = features.withColumn( 'features', 
                                        insert_feature(features['features'],  features['cat_he_{}'.format(i)]) ) 
            features = features.drop('cat_he_{}'.format(i))
        to_float_list = udf(lambda x: [float(i) for i in (', '.join(x).replace(' ', '')).split(',')], ArrayType(DoubleType()))
        features = features.withColumn('features', to_float_list(features['features']))
        return features

    def addScore(self, df):
        """ 
            Input: $df represents a DataFrame with four columns: "id", "rawFeatures", "features", and "prediction"
            Output: Return a new DataFrame that adds the "score" column into the input $df

            To compute the score of a data point x, we use:

                 score(x) = (N_max - N_x)/(N_max - N_min), 

            where N_max and N_min represent the size of the largest and smallest clusters, respectively,
                  and N_x represents the size of the cluster assigned to x 
        """
        prediction_groupby = df.groupBy('prediction')
        df_predcount = prediction_groupby.count().cache()
        N_max = df_predcount.agg({'count': 'max'}).collect()[0]['max(count)']
        N_min = df_predcount.agg({'count': 'min'}).collect()[0]['min(count)']
        get_score = udf(lambda N_x: 0.0 if N_max == N_min else ((N_max - N_x)/(N_max - N_min)), FloatType())
        df = df.join(df_predcount, 'prediction', 'inner').select('rawFeatures', 'features', 'prediction', 'count')
        df = df.withColumn('score', get_score(df['count']))
        return df.select('rawFeatures', 'features', 'prediction', 'score')

    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly 
        df3 = self.addScore(df2).cache()
        df3.show()    

        return df3.where(df3.score > t)


# In[165]:


if __name__ == "__main__":
    ad = AnomalyDetection()
    use_toy_data = False
    ad.readData('./data/logs-features-sample', use_toy_data)
    if use_toy_data:
        anomalies = ad.detect(2, 0.9)
    else:
        anomalies = ad.detect(8, 0.97)
    print(anomalies.count())
    anomalies.show()

