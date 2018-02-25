
# coding: utf-8

# In[1]:


# entity_resolution.py
import re
import operator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, types, functions
from pyspark.sql.functions import concat, concat_ws, col, lit, explode


# In[2]:


spark = SparkSession.builder.appName('entity_res').getOrCreate()
sc = spark.sparkContext


# In[3]:


class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()



    def preprocessDF(self, df, cols):
        """
            Input: $df represents a DataFrame
                   $cols represents the list of columns (in $df) that will be concatenated and be tokenized

            Output: Return a new DataFrame that adds the "joinKey" column into the input $df

            Comments: The "joinKey" column is a list of tokens, which is generated as follows:
                     (1) concatenate the $cols in $df;
                     (2) apply the tokenizer to the concatenated string
            Here is how the tokenizer should work:
                     (1) Use "re.split(r'\W+', string)" to split a string into a set of tokens
                     (2) Convert each token to its lower-case
                     (3) Remove stop words
        """
        stop_words = self.stopWordsBC
        def tokenized_filterized_string (string):
            string = re.sub('\s+',' ',string).strip().lower() # Remove extra whitespace and finally remove trailing spaces
            tokens = re.split(r'\W+', string)
            stop_words.add('')
            tokens = set(tokens) - stop_words
            return list(tokens)

        get_tokenized_string = functions.udf(tokenized_filterized_string, types.ArrayType(types.StringType()))
        concatanated_column = 'joinKey'
        df = df.withColumn(concatanated_column, concat_ws(' ', df[cols[0]], df[cols[1]]))
        df = df.withColumn(concatanated_column, get_tokenized_string(df[concatanated_column]))
        return df

    def filtering(self, df1, df2):
        """
            Input: $df1 and $df2 are two input DataFrames, where each of them
                   has a 'joinKey' column added by the preprocessDF function

            Output: Return a new DataFrame $candDF with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
                    where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
                    Intuitively, $candDF is the joined result between $df1 and $df2 on the condition that
                    their joinKeys share at least one token.

            Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons,
                      you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function.
                      Please come up with a more efficient algorithm (see hints in Lecture 2).
        """

        df1 = df1.select('id', 'joinKey').withColumn("flattened_key", explode(df1['joinKey']))
        df2 = df2.select('id', 'joinKey').withColumn("flattened_key", explode(df2['joinKey']))
        df1.createOrReplaceTempView("df1")
        df2.createOrReplaceTempView("df2")
        common_item = spark.sql('SELECT distinct df1.id as id1, df1.joinKey as joinKey1, df2.id as id2, df2.joinKey as joinKey2         FROM df1, df2 WHERE df1.flattened_key = df2.flattened_key')
        return common_item



    def verification(self, candDF, threshold):
        """
            Input: $candDF is the output DataFrame from the 'filtering' function.
                   $threshold is a float value between (0, 1]

            Output: Return a new DataFrame $resultDF that represents the ER result.
                    It has five columns: id1, joinKey1, id2, joinKey2, jaccard

            Comments: There are two differences between $candDF and $resultDF
                      (1) $resultDF adds a new column, called jaccard, which stores the jaccard similarity
                          between $joinKey1 and $joinKey2
                      (2) $resultDF removes the rows whose jaccard similarity is smaller than $threshold
        """
        def get_jaccard_similarity(set_1, set_2):
            set_1 = set(set_1)
            set_2 = set(set_2)
            return len(set_1 & set_2) * 1.00 / len(set_1 | set_2) * 1.00

        calculate_jaccard = functions.udf(get_jaccard_similarity, types.DoubleType())
        candDF = candDF.withColumn('jaccard', calculate_jaccard(candDF['joinKey1'], candDF['joinKey2']))
        candDF = candDF.filter(candDF.jaccard >= threshold)
        return candDF


    def evaluate(self, result, groundTruth):
        """
            Input: $result is a list of matching pairs identified by the ER algorithm
                   $groundTrueth is a list of matching pairs labeld by humans

            Output: Compute precision, recall, and fmeasure of $result based on $groundTruth, and
                    return the evaluation result as a triple: (precision, recall, fmeasure)

        """

        result_count = len(result) # Value of R
        groundTruth_count = len(groundTruth) # Value of A
        correctly_identified_result = set(result) & set(groundTruth)
        correctly_identified_result_count = len(correctly_identified_result) # Value of T
        precision = correctly_identified_result_count / result_count
        recall = correctly_identified_result_count / groundTruth_count
        fm_measure = 2 * precision * recall / (precision + recall)
        return (precision, recall, fm_measure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))
        candDF = self.filtering(newDF1, newDF2)
        print ("After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print ("After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF


    def __del__(self):
        self.f.close()


# In[4]:


if __name__ == "__main__":
    er = EntityResolution("Amazon_sample", "Google_sample", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample").rdd \
                        .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
