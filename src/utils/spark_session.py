import os
from pyspark.sql import SparkSession
def get_spark():
    os.environ['JAVA_HOME'] = '/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home'

    spark = (
        SparkSession.builder
        .appName('Recidivai')
        .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.1.0')
        .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')
        .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog')
        .config('spark.driver.extraJavaOptions', '-Dlog4j.rootCategory=WARN,console')
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel('WARN')
    return spark