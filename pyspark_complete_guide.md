# Complete PySpark Tutorial for Beginners

This guide covers all the essential aspects of PySpark, from basic setup to advanced techniques, with explanations for each step, example outputs, and interview questions at the end.

## Table of Contents
1. [Introduction to PySpark](#introduction-to-pyspark)
2. [Installation and Setup](#installation-and-setup)
3. [Creating a SparkSession](#creating-a-sparksession)
4. [Working with DataFrames](#working-with-dataframes)
5. [Data Manipulation](#data-manipulation)
6. [Working with Functions](#working-with-functions)
7. [Handling Missing Values](#handling-missing-values)
8. [Machine Learning with PySpark ML](#machine-learning-with-pyspark-ml)
9. [Saving and Loading Data](#saving-and-loading-data)
10. [Performance Optimization](#performance-optimization)
11. [Spark Streaming Basics](#spark-streaming-basics)
12. [Interview Questions and Answers](#interview-questions-and-answers)

## Introduction to PySpark

PySpark is the Python API for Apache Spark, a distributed computing framework designed for big data processing. It allows you to process large datasets in parallel across a cluster of computers.

Key advantages of PySpark:
- Fast processing through in-memory computation
- Built-in modules for SQL, streaming, machine learning, and graph processing
- Fault tolerance through RDD lineage
- Scalability from a single machine to thousands of nodes
- Easy-to-use APIs in Python

## Installation and Setup

First, you need to install PySpark:

```python
# Install PySpark using pip
pip install pyspark

# For visualization support
pip install matplotlib
pip install pandas
```

Basic imports you'll need:

```python
# Core PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# ML-specific imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

## Creating a SparkSession

The SparkSession is your entry point to all Spark functionality:

```python
# Create a basic SparkSession
spark = SparkSession.builder \
    .appName("PySpark Tutorial") \
    .getOrCreate()

# Create a more configured SparkSession
spark = SparkSession.builder \
    .appName("PySpark Tutorial") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# To use Spark with Hadoop/S3
spark = SparkSession.builder \
    .appName("PySpark Tutorial") \
    .config("spark.hadoop.fs.s3a.access.key", "YOUR_ACCESS_KEY") \
    .config("spark.hadoop.fs.s3a.secret.key", "YOUR_SECRET_KEY") \
    .getOrCreate()

# Check Spark version
print(f"Spark version: {spark.version}")
# Output: Spark version: 3.3.0
```

## Working with DataFrames

DataFrames are the most common data structure in PySpark, similar to pandas DataFrames but distributed:

```python
# Create a DataFrame from a list
data = [("John", 30), ("Alice", 25), ("Bob", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# Show the DataFrame (displays first 20 rows by default)
df.show()
# Output:
# +-----+---+
# | name|age|
# +-----+---+
# | John| 30|
# |Alice| 25|
# |  Bob| 35|
# +-----+---+

# Create a more complex DataFrame
from datetime import date
more_data = [
    (1, "John", 30, 70000, date(1992, 10, 15), "IT"),
    (2, "Alice", 25, 90000, date(1997, 5, 22), "HR"),
    (3, "Bob", 35, 85000, date(1987, 3, 10), "IT"),
    (4, "Maria", 40, 95000, date(1982, 1, 5), "Finance"),
    (5, "David", 28, 75000, date(1994, 8, 30), "Marketing")
]
columns = ["id", "name", "age", "salary", "dob", "department"]
employees_df = spark.createDataFrame(more_data, columns)
employees_df.show()
# Output:
# +---+-----+---+------+----------+----------+
# | id| name|age|salary|       dob|department|
# +---+-----+---+------+----------+----------+
# |  1| John| 30| 70000|1992-10-15|        IT|
# |  2|Alice| 25| 90000|1997-05-22|        HR|
# |  3|  Bob| 35| 85000|1987-03-10|        IT|
# |  4|Maria| 40| 95000|1982-01-05|   Finance|
# |  5|David| 28| 75000|1994-08-30| Marketing|
# +---+-----+---+------+----------+----------+

# Examine DataFrame schema
employees_df.printSchema()
# Output:
# root
#  |-- id: long (nullable = true)
#  |-- name: string (nullable = true)
#  |-- age: long (nullable = true)
#  |-- salary: long (nullable = true)
#  |-- dob: date (nullable = true)
#  |-- department: string (nullable = true)

# Get DataFrame information
print(f"Number of rows: {employees_df.count()}")
print(f"Number of columns: {len(employees_df.columns)}")
# Output:
# Number of rows: 5
# Number of columns: 6
```

### Reading Data from Files

```python
# Read from CSV
df_csv = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)

# Read from JSON
df_json = spark.read.json("path/to/file.json")

# Read from Parquet
df_parquet = spark.read.parquet("path/to/file.parquet")

# Read from a database using JDBC
df_jdbc = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydatabase") \
    .option("dbtable", "schema.tablename") \
    .option("user", "username") \
    .option("password", "password") \
    .load()
```

## Data Manipulation

### Basic DataFrame Operations

```python
# Select specific columns
employees_df.select("name", "age", "department").show()
# Output:
# +-----+---+----------+
# | name|age|department|
# +-----+---+----------+
# | John| 30|        IT|
# |Alice| 25|        HR|
# |  Bob| 35|        IT|
# |Maria| 40|   Finance|
# |David| 28| Marketing|
# +-----+---+----------+

# Filter rows
employees_df.filter(employees_df.age > 30).show()
# Output:
# +---+-----+---+------+----------+----------+
# | id| name|age|salary|       dob|department|
# +---+-----+---+------+----------+----------+
# |  3|  Bob| 35| 85000|1987-03-10|        IT|
# |  4|Maria| 40| 95000|1982-01-05|   Finance|
# +---+-----+---+------+----------+----------+

# Alternative syntax
employees_df.filter("age > 30").show()
# Same output as above

# Sort data
employees_df.sort("age").show()  # Ascending by default
# Output:
# +---+-----+---+------+----------+----------+
# | id| name|age|salary|       dob|department|
# +---+-----+---+------+----------+----------+
# |  2|Alice| 25| 90000|1997-05-22|        HR|
# |  5|David| 28| 75000|1994-08-30| Marketing|
# |  1| John| 30| 70000|1992-10-15|        IT|
# |  3|  Bob| 35| 85000|1987-03-10|        IT|
# |  4|Maria| 40| 95000|1982-01-05|   Finance|
# +---+-----+---+------+----------+----------+

employees_df.sort(F.col("age").desc()).show()  # Descending
# Output:
# +---+-----+---+------+----------+----------+
# | id| name|age|salary|       dob|department|
# +---+-----+---+------+----------+----------+
# |  4|Maria| 40| 95000|1982-01-05|   Finance|
# |  3|  Bob| 35| 85000|1987-03-10|        IT|
# |  1| John| 30| 70000|1992-10-15|        IT|
# |  5|David| 28| 75000|1994-08-30| Marketing|
# |  2|Alice| 25| 90000|1997-05-22|        HR|
# +---+-----+---+------+----------+----------+

# Limit results
employees_df.limit(2).show()
# Output:
# +---+-----+---+------+----------+----------+
# | id| name|age|salary|       dob|department|
# +---+-----+---+------+----------+----------+
# |  1| John| 30| 70000|1992-10-15|        IT|
# |  2|Alice| 25| 90000|1997-05-22|        HR|
# +---+-----+---+------+----------+----------+

# Count by group
employees_df.groupBy("department").count().show()
# Output:
# +----------+-----+
# |department|count|
# +----------+-----+
# |        HR|    1|
# | Marketing|    1|
# |   Finance|    1|
# |        IT|    2|
# +----------+-----+

# Aggregate functions
from pyspark.sql.functions import avg, sum, min, max, count
employees_df.groupBy("department") \
    .agg(
        count("id").alias("employee_count"),
        avg("age").alias("avg_age"),
        avg("salary").alias("avg_salary"),
        min("salary").alias("min_salary"),
        max("salary").alias("max_salary")
    ).show()
# Output:
# +----------+--------------+-------+------------------+----------+----------+
# |department|employee_count|avg_age|        avg_salary|min_salary|max_salary|
# +----------+--------------+-------+------------------+----------+----------+
# |        HR|             1|   25.0|           90000.0|     90000|     90000|
# | Marketing|             1|   28.0|           75000.0|     75000|     75000|
# |   Finance|             1|   40.0|           95000.0|     95000|     95000|
# |        IT|             2|   32.5|77500.00000000001|     70000|     85000|
# +----------+--------------+-------+------------------+----------+----------+
```

### Adding and Modifying Columns

```python
# Add a new column
employees_df = employees_df.withColumn("salary_thousands", employees_df.salary / 1000)
employees_df.show()
# Output:
# +---+-----+---+------+----------+----------+----------------+
# | id| name|age|salary|       dob|department|salary_thousands|
# +---+-----+---+------+----------+----------+----------------+
# |  1| John| 30| 70000|1992-10-15|        IT|            70.0|
# |  2|Alice| 25| 90000|1997-05-22|        HR|            90.0|
# |  3|  Bob| 35| 85000|1987-03-10|        IT|            85.0|
# |  4|Maria| 40| 95000|1982-01-05|   Finance|            95.0|
# |  5|David| 28| 75000|1994-08-30| Marketing|            75.0|
# +---+-----+---+------+----------+----------+----------------+

# Add a column with an expression
employees_df = employees_df.withColumn("experience", F.current_date().year - F.year("dob"))
employees_df.show()
# Output (assuming current year is 2023):
# +---+-----+---+------+----------+----------+----------------+----------+
# | id| name|age|salary|       dob|department|salary_thousands|experience|
# +---+-----+---+------+----------+----------+----------------+----------+
# |  1| John| 30| 70000|1992-10-15|        IT|            70.0|        31|
# |  2|Alice| 25| 90000|1997-05-22|        HR|            90.0|        26|
# |  3|  Bob| 35| 85000|1987-03-10|        IT|            85.0|        36|
# |  4|Maria| 40| 95000|1982-01-05|   Finance|            95.0|        41|
# |  5|David| 28| 75000|1994-08-30| Marketing|            75.0|        29|
# +---+-----+---+------+----------+----------+----------------+----------+

# Update an existing column
employees_df = employees_df.withColumn("salary", employees_df.salary * 1.1)
employees_df.show()
# Output:
# +---+-----+---+------+----------+----------+----------------+----------+
# | id| name|age|salary|       dob|department|salary_thousands|experience|
# +---+-----+---+------+----------+----------+----------------+----------+
# |  1| John| 30| 77000|1992-10-15|        IT|            70.0|        31|
# |  2|Alice| 25| 99000|1997-05-22|        HR|            90.0|        26|
# |  3|  Bob| 35| 93500|1987-03-10|        IT|            85.0|        36|
# |  4|Maria| 40|104500|1982-01-05|   Finance|            95.0|        41|
# |  5|David| 28| 82500|1994-08-30| Marketing|            75.0|        29|
# +---+-----+---+------+----------+----------+----------------+----------+

# Rename a column
employees_df = employees_df.withColumnRenamed("dob", "date_of_birth")
employees_df.show()
# Output:
# +---+-----+---+------+------------+----------+----------------+----------+
# | id| name|age|salary|date_of_birth|department|salary_thousands|experience|
# +---+-----+---+------+------------+----------+----------------+----------+
# |  1| John| 30| 77000|  1992-10-15|        IT|            70.0|        31|
# |  2|Alice| 25| 99000|  1997-05-22|        HR|            90.0|        26|
# |  3|  Bob| 35| 93500|  1987-03-10|        IT|            85.0|        36|
# |  4|Maria| 40|104500|  1982-01-05|   Finance|            95.0|        41|
# |  5|David| 28| 82500|  1994-08-30| Marketing|            75.0|        29|
# +---+-----+---+------+------------+----------+----------------+----------+

# Drop a column
employees_df = employees_df.drop("experience")
```

### SQL Queries

PySpark allows you to use SQL queries on DataFrames:

```python
# Register DataFrame as a temporary view
employees_df.createOrReplaceTempView("employees")

# Run SQL query
result = spark.sql("""
    SELECT department, 
           COUNT(*) as employee_count, 
           AVG(age) as avg_age, 
           AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
    HAVING COUNT(*) > 1
    ORDER BY avg_salary DESC
""")
result.show()
```

## Working with Functions

PySpark provides many built-in functions for data manipulation:

```python
# Import functions
from pyspark.sql import functions as F

# String functions
employees_df = employees_df.withColumn("name_upper", F.upper(F.col("name")))
employees_df = employees_df.withColumn("name_length", F.length(F.col("name")))

# Date functions
employees_df = employees_df.withColumn("hire_year", F.year(F.col("dob")))
employees_df = employees_df.withColumn("days_employed", F.datediff(F.current_date(), F.col("dob")))

# Conditional expressions
employees_df = employees_df.withColumn(
    "salary_category",
    F.when(F.col("salary") < 80000, "Low")
     .when(F.col("salary") < 90000, "Medium")
     .otherwise("High")
)

# Working with arrays
data = [("John", ["Python", "Java", "Scala"]), ("Alice", ["Spark", "Python"])]
skills_df = spark.createDataFrame(data, ["name", "skills"])

# Array operations
skills_df = skills_df.withColumn("num_skills", F.size(F.col("skills")))
skills_df = skills_df.withColumn("has_python", F.array_contains(F.col("skills"), "Python"))
skills_df = skills_df.withColumn("first_skill", F.element_at(F.col("skills"), 1))
skills_df.show()
```

### User-Defined Functions (UDFs)

When built-in functions aren't enough, you can create your own:

```python
# Define a Python function
def calculate_experience(age):
    if age is None:
        return None
    return max(0, age - 22)  # Assuming college graduation at 22

# Register as UDF
experience_udf = F.udf(calculate_experience, IntegerType())

# Apply UDF to DataFrame
employees_df = employees_df.withColumn("work_experience", experience_udf(F.col("age")))

# Alternative: register UDF for SQL
spark.udf.register("sql_experience", calculate_experience, IntegerType())
spark.sql("SELECT name, age, sql_experience(age) as experience FROM employees").show()
```

## Handling Missing Values

Real-world data often contains missing values:

```python
# Create DataFrame with missing values
data_with_nulls = [
    (1, "John", 30),
    (2, "Alice", None),
    (3, None, 25),
    (4, "Bob", None)
]
null_df = spark.createDataFrame(data_with_nulls, ["id", "name", "age"])

# Check for null values
null_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in null_df.columns]).show()

# Drop rows with any null values
null_df.dropna().show()

# Drop rows with null values in specific columns
null_df.dropna(subset=["name"]).show()

# Fill null values with a constant
null_df.fillna(0).show()  # Fills all numeric columns with 0

# Fill null values with different values per column
null_df.fillna({"age": 0, "name": "Unknown"}).show()

# Replace values
null_df.replace(30, 31).show()  # Replace in all columns
null_df.replace([30], [31], "age").show()  # Replace in specific column
```

## Machine Learning with PySpark ML

PySpark ML provides a high-level API for machine learning:

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Sample data for classification
data = [
    (0, "a", 0.1, 0.2, 1.0),
    (1, "b", 0.3, 0.4, 0.0),
    (2, "c", 0.5, 0.6, 1.0),
    (3, "a", 0.7, 0.8, 1.0),
    (4, "b", 0.9, 1.0, 0.0),
    (5, "c", 0.2, 0.3, 1.0)
]
df = spark.createDataFrame(data, ["id", "category", "feature1", "feature2", "label"])

# 1. Convert categorical feature to numeric
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")

# 2. Combine features into a vector
assembler = VectorAssembler(
    inputCols=["categoryIndex", "feature1", "feature2"],
    outputCol="features"
)

# 3. Standardize features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# 4. Define the model
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label")

# 5. Create and run pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Split data
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Train model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)
predictions.select("id", "label", "prediction", "probability").show()

# Evaluate model
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Access the model
lr_model = model.stages[-1]
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
```

## Saving and Loading Data

Persisting your data and models:

```python
# Save DataFrame in various formats
df.write.csv("path/to/output/csv", header=True, mode="overwrite")
df.write.parquet("path/to/output/parquet", mode="overwrite")
df.write.json("path/to/output/json", mode="overwrite")

# Save as a Hive table
df.write.saveAsTable("my_table")

# Partition data when saving
df.write.partitionBy("department").parquet("path/to/partitioned/data")

# Save ML model
model.save("path/to/model")

# Load ML model
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load("path/to/model")
```

## Performance Optimization

Techniques to improve Spark performance:

```python
# Cache DataFrame in memory
df.cache()

# Persist with different storage levels
from pyspark.storagelevel import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist when done
df.unpersist()

# Repartition to control parallelism
df = df.repartition(10)  # Increase partitions for more parallelism

# Coalesce to reduce partitions (more efficient than repartition for reducing)
df = df.coalesce(1)  # Reduce to a single partition

# Broadcast join for small DataFrames
small_df = spark.createDataFrame([("IT", "Technology"), ("HR", "Human Resources")], ["dept_code", "dept_name"])
result = employees_df.join(F.broadcast(small_df), employees_df.department == small_df.dept_code)

# Set configuration for better performance
spark.conf.set("spark.sql.shuffle.partitions", 100)  # Default is 200
spark.conf.set("spark.executor.memory", "4g")
spark.conf.set("spark.driver.memory", "2g")
```

## Spark Streaming Basics

Process data in real-time with Spark Streaming:

```python
# Structured Streaming example
from pyspark.sql.streaming import StreamingQuery

# Create streaming DataFrame from a socket source
stream_df = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Process streaming data - word count example
from pyspark.sql.functions import explode, split
words_df = stream_df.select(explode(split(stream_df.value, " ")).alias("word"))
counts_df = words_df.groupBy("word").count()

# Output to console
query = counts_df \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Wait for the query to terminate
query.awaitTermination()

# Alternatively, write to files
query = counts_df \
    .writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "path/to/output") \
    .option("checkpointLocation", "path/to/checkpoint") \
    .start()
```

## Interview Questions and Answers

### 1. What is Apache Spark and how does it differ from Hadoop MapReduce?

**Answer:** Apache Spark is an open-source, distributed computing framework designed for big data processing. Key differences from Hadoop MapReduce include:

- **Speed:** Spark is significantly faster (up to 100x for in-memory operations) because it processes data in-memory rather than writing intermediate results to disk.
- **Ease of use:** Spark offers high-level APIs in Python, Java, Scala, and R, making it more developer-friendly.
- **Versatility:** Spark provides a unified platform for batch processing, interactive queries, streaming, machine learning, and graph processing, while MapReduce is primarily for batch processing.
- **Lazy evaluation:** Spark uses lazy evaluation to optimize the execution plan.
- **Iterative algorithms:** Spark excels at iterative algorithms common in machine learning, while MapReduce is inefficient for such workloads.

### 2. Explain the core components of Spark architecture.

**Answer:** Spark architecture consists of:

- **Driver Program:** Contains the main function and creates the SparkContext, which coordinates the execution.
- **Cluster Manager:** Allocates resources across applications (can be Spark's standalone manager, YARN, Mesos, or Kubernetes).
- **Executors:** Worker processes that run tasks and store data in memory or disk storage across the cluster.
- **SparkContext:** Main entry point for Spark functionality, coordinates with the cluster manager.
- **RDDs/DataFrames/Datasets:** Distributed data abstractions that represent data across the cluster.
- **DAG Scheduler:** Builds a directed acyclic graph of stages for each job and determines the optimal execution plan.
- **Task Scheduler:** Assigns tasks to executors.

### 3. What is the difference between RDD, DataFrame, and Dataset in Spark?

**Answer:**
- **RDD (Resilient Distributed Dataset):** The fundamental data structure in Spark. It's a distributed collection of elements that can be processed in parallel. RDDs are immutable, fault-tolerant, and provide low-level control but lack schema information.

- **DataFrame:** A distributed collection of data organized into named columns, similar to a table in a relational database. DataFrames provide schema information, optimized execution through Catalyst optimizer, and are more user-friendly than RDDs. They're untyped at compile time.

- **Dataset:** A strongly-typed version of DataFrame where objects map to a specific JVM type. Datasets provide type safety at compile time and object-oriented programming interface. In Python, due to its dynamic nature, there's no distinction between DataFrame and Dataset (PySpark only has DataFrame API).

### 4. Explain transformations and actions in Spark with examples.

**Answer:**
- **Transformations:** Operations that create a new RDD/DataFrame from an existing one without executing computations immediately (lazy evaluation). Examples include:
  - `map()`, `filter()`, `flatMap()`, `groupBy()`, `join()`, `select()`, `withColumn()`

- **Actions:** Operations that trigger computation and return results to the driver program or write data to external storage. Examples include:
  - `collect()`, `count()`, `first()`, `take(n)`, `show()`, `save()`, `saveAsTextFile()`

Transformations are lazy and only build up the execution plan, while actions trigger the actual execution of the plan.

### 5. What is lazy evaluation in Spark and why is it important?

**Answer:** Lazy evaluation means that Spark delays the execution of transformations until an action is called. This is important because:

1. **Optimization:** Spark can analyze the entire chain of transformations and optimize the execution plan (e.g., combining multiple filters).
2. **Reduced computation:** Unnecessary computations can be avoided.
3. **Improved performance:** Data shuffling can be minimized by optimizing the execution plan.
4. **Memory efficiency:** Intermediate results don't need to be materialized unless necessary.

For example, if you chain multiple `filter()` operations, Spark will combine them into a single pass over the data rather than executing each filter separately.

### 6. How does Spark handle fault tolerance?

**Answer:** Spark achieves fault tolerance through:

1. **RDD Lineage:** Spark tracks the lineage (sequence of transformations) used to build each RDD. If a partition is lost, Spark can rebuild it by recomputing the lost data from the original source using this lineage information.

2. **Checkpointing:** For long lineage chains, Spark allows saving intermediate RDDs to disk to avoid recomputation from the beginning.

3. **Data replication:** In Spark's storage module, data can be replicated across nodes to prevent data loss.

4. **Executor failures:** If an executor fails, the driver reassigns its tasks to other executors.

5. **Driver recovery:** With cluster managers like YARN, if the driver fails, the entire application can be restarted from the last checkpoint.

### 7. Explain the difference between `map()` and `flatMap()` in Spark.

**Answer:**
- **map():** Applies a function to each element in the RDD/DataFrame and returns a new RDD/DataFrame of the same size. Each input item maps to exactly one output item.

```python
# map example
rdd = sc.parallelize([1, 2, 3])
mapped = rdd.map(lambda x: x * 2)  # Result: [2, 4, 6]
```

- **flatMap():** Similar to map(), but each input item can map to 0, 1, or more output items. It "flattens" the results into a single RDD/DataFrame.

```python
# flatMap example
rdd = sc.parallelize(["hello world", "spark python"])
flatmapped = rdd.flatMap(lambda x: x.split(" "))  # Result: ["hello", "world", "spark", "python"]
```

The key difference is that `flatMap()` flattens the result, which is useful for operations that produce variable-length output for each input element.

### 8. What is a broadcast variable in Spark and when would you use it?

**Answer:** A broadcast variable is a read-only variable that is cached on each worker node rather than shipped with each task. It's used to efficiently share large, immutable data across all nodes in a Spark cluster.

You would use broadcast variables when:
- You have a large lookup table or reference data that needs to be accessed by tasks across the cluster.
- The same data is used repeatedly across multiple operations.
- You want to avoid shipping the same data with every task, which can cause network overhead.

Example:
```python
# Without broadcast
large_dict = {"key1": "value1", "key2": "value2", ...}  # Large dictionary
rdd = sc.parallelize([1, 2, 3])
result = rdd.map(lambda x: large_dict.get(x, "default"))  # Dict is serialized with each task

# With broadcast
broadcast_dict = sc.broadcast(large_dict)
result = rdd.map(lambda x: broadcast_dict.value.get(x, "default"))  # Dict is broadcast once
```

### 9. What is data skew in Spark and how can you handle it?

**Answer:** Data skew occurs when data is unevenly distributed across partitions, causing some tasks to process significantly more data than others. This leads to performance bottlenecks as the job can only complete as fast as the slowest task.

Techniques to handle data skew:

1. **Salting/Key Redistribution:** Add random prefixes to keys in the skewed partition to distribute the data more evenly.
   ```python
   # Add salt to keys
   from pyspark.sql.functions import rand, concat, lit
   df = df.withColumn("salted_key", concat(lit(int(rand()*10)), lit("_"), df.skewed_key))
   ```

2. **Custom Partitioning:** Implement a custom partitioner that ensures even distribution.
   ```python
   # Custom partitioning
   df.repartition(col("custom_partition_key"))
   ```

3. **Broadcast Join:** For joins with a skewed distribution, broadcast the smaller DataFrame.
   ```python
   from pyspark.sql.functions import broadcast
   df_result = df_large.join(broadcast(df_small), "key")
   ```

4. **Separate Processing:** Identify and process skewed keys separately from the rest.

5. **Increase Parallelism:** Increase the number of partitions to distribute the data more evenly.
   ```python
   df = df.repartition(1000)
   ```

### 10. How would you optimize a slow-running Spark job?

**Answer:** To optimize a slow Spark job:

1. **Identify bottlenecks:**
   - Use Spark UI to identify stages with data skew or slow tasks
   - Check for shuffle operations which are expensive

2. **Data optimization:**
   - Use appropriate file formats (Parquet, ORC) with compression
   - Partition data appropriately
   - Filter data early to reduce processing volume

3. **Reduce shuffles:**
   - Use broadcast joins for small tables
   - Use `mapPartitions()` instead of `map()` to reduce overhead
   - Repartition data strategically

4. **Memory management:**
   - Adjust executor memory and cores
   - Use caching/persistence appropriately
   - Set the right serialization format

5. **Configuration tuning:**
   ```python
   spark.conf.set("spark.sql.shuffle.partitions", 200)  # Default is 200
   spark.conf.set("spark.default.parallelism", 100)
   spark.conf.set("spark.executor.memory", "10g")
   spark.conf.set("spark.driver.memory", "4g")
   ```

6. **Code optimization:**
   - Use DataFrame operations instead of RDDs when possible
   - Use built-in functions instead of UDFs when possible
   - Avoid `collect()` on large datasets

7. **Resource allocation:**
   - Adjust number of executors
   - Balance cores per executor

### 11. Explain the difference between `cache()` and `persist()` in Spark.

**Answer:** Both `cache()` and `persist()` are used to store intermediate results in memory, but they differ in storage level flexibility:

- **cache():** A shorthand for `persist(StorageLevel.MEMORY_ONLY)`. It stores the RDD/DataFrame only in memory.

- **persist():** Allows specifying the storage level, giving more control over how data is stored:
  - `MEMORY_ONLY`: Store in memory only (same as cache())
  - `MEMORY_AND_DISK`: Store in memory, spill to disk if needed
  - `MEMORY_ONLY_SER`: Store in memory as serialized objects
  - `MEMORY_AND_DISK_SER`: Store serialized objects in memory, spill to disk if needed
  - `DISK_ONLY`: Store only on disk
  - `OFF_HEAP`: Store in off-heap memory

Example:
```python
from pyspark.storagelevel import StorageLevel

# Using cache
df.cache()

# Using persist with different storage levels
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.DISK_ONLY)
```

### 12. What are accumulators in Spark and when would you use them?

**Answer:** Accumulators are variables that can be "accumulated" (added to) across the executors in a distributed Spark job and sent back to the driver program. They're useful for:

1. **Counters and sums:** Tracking metrics during job execution
2. **Debugging:** Counting events or errors during processing
3. **Custom aggregations:** Implementing custom reduction operations

Example:
```python
# Create an accumulator
error_count = spark.sparkContext.accumulator(0)

# Use in a transformation
def process_record(record):
    if not is_valid(record):
        error_count.add(1)
    return process_valid_record(record)

processed_rdd = rdd.map(process_record)
processed_rdd.count()  # Force evaluation

# Access the accumulator value (only reliable after an action)
print(f"Errors encountered: {error_count.value}")
```

Accumulators are only updated during actions, not transformations (due to lazy evaluation).

### 13. How do you handle streaming data in Spark?

**Answer:** Spark offers two APIs for streaming:

1. **Structured Streaming:** The newer, recommended API based on the DataFrame API:
   ```python
   # Read streaming data
   stream_df = spark.readStream \
       .format("kafka") \
       .option("kafka.bootstrap.servers", "host:port") \
       .option("subscribe", "topic") \
       .load()
   
   # Process data
   result = stream_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
   
   # Write results
   query = result.writeStream \
       .outputMode("append") \
       .format("console") \
       .start()
   
   query.awaitTermination()
   ```

2. **DStream API (older):** Based on RDDs:
   ```python
   from pyspark.streaming import StreamingContext
   
   ssc = StreamingContext(spark.sparkContext, batchDuration=1)
   lines = ssc.socketTextStream("localhost", 9999)
   word_counts = lines.flatMap(lambda line: line.split(" ")) \
       .map(lambda word: (word, 1)) \
       .reduceByKey(lambda a, b: a + b)
   
   word_counts.pprint()
   ssc.start()
   ssc.awaitTermination()
   ```

Key concepts in Spark Streaming:
- **Input sources:** Kafka, files, sockets, etc.
- **Output modes:** Append, update, complete
- **Watermarking:** For handling late data
- **Triggers:** Control batch timing
- **Checkpointing:** For fault tolerance

### 14. What is the difference between `join`, `broadcast join`, and `cross join` in Spark?

**Answer:**

1. **Regular Join:** Redistributes data across the cluster based on join keys.
   ```python
   # Regular join
   df1.join(df2, "key")  # Inner join by default
   df1.join(df2, "key", "left")  # Left outer join
   ```

2. **Broadcast Join:** Sends the smaller DataFrame to all nodes, avoiding shuffling of the larger DataFrame.
   ```python
   # Broadcast join
   from pyspark.sql.functions import broadcast
   df1.join(broadcast(df2), "key")
   ```
   Use when one DataFrame is small enough to fit in memory.

3. **Cross Join:** Cartesian product of two DataFrames (every row from first DF joined with every row from second DF).
   ```python
   # Cross join
   df1.crossJoin(df2)
   ```
   Results in m×n rows where m and n are the number of rows in the input DataFrames.

The choice impacts performance significantly:
- Regular joins require shuffling data, which is expensive for large datasets
- Broadcast joins avoid shuffling but require enough memory to hold the broadcast DataFrame
- Cross joins explode the data size and should be used carefully

### 15. How do you deploy a Spark application in production?

**Answer:** Deploying Spark applications in production involves several considerations:

1. **Packaging:**
   ```bash
   # Create a JAR or Python package
   sbt package  # For Scala
   # For Python, create a .zip or .egg file
   ```
