# 1. Imports
# ------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import log1p
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.clustering import BisectingKMeans
import matplotlib.pyplot as plt
import pandas as pd

# 2. Spark Session
# ---------------------------------------
spark = (
    SparkSession.builder
    .appName("LendingClub_Clustering")
    .master("local[*]")
    .getOrCreate()
)

# 3. Dataset Loading
# ----------------------------------------------
def load_datasets(spark, train_path, test_path):
    
    df_train = spark.read.csv(train_path, header=True, inferSchema=True)
    df_test = spark.read.csv(test_path, header=True, inferSchema=True)

    df_all = df_train.unionByName(df_test, allowMissingColumns=True)

    print("train rows:", df_train.count())
    print("test rows :", df_test.count())
    print("all rows  :", df_all.count())
    print("n_cols    :", len(df_all.columns))

    return df_train, df_test, df_all

train_path = "train.csv"
test_path = "test.csv"

# combine train and test to analyze borrower behaviour across the full dataset
df_train, df_test, df_all = load_datasets(spark, train_path, test_path)

# 4. Feature Selection
# --------------------------------------------------
def prepare_clustering_features(df):

    features = [
        "revol_util",
        "percent_bc_gt_75",
        "dti",
        "inq_last_6mths",
        "acc_open_past_24mths",
        "pct_tl_nvr_dlq",
        "num_actv_rev_tl",
        "credit_hist_months"
    ]

    df_cluster = df.select(features)

    df_cluster = df_cluster.withColumn(
        "log_credit_hist_months",
        log1p("credit_hist_months")
    )

    df_cluster = df_cluster.drop("credit_hist_months")

    return df_cluster

df_cluster = prepare_clustering_features(df_all)

feature_cols = [
    "revol_util",
    "percent_bc_gt_75",
    "dti",
    "inq_last_6mths",
    "acc_open_past_24mths",
    "pct_tl_nvr_dlq",
    "num_actv_rev_tl",
    "log_credit_hist_months"
]
df_cluster.select(feature_cols).describe().show()

def winsorize_df(df, cols, lower_q=0.01, upper_q=0.99):
    """
    Caps extreme values using winsorization.
    Values below the 1st percentile and above the 99th percentile
    are clipped to reduce the influence of outliers.
    """
    quantiles = {
        c: df.approxQuantile(c, [lower_q, upper_q], 0.001)
        for c in cols
    }

    # clip values
    for c in cols:
        lower, upper = quantiles[c]

        df = df.withColumn(
            c,
            F.when(F.col(c) < lower, lower)
             .when(F.col(c) > upper, upper)
             .otherwise(F.col(c))
        )

    return df
df_cluster = winsorize_df(df_cluster, feature_cols)
for c in feature_cols:
    print(c, df_cluster.approxQuantile(c, [0.01, 0.99], 0.001))

def scale_features(df, feature_cols):
    """
    Assembles numerical features and applies standard scaling
    """
    # assemble feature vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )

    df_vec = assembler.transform(df)

    # standard scaling
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(df_vec)

    df_scaled = scaler_model.transform(df_vec)

    return df_scaled

df_scaled = scale_features(df_cluster, feature_cols)

# 5. PCA
# ------------------------------------------------
def apply_pca(df, k=2):

    pca = PCA(
        k=k,
        inputCol="features_scaled",
        outputCol="pca_features"
    )

    pca_model = pca.fit(df)
    df_pca = pca_model.transform(df)

    return df_pca, pca_model
df_pca, pca_model = apply_pca(df_scaled, k=2)
print("Explained variance:", pca_model.explainedVariance.toArray())

# 6. KMeans Evaluation
# ------------------------------------------------------
def evaluate_kmeans_range(df, k_values, features_col="features_scaled", seed=42):
    sse_list = []
    silhouette_list = []

    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    for k in k_values:
        kmeans = KMeans(
            k=k,
            seed=seed,
            featuresCol=features_col,
            predictionCol="cluster"
        )

        model = kmeans.fit(df)
        pred = model.transform(df)

        sse_list.append(model.summary.trainingCost)
        silhouette_list.append(evaluator.evaluate(pred))

    results_df = pd.DataFrame({
        "k": list(k_values),
        "SSE": sse_list,
        "Silhouette": silhouette_list
    })

    return results_df


ks = range(2, 12)
results_df = evaluate_kmeans_range(df_scaled, ks)

print(results_df)

# SSE plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["SSE"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method using SSE")
plt.xticks(results_df["k"])
plt.grid(True)
plt.show()

# Silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["Silhouette"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.xticks(results_df["k"])
plt.grid(True)
plt.show()

kmeans_final = KMeans(
    k=3,
    seed=42,
    featuresCol="features_scaled",
    predictionCol="cluster"
)

kmeans_model = kmeans_final.fit(df_scaled)
df_kmeans = kmeans_model.transform(df_scaled)

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

final_sse = kmeans_model.summary.trainingCost
final_silhouette = evaluator.evaluate(df_kmeans)

print("Final KMeans Evaluation (k=3)")
print("SSE:", final_sse)
print("Silhouette:", final_silhouette)

df_kmeans.groupBy("cluster").count().orderBy("cluster").show()

df_kmeans.groupBy("cluster").agg(
    F.mean("revol_util").alias("avg_revol_util"),
    F.mean("percent_bc_gt_75").alias("avg_percent_bc_gt_75"),
    F.mean("dti").alias("avg_dti"),
    F.mean("inq_last_6mths").alias("avg_inq_last_6mths"),
    F.mean("acc_open_past_24mths").alias("avg_acc_open_past_24mths"),
    F.mean("pct_tl_nvr_dlq").alias("avg_pct_tl_nvr_dlq"),
    F.mean("num_actv_rev_tl").alias("avg_num_actv_rev_tl"),
    F.mean("log_credit_hist_months").alias("avg_log_credit_hist_months")
).orderBy("cluster").show(truncate=False)

df_kmeans_pca = kmeans_model.transform(df_pca)

# sample 50% of points for plotting to avoid heavy visualization
pca_pd = df_kmeans_pca.select("pca_features", "cluster").sample(0.5, seed=42).toPandas()

pca_pd["PC1"] = pca_pd["pca_features"].apply(lambda x: float(x[0]))
pca_pd["PC2"] = pca_pd["pca_features"].apply(lambda x: float(x[1]))

plt.figure(figsize=(7, 6))

for c in sorted(pca_pd["cluster"].unique()):
    temp = pca_pd[pca_pd["cluster"] == c]
    plt.scatter(temp["PC1"], temp["PC2"], s=5, label=f"Cluster {c}", alpha=0.5)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters in PCA Space (k=3)")
plt.legend()
plt.show()

df_all_id = df_all.withColumn("row_id", monotonically_increasing_id())
df_cluster_id = df_cluster.withColumn("row_id", monotonically_increasing_id())
df_kmeans_id = df_kmeans.withColumn("row_id", monotonically_increasing_id())

df_eval = df_all_id.join(
    df_kmeans_id.select("row_id", "cluster"),
    on="row_id",
    how="inner"
)

df_eval.groupBy("cluster").agg(
    F.count("*").alias("n"),
    F.mean("fico_avg").alias("avg_fico"),
    F.min("fico_avg").alias("min_fico"),
    F.max("fico_avg").alias("max_fico")
).orderBy("cluster").show()

df_eval.groupBy("cluster").agg(
    F.mean("int_rate").alias("avg_interest"),
    F.min("int_rate").alias("min_interest"),
    F.max("int_rate").alias("max_interest")
).orderBy("cluster").show()

df_eval.groupBy("cluster","grade") \
    .count() \
    .orderBy("cluster","grade") \
    .show(100, truncate=False)

# 7. Bisecting KMeans
# --------------------------------------------------
def evaluate_bisecting_kmeans_range(df, k_values, features_col="features_scaled", seed=42):
    sse_list = []
    silhouette_list = []

    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    for k in k_values:
        bkmeans = BisectingKMeans(
            k=k,
            seed=seed,
            featuresCol=features_col,
            predictionCol="cluster"
        )

        model = bkmeans.fit(df)
        pred = model.transform(df)

        sse_list.append(model.summary.trainingCost)
        silhouette_list.append(evaluator.evaluate(pred))

    results_df = pd.DataFrame({
        "k": list(k_values),
        "SSE": sse_list,
        "Silhouette": silhouette_list
    })

    return results_df

ks = range(2, 12)
bk_results_df = evaluate_bisecting_kmeans_range(df_scaled, ks)

print(bk_results_df.round(4))

plt.figure(figsize=(8, 5))
plt.plot(bk_results_df["k"], bk_results_df["SSE"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE")
plt.title("Bisecting K-Means: Elbow Method using SSE")
plt.xticks(bk_results_df["k"])
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(bk_results_df["k"], bk_results_df["Silhouette"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Bisecting K-Means: Silhouette Score by Number of Clusters")
plt.xticks(bk_results_df["k"])
plt.grid(True)
plt.show()

bkmeans = BisectingKMeans(
    k=3,
    seed=42,
    featuresCol="features_scaled",
    predictionCol="cluster"
)

bkmeans_model = bkmeans.fit(df_scaled)
df_bkmeans = bkmeans_model.transform(df_scaled)

evaluator = ClusteringEvaluator(
    featuresCol="features_scaled",
    predictionCol="cluster",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

bk_sse = bkmeans_model.summary.trainingCost
bk_silhouette = evaluator.evaluate(df_bkmeans)

print("Bisecting KMeans Results (k=3)")
print("SSE:", bk_sse)
print("Silhouette:", bk_silhouette)

df_bkmeans.groupBy("cluster") \
    .count() \
    .orderBy("cluster") \
    .show()

df_bkmeans.groupBy("cluster").agg(
    F.mean("revol_util").alias("avg_revol_util"),
    F.mean("percent_bc_gt_75").alias("avg_percent_bc_gt_75"),
    F.mean("dti").alias("avg_dti"),
    F.mean("inq_last_6mths").alias("avg_inq_last_6mths"),
    F.mean("acc_open_past_24mths").alias("avg_acc_open_past_24mths"),
    F.mean("pct_tl_nvr_dlq").alias("avg_pct_tl_nvr_dlq"),
    F.mean("num_actv_rev_tl").alias("avg_num_actv_rev_tl"),
    F.mean("log_credit_hist_months").alias("avg_log_credit_hist_months")
).orderBy("cluster").show(truncate=False)

df_all_id = df_all.withColumn("row_id", monotonically_increasing_id())
df_bkmeans_id = df_bkmeans.withColumn("row_id", monotonically_increasing_id())

df_eval_bk = df_all_id.join(
    df_bkmeans_id.select("row_id","cluster"),
    on="row_id",
    how="inner"
)

df_eval_bk.groupBy("cluster").agg(
    F.count("*").alias("n"),
    F.mean("fico_avg").alias("avg_fico"),
    F.min("fico_avg").alias("min_fico"),
    F.max("fico_avg").alias("max_fico")
).orderBy("cluster").show()

df_eval_bk.groupBy("cluster").agg(
    F.mean("int_rate").alias("avg_interest"),
    F.min("int_rate").alias("min_interest"),
    F.max("int_rate").alias("max_interest")
).orderBy("cluster").show()

df_eval_bk.groupBy("cluster","grade") \
    .count() \
    .orderBy("cluster","grade") \
    .show(100, truncate=False)

pca = PCA(
    k=2,
    inputCol="features_scaled",
    outputCol="pca_features"
)

pca_model = pca.fit(df_scaled)

df_bkmeans_pca = pca_model.transform(df_bkmeans)
pca_pd_bk = df_bkmeans_pca.select("pca_features", "cluster").toPandas()
pca_pd_bk["PC1"] = pca_pd_bk["pca_features"].apply(lambda x: float(x[0]))
pca_pd_bk["PC2"] = pca_pd_bk["pca_features"].apply(lambda x: float(x[1]))

plt.figure(figsize=(7,6))

for c in sorted(pca_pd_bk["cluster"].unique()):
    temp = pca_pd_bk[pca_pd_bk["cluster"] == c]

    plt.scatter(
        temp["PC1"],
        temp["PC2"],
        s=5,
        alpha=0.5,
        label=f"Cluster {c}"
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Bisecting KMeans Clusters in PCA Space (k=3)")
plt.legend()
plt.show()

