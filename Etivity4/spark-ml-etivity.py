#!/usr/bin/env python3
"""
Etivity 4 - Implementare la regressione lineare per la classificazione
Implementare la classificazione con gli alberi di decisione
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np
import matplotlib.pyplot as plt

# Creazione sessione Spark
print("Inizializzazione Spark Session...")
spark = SparkSession.builder \
    .appName("Etivity4_ML_Spark") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

print(f"Versione Spark: {spark.version}")
print("-" * 80)

# ==============================================================================
# PARTE 1: PREPARAZIONE DEI DATI
# ==============================================================================

print("\n1. PREPARAZIONE DEI DATI")
print("-" * 50)

# Creazione dataset per classificazione binaria (esempio: previsione churn clienti)
print("\n1.1 Creazione dataset per classificazione")
classification_data = [
    # (età, reddito, mesi_cliente, num_prodotti, ha_carta_credito, è_attivo, churn)
    (25, 35000, 12, 2, 1, 1, 0),
    (30, 45000, 24, 3, 1, 1, 0),
    (35, 55000, 36, 4, 1, 1, 0),
    (28, 32000, 6, 1, 0, 0, 1),
    (45, 75000, 48, 5, 1, 1, 0),
    (22, 28000, 3, 1, 0, 0, 1),
    (38, 62000, 30, 3, 1, 1, 0),
    (50, 85000, 60, 4, 1, 1, 0),
    (27, 30000, 8, 1, 0, 1, 1),
    (33, 48000, 18, 2, 1, 0, 1),
    (41, 68000, 42, 3, 1, 1, 0),
    (29, 34000, 10, 2, 0, 0, 1),
    (36, 58000, 28, 3, 1, 1, 0),
    (24, 26000, 4, 1, 0, 0, 1),
    (47, 78000, 54, 5, 1, 1, 0),
    (31, 42000, 15, 2, 1, 0, 1),
    (39, 65000, 33, 4, 1, 1, 0),
    (26, 31000, 7, 1, 0, 0, 1),
    (43, 71000, 45, 4, 1, 1, 0),
    (34, 52000, 20, 3, 1, 1, 0),
]

columns_classification = ["eta", "reddito", "mesi_cliente", "num_prodotti", 
                         "ha_carta_credito", "e_attivo", "churn"]

df_classification = spark.createDataFrame(classification_data, columns_classification)

print("Dataset classificazione creato:")
df_classification.show(10)

# Creazione dataset per regressione (esempio: previsione prezzo casa)
print("\n1.2 Creazione dataset per regressione")
regression_data = [
    # (mq, num_stanze, num_bagni, anno_costruzione, ha_garage, distanza_centro_km, prezzo)
    (80, 2, 1, 1990, 0, 5.5, 180000),
    (120, 3, 2, 2005, 1, 3.2, 320000),
    (150, 4, 2, 2010, 1, 2.8, 420000),
    (65, 1, 1, 1985, 0, 7.2, 150000),
    (100, 3, 1, 1995, 1, 4.5, 250000),
    (180, 5, 3, 2015, 1, 1.5, 580000),
    (90, 2, 1, 2000, 0, 6.0, 210000),
    (140, 4, 2, 2008, 1, 3.5, 380000),
    (70, 2, 1, 1988, 0, 8.0, 160000),
    (110, 3, 2, 2003, 1, 4.0, 290000),
    (200, 5, 3, 2018, 1, 2.0, 650000),
    (85, 2, 1, 1992, 0, 5.8, 190000),
    (130, 3, 2, 2007, 1, 3.8, 350000),
    (160, 4, 3, 2012, 1, 2.5, 480000),
    (75, 2, 1, 1987, 0, 7.5, 170000),
    (95, 3, 1, 1998, 0, 5.2, 230000),
    (170, 4, 3, 2014, 1, 2.2, 520000),
    (105, 3, 2, 2002, 1, 4.2, 280000),
    (145, 4, 2, 2009, 1, 3.0, 400000),
    (55, 1, 1, 1980, 0, 9.0, 120000),
]

columns_regression = ["mq", "num_stanze", "num_bagni", "anno_costruzione", 
                     "ha_garage", "distanza_centro_km", "prezzo"]

df_regression = spark.createDataFrame(regression_data, columns_regression)

print("Dataset regressione creato:")
df_regression.show(10)

# ==============================================================================
# PARTE 2: PREPROCESSING E FEATURE ENGINEERING
# ==============================================================================

print("\n2. PREPROCESSING E FEATURE ENGINEERING")
print("-" * 50)

# Funzione helper per preparare i dati
def prepare_features(df, feature_cols, label_col, is_classification=True):
    """Prepara le features per il modello ML"""
    
    # 1. Assembla le features in un vettore
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )
    
    # 2. Scala le features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    # 3. Pipeline di preprocessing
    pipeline = Pipeline(stages=[assembler, scaler])
    
    # 4. Fit e transform
    model = pipeline.fit(df)
    df_prepared = model.transform(df)
    
    # 5. Rinomina la colonna label se necessario
    if is_classification:
        df_prepared = df_prepared.withColumn("label", col(label_col).cast("double"))
    else:
        df_prepared = df_prepared.withColumnRenamed(label_col, "label")
    
    return df_prepared.select("features", "label", *feature_cols, label_col)

# Prepara dati per classificazione
feature_cols_class = ["eta", "reddito", "mesi_cliente", "num_prodotti", 
                     "ha_carta_credito", "e_attivo"]
df_class_prepared = prepare_features(df_classification, feature_cols_class, "churn", True)

print("Dataset classificazione preparato:")
df_class_prepared.select("features", "label").show(5, truncate=False)

# Prepara dati per regressione
feature_cols_reg = ["mq", "num_stanze", "num_bagni", "anno_costruzione", 
                   "ha_garage", "distanza_centro_km"]
df_reg_prepared = prepare_features(df_regression, feature_cols_reg, "prezzo", False)

print("\nDataset regressione preparato:")
df_reg_prepared.select("features", "label").show(5, truncate=False)

# Split train/test
train_class, test_class = df_class_prepared.randomSplit([0.7, 0.3], seed=42)
train_reg, test_reg = df_reg_prepared.randomSplit([0.7, 0.3], seed=42)

print(f"\nDimensioni dataset classificazione - Train: {train_class.count()}, Test: {test_class.count()}")
print(f"Dimensioni dataset regressione - Train: {train_reg.count()}, Test: {test_reg.count()}")

# ==============================================================================
# PARTE 3: REGRESSIONE LOGISTICA PER CLASSIFICAZIONE
# ==============================================================================

print("\n3. REGRESSIONE LOGISTICA PER CLASSIFICAZIONE")
print("-" * 50)

# 3.1 Creazione e training del modello
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.8,
    family="binomial"
)

print("Training modello di regressione logistica...")
lr_model = lr.fit(train_class)

# 3.2 Coefficienti del modello
print(f"\nCoefficenti: {lr_model.coefficients}")
print(f"Intercetta: {lr_model.intercept}")

# 3.3 Predizioni
predictions_lr = lr_model.transform(test_class)
print("\nPredizioni regressione logistica:")
predictions_lr.select("label", "prediction", "probability").show(10)

# 3.4 Valutazione del modello
evaluator_binary = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

evaluator_multi = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

auc = evaluator_binary.evaluate(predictions_lr)
accuracy = evaluator_multi.evaluate(predictions_lr, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions_lr, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions_lr, {evaluator_multi.metricName: "weightedRecall"})
f1 = evaluator_multi.evaluate(predictions_lr, {evaluator_multi.metricName: "f1"})

print("\n3.5 Metriche Regressione Logistica:")
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ==============================================================================
# PARTE 4: ALBERO DI DECISIONE PER CLASSIFICAZIONE
# ==============================================================================

print("\n4. ALBERO DI DECISIONE PER CLASSIFICAZIONE")
print("-" * 50)

# 4.1 Creazione e training del modello
dt_classifier = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    maxBins=32,
    minInstancesPerNode=1,
    minInfoGain=0.0,
    impurity="gini"
)

print("Training albero di decisione per classificazione...")
dt_class_model = dt_classifier.fit(train_class)

# 4.2 Informazioni sull'albero
print(f"\nProfondità albero: {dt_class_model.depth}")
print(f"Numero di nodi: {dt_class_model.numNodes}")

# 4.3 Feature importance
feature_importance = dt_class_model.featureImportances
print("\nImportanza delle features:")
for i, importance in enumerate(feature_importance):
    if importance > 0:
        print(f"Feature {i} ({feature_cols_class[i]}): {importance:.4f}")

# 4.4 Predizioni
predictions_dt = dt_class_model.transform(test_class)
print("\nPredizioni albero di decisione:")
predictions_dt.select("label", "prediction", "probability").show(10)

# 4.5 Valutazione
auc_dt = evaluator_binary.evaluate(predictions_dt)
accuracy_dt = evaluator_multi.evaluate(predictions_dt, {evaluator_multi.metricName: "accuracy"})
precision_dt = evaluator_multi.evaluate(predictions_dt, {evaluator_multi.metricName: "weightedPrecision"})
recall_dt = evaluator_multi.evaluate(predictions_dt, {evaluator_multi.metricName: "weightedRecall"})
f1_dt = evaluator_multi.evaluate(predictions_dt, {evaluator_multi.metricName: "f1"})

print("\n4.6 Metriche Albero di Decisione:")
print(f"AUC-ROC: {auc_dt:.4f}")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-Score: {f1_dt:.4f}")

# ==============================================================================
# PARTE 5: REGRESSIONE LINEARE
# ==============================================================================

print("\n5. REGRESSIONE LINEARE")
print("-" * 50)

# 5.1 Creazione e training del modello
linear_reg = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.8
)

print("Training modello di regressione lineare...")
linear_model = linear_reg.fit(train_reg)

# 5.2 Coefficienti del modello
print(f"\nCoefficenti: {linear_model.coefficients}")
print(f"Intercetta: {linear_model.intercept}")

# 5.3 Predizioni
predictions_linear = linear_model.transform(test_reg)
print("\nPredizioni regressione lineare:")
predictions_linear.select("label", "prediction", "mq", "num_stanze", "distanza_centro_km").show(10)

# 5.4 Valutazione
evaluator_reg = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

rmse = evaluator_reg.evaluate(predictions_linear, {evaluator_reg.metricName: "rmse"})
mae = evaluator_reg.evaluate(predictions_linear, {evaluator_reg.metricName: "mae"})
r2 = evaluator_reg.evaluate(predictions_linear, {evaluator_reg.metricName: "r2"})

print("\n5.5 Metriche Regressione Lineare:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# ==============================================================================
# PARTE 6: ALBERO DI DECISIONE PER REGRESSIONE
# ==============================================================================

print("\n6. ALBERO DI DECISIONE PER REGRESSIONE")
print("-" * 50)

# 6.1 Creazione e training del modello
dt_regressor = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    maxBins=32
)

print("Training albero di decisione per regressione...")
dt_reg_model = dt_regressor.fit(train_reg)

# 6.2 Informazioni sull'albero
print(f"\nProfondità albero: {dt_reg_model.depth}")
print(f"Numero di nodi: {dt_reg_model.numNodes}")

# 6.3 Feature importance
feature_importance_reg = dt_reg_model.featureImportances
print("\nImportanza delle features:")
for i, importance in enumerate(feature_importance_reg):
    if importance > 0:
        print(f"Feature {i} ({feature_cols_reg[i]}): {importance:.4f}")

# 6.4 Predizioni
predictions_dt_reg = dt_reg_model.transform(test_reg)
print("\nPredizioni albero di decisione regressione:")
predictions_dt_reg.select("label", "prediction", "mq", "num_stanze", "distanza_centro_km").show(10)

# 6.5 Valutazione
rmse_dt = evaluator_reg.evaluate(predictions_dt_reg, {evaluator_reg.metricName: "rmse"})
mae_dt = evaluator_reg.evaluate(predictions_dt_reg, {evaluator_reg.metricName: "mae"})
r2_dt = evaluator_reg.evaluate(predictions_dt_reg, {evaluator_reg.metricName: "r2"})

print("\n6.6 Metriche Albero di Decisione Regressione:")
print(f"RMSE: {rmse_dt:.2f}")
print(f"MAE: {mae_dt:.2f}")
print(f"R²: {r2_dt:.4f}")

# ==============================================================================
# PARTE 7: CROSS-VALIDATION E HYPERPARAMETER TUNING
# ==============================================================================

print("\n7. CROSS-VALIDATION E HYPERPARAMETER TUNING")
print("-" * 50)

# 7.1 Cross-validation per Regressione Logistica
print("\n7.1 Ottimizzazione Regressione Logistica")

paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

cv_lr = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid_lr,
    evaluator=evaluator_binary,
    numFolds=3
)

print("Esecuzione cross-validation...")
cv_model_lr = cv_lr.fit(train_class)
best_model_lr = cv_model_lr.bestModel

print(f"Migliori parametri trovati:")
print(f"regParam: {best_model_lr._java_obj.getRegParam()}")
print(f"elasticNetParam: {best_model_lr._java_obj.getElasticNetParam()}")

# 7.2 Cross-validation per Albero di Decisione
print("\n7.2 Ottimizzazione Albero di Decisione")

paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt_classifier.maxDepth, [3, 5, 7]) \
    .addGrid(dt_classifier.minInstancesPerNode, [1, 2, 5]) \
    .build()

cv_dt = CrossValidator(
    estimator=dt_classifier,
    estimatorParamMaps=paramGrid_dt,
    evaluator=evaluator_binary,
    numFolds=3
)

print("Esecuzione cross-validation...")
cv_model_dt = cv_dt.fit(train_class)
best_model_dt = cv_model_dt.bestModel

print(f"Migliori parametri trovati:")
print(f"maxDepth: {best_model_dt._java_obj.getMaxDepth()}")
print(f"minInstancesPerNode: {best_model_dt._java_obj.getMinInstancesPerNode()}")

# ==============================================================================
# PARTE 8: CONFRONTO FINALE DEI MODELLI
# ==============================================================================

print("\n8. CONFRONTO FINALE DEI MODELLI")
print("-" * 50)

print("\n8.1 Confronto modelli di classificazione:")
print(f"{'Modello':<25} {'AUC-ROC':<10} {'Accuracy':<10} {'F1-Score':<10}")
print("-" * 55)
print(f"{'Regressione Logistica':<25} {auc:<10.4f} {accuracy:<10.4f} {f1:<10.4f}")
print(f"{'Albero di Decisione':<25} {auc_dt:<10.4f} {accuracy_dt:<10.4f} {f1_dt:<10.4f}")

print("\n8.2 Confronto modelli di regressione:")
print(f"{'Modello':<25} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
print("-" * 65)
print(f"{'Regressione Lineare':<25} {rmse:<15.2f} {mae:<15.2f} {r2:<10.4f}")
print(f"{'Albero di Decisione':<25} {rmse_dt:<15.2f} {mae_dt:<15.2f} {r2_dt:<10.4f}")

# ==============================================================================
# PARTE 9: SALVATAGGIO MODELLI
# ==============================================================================

print("\n9. SALVATAGGIO MODELLI")
print("-" * 50)

# Salva i modelli migliori
lr_model.save("models/logistic_regression_model")
print("Modello regressione logistica salvato in: models/logistic_regression_model")

dt_class_model.save("models/decision_tree_classifier_model")
print("Modello albero decisione classificazione salvato in: models/decision_tree_classifier_model")

linear_model.save("models/linear_regression_model")
print("Modello regressione lineare salvato in: models/linear_regression_model")

dt_reg_model.save("models/decision_tree_regressor_model")
print("Modello albero decisione regressione salvato in: models/decision_tree_regressor_model")

# ==============================================================================
# PARTE 10: ESEMPIO DI UTILIZZO DEI MODELLI SALVATI
# ==============================================================================

print("\n10. ESEMPIO DI CARICAMENTO E UTILIZZO MODELLI")
print("-" * 50)

# Esempio: nuovo cliente per predizione churn
nuovo_cliente = spark.createDataFrame([
    (32, 50000, 20, 3, 1, 1)
], ["eta", "reddito", "mesi_cliente", "num_prodotti", "ha_carta_credito", "e_attivo"])

# Prepara features
nuovo_cliente_prep = prepare_features(
    nuovo_cliente, 
    feature_cols_class, 
    "e_attivo",  # dummy label column
    True
)

# Predizione
predizione = lr_model.transform(nuovo_cliente_prep)
print("\nPredizione churn per nuovo cliente:")
predizione.select("eta", "reddito", "mesi_cliente", "prediction", "probability").show()

# Chiusura sessione Spark
print("\nChiusura Spark Session...")
spark.stop()

print("\nProgramma completato con successo!")