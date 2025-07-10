#!/usr/bin/env python3
"""
Etivity 3 - Esercizi con SparkSQL e Python
1. Installare Spark e Python ed aprire una sessione SparkSQL.
2. Creare un DataFrame
a. Da una lista Python
b. Da un file csv
c. Da un file jason
3. Inserire la riga di intestazione
4. Visualizzare il dataset creato
5. Visualizzare la struttura
6. Selezione delle colonne.
7. Definire una colonna numerica e su cui fare operazioni di aggregazione dopo l’operazione
di group by
8. 9. Implementare le operazioni di select e filter
Fare un esempio di operazioni di ordinamento
10. Richiamare da SparkSQL istruzioni SQL
"""

# 1. INSTALLAZIONE E IMPORT
# Prima di eseguire: pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import col, sum, avg, max, min, count
import json

# Creazione della sessione Spark
print("1. Creazione sessione SparkSQL...")
spark = SparkSession.builder \
    .appName("Etivity3_SparkSQL") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

print(f"Versione Spark: {spark.version}")
print("-" * 50)

# 2. CREAZIONE DATAFRAME

# 2a. DataFrame da lista Python
print("\n2a. Creazione DataFrame da lista Python")
data_list = [
    ("Mario", "Rossi", 25, "IT", 2500.0),
    ("Luigi", "Verdi", 30, "IT", 3000.0),
    ("Anna", "Bianchi", 28, "HR", 2800.0),
    ("Giuseppe", "Nero", 35, "IT", 3500.0),
    ("Maria", "Gialli", 32, "HR", 3200.0),
    ("Francesco", "Blu", 27, "Sales", 2700.0),
    ("Paola", "Rosa", 29, "IT", 2900.0),
    ("Giovanni", "Viola", 33, "Sales", 3300.0)
]

# Schema esplicito
schema = StructType([
    StructField("nome", StringType(), True),
    StructField("cognome", StringType(), True),
    StructField("eta", IntegerType(), True),
    StructField("dipartimento", StringType(), True),
    StructField("stipendio", FloatType(), True)
])

df_from_list = spark.createDataFrame(data_list, schema=schema)
print("DataFrame da lista creato!")

# 2b. DataFrame da file CSV
print("\n2b. Creazione DataFrame da file CSV")
# Creiamo prima un file CSV di esempio
csv_data = """id,prodotto,categoria,prezzo,quantita
1,Laptop,Elettronica,899.99,10
2,Mouse,Elettronica,29.99,50
3,Scrivania,Mobili,299.99,5
4,Sedia,Mobili,149.99,20
5,Monitor,Elettronica,199.99,15
6,Tastiera,Elettronica,49.99,30
7,Lampada,Mobili,39.99,25
8,Webcam,Elettronica,79.99,12"""

with open("prodotti.csv", "w") as f:
    f.write(csv_data)

# Lettura del CSV con header
df_from_csv = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("prodotti.csv")

print("DataFrame da CSV creato!")

# 2c. DataFrame da file JSON
print("\n2c. Creazione DataFrame da file JSON")
# Creiamo un file JSON di esempio
json_data = [
    {"cliente": "Cliente A", "ordine_id": 1001, "importo": 150.50, "stato": "Completato"},
    {"cliente": "Cliente B", "ordine_id": 1002, "importo": 299.99, "stato": "In Processo"},
    {"cliente": "Cliente A", "ordine_id": 1003, "importo": 450.00, "stato": "Completato"},
    {"cliente": "Cliente C", "ordine_id": 1004, "importo": 89.99, "stato": "Annullato"},
    {"cliente": "Cliente B", "ordine_id": 1005, "importo": 199.99, "stato": "Completato"}
]

with open("ordini.json", "w") as f:
    for record in json_data:
        f.write(json.dumps(record) + "\n")

df_from_json = spark.read.json("ordini.json")
print("DataFrame da JSON creato!")

# 3. INSERIRE RIGA DI INTESTAZIONE (già fatto con header=true per CSV)
print("\n3. Le righe di intestazione sono già presenti nei DataFrame")
print("Colonne df_from_list:", df_from_list.columns)
print("Colonne df_from_csv:", df_from_csv.columns)
print("Colonne df_from_json:", df_from_json.columns)

# 4. VISUALIZZARE I DATASET CREATI
print("\n4. Visualizzazione dei dataset:")
print("\nDataFrame da Lista (Dipendenti):")
df_from_list.show(5)

print("\nDataFrame da CSV (Prodotti):")
df_from_csv.show(5)

print("\nDataFrame da JSON (Ordini):")
df_from_json.show()

# 5. VISUALIZZARE LA STRUTTURA
print("\n5. Struttura dei DataFrame:")
print("\nStruttura DataFrame Dipendenti:")
df_from_list.printSchema()

print("\nStruttura DataFrame Prodotti:")
df_from_csv.printSchema()

print("\nStruttura DataFrame Ordini:")
df_from_json.printSchema()

# 6. SELEZIONE DELLE COLONNE
print("\n6. Selezione delle colonne:")
print("\nSelezione nome e stipendio dei dipendenti:")
df_from_list.select("nome", "stipendio").show(5)

print("\nSelezione prodotto e prezzo:")
df_from_csv.select(col("prodotto"), col("prezzo")).show(5)

# 7. COLONNA NUMERICA E OPERAZIONI DI AGGREGAZIONE CON GROUP BY
print("\n7. Operazioni di aggregazione dopo GROUP BY:")

# Aggregazione per dipartimento
print("\nStatistiche stipendi per dipartimento:")
df_aggregated = df_from_list.groupBy("dipartimento") \
    .agg(
        count("*").alias("num_dipendenti"),
        avg("stipendio").alias("stipendio_medio"),
        max("stipendio").alias("stipendio_max"),
        min("stipendio").alias("stipendio_min"),
        sum("stipendio").alias("totale_stipendi")
    )
df_aggregated.show()

# Aggregazione per categoria prodotti
print("\nStatistiche prodotti per categoria:")
df_from_csv.groupBy("categoria") \
    .agg(
        count("*").alias("num_prodotti"),
        avg("prezzo").alias("prezzo_medio"),
        sum("quantita").alias("quantita_totale")
    ) \
    .show()

# 8. OPERAZIONI SELECT E FILTER
print("\n8. Operazioni SELECT e FILTER:")

# Filter su dipendenti
print("\nDipendenti del dipartimento IT con stipendio > 2800:")
df_filtered = df_from_list \
    .filter((col("dipartimento") == "IT") & (col("stipendio") > 2800)) \
    .select("nome", "cognome", "stipendio")
df_filtered.show()

# Filter su prodotti
print("\nProdotti di Elettronica con prezzo < 100:")
df_from_csv \
    .filter((col("categoria") == "Elettronica") & (col("prezzo") < 100)) \
    .select("prodotto", "prezzo", "quantita") \
    .show()

# 9. OPERAZIONI DI ORDINAMENTO
print("\n9. Operazioni di ordinamento:")

# Ordinamento crescente per stipendio
print("\nDipendenti ordinati per stipendio (crescente):")
df_from_list \
    .orderBy("stipendio") \
    .select("nome", "cognome", "stipendio") \
    .show()

# Ordinamento decrescente per prezzo
print("\nProdotti ordinati per prezzo (decrescente):")
df_from_csv \
    .orderBy(col("prezzo").desc()) \
    .show()

# Ordinamento multiplo
print("\nDipendenti ordinati per dipartimento e stipendio:")
df_from_list \
    .orderBy("dipartimento", col("stipendio").desc()) \
    .select("nome", "dipartimento", "stipendio") \
    .show()

# 10. ISTRUZIONI SQL IN SPARKSQL
print("\n10. Esecuzione di query SQL:")

# Registrazione dei DataFrame come tabelle temporanee
df_from_list.createOrReplaceTempView("dipendenti")
df_from_csv.createOrReplaceTempView("prodotti")
df_from_json.createOrReplaceTempView("ordini")

# Query SQL 1: Selezione con WHERE
print("\nQuery SQL 1 - Dipendenti giovani con stipendio alto:")
query1 = """
    SELECT nome, cognome, eta, stipendio
    FROM dipendenti
    WHERE eta < 30 AND stipendio > 2600
    ORDER BY stipendio DESC
"""
spark.sql(query1).show()

# Query SQL 2: JOIN virtuale (self-join per esempio)
print("\nQuery SQL 2 - Analisi stipendi per dipartimento:")
query2 = """
    SELECT 
        dipartimento,
        COUNT(*) as num_dipendenti,
        ROUND(AVG(stipendio), 2) as stipendio_medio,
        MAX(stipendio) as max_stipendio,
        MIN(stipendio) as min_stipendio
    FROM dipendenti
    GROUP BY dipartimento
    HAVING AVG(stipendio) > 2800
    ORDER BY stipendio_medio DESC
"""
spark.sql(query2).show()

# Query SQL 3: Subquery
print("\nQuery SQL 3 - Prodotti con prezzo sopra la media:")
query3 = """
    SELECT prodotto, categoria, prezzo
    FROM prodotti
    WHERE prezzo > (SELECT AVG(prezzo) FROM prodotti)
    ORDER BY prezzo DESC
"""
spark.sql(query3).show()

# Query SQL 4: CASE WHEN
print("\nQuery SQL 4 - Classificazione ordini per importo:")
query4 = """
    SELECT 
        cliente,
        ordine_id,
        importo,
        stato,
        CASE 
            WHEN importo < 100 THEN 'Piccolo'
            WHEN importo BETWEEN 100 AND 300 THEN 'Medio'
            ELSE 'Grande'
        END as tipo_ordine
    FROM ordini
    WHERE stato = 'Completato'
"""
spark.sql(query4).show()

# Query SQL 5: Window Functions (se supportate)
print("\nQuery SQL 5 - Ranking dipendenti per stipendio nel dipartimento:")
query5 = """
    SELECT 
        nome,
        dipartimento,
        stipendio,
        RANK() OVER (PARTITION BY dipartimento ORDER BY stipendio DESC) as rank_stipendio
    FROM dipendenti
"""
spark.sql(query5).show()

# Salva in formato CSV
print("Salvataggio in formato CSV...")
df_filtered.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("output/dipendenti_it_filtrati.csv")

print("\nProgramma completato con successo!")

# Chiusura della sessione Spark
spark.stop()

# Pulizia file temporanei
import os
os.remove("prodotti.csv")
os.remove("ordini.json")