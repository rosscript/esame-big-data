# Sistema di Monitoraggio del Traffico Urbano in Tempo Reale

## 1. Executive Summary

Progettazione di un sistema Big Data per il monitoraggio e l'ottimizzazione del traffico urbano in una città di medie dimensioni, con capacità di elaborazione in tempo reale e predizione delle congestioni.

## 2. Descrizione del Problema

### Obiettivi
- Monitorare il traffico in tempo reale su 200 incroci principali
- Prevedere congestioni con 15-30 minuti di anticipo
- Suggerire percorsi alternativi ai cittadini tramite app mobile
- Ottimizzare i tempi dei semafori basandosi sui pattern di traffico

### Task del progetto
- Gestione di stream di dati continui da multiple fonti
- Elaborazione in tempo reale con latenza < 1 minuto
- Scalabilità per gestire picchi di traffico
- Integrazione di dati eterogenei (IoT, GPS, mobile)

## 3. Fonti di Dati

| Fonte | Tipo | Volume | Frequenza |
|-------|------|---------|-----------|
| Sensori IoT | Contatori veicoli | 200 sensori | Ogni 30 sec |
| GPS Autobus | Posizione/Velocità | 100 mezzi | Ogni 30 sec |
| App Mobile | Posizione utenti | ~10.000 utenti | Ogni 60 sec |
| API Meteo | Condizioni meteo | 1 città | Ogni 15 min |

**Volume totale stimato**: ~5 GB/giorno di dati grezzi

## 4. Architettura del Sistema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Sensori IoT    │     │  GPS Autobus    │     │  App Mobile     │
│  (Incroci)      │     │                 │     │  Cittadini      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Apache Kafka          │
                    │  (Ingestion Layer)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Apache Spark          │
                    │  Streaming              │
                    │  (Processing Layer)     │
                    └──────┬─────────────┬────┘
                           │             │
              ┌────────────▼───┐   ┌─────▼──────────┐
              │   MongoDB      │   │  Redis         │
              │ (Storage Raw)  │   │ (Cache/RT)     │
              └────────────────┘   └────────────────┘
                           │             │
                    ┌──────▼─────────────▼────┐
                    │   REST API              │
                    │  (Node.js/Express)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Dashboard Web          │
                    │  (React + Leaflet)      │
                    └─────────────────────────┘
```

## 5. Componenti Dettagliate

### 5.1 Livello di Acquisizione Dati

#### Apache Kafka
- **Versione**: 3.5.0
- **Configurazione**: Cluster con 3 broker
- **Topic principali**:
  - `traffic-sensors`: Dati dai sensori di traffico
  - `gps-tracking`: Posizioni GPS dei mezzi pubblici
  - `mobile-app`: Dati crowdsourced dalle app
- **Retention**: 24 ore per topic real-time, 7 giorni per topic storici

### 5.2 Livello di Elaborazione

#### Apache Spark Streaming
- **Versione**: 3.4.0
- **Modalità**: Structured Streaming
- **Funzionalità implementate**:

```python
# Aggregazione dati per zona ogni 5 minuti
def aggregate_traffic_by_zone(df):
    return df.groupBy(
        window(df.timestamp, "5 minutes"),
        df.zone
    ).agg(
        avg("vehicle_speed").alias("avg_speed"),
        sum("vehicle_count").alias("total_vehicles"),
        stddev("vehicle_speed").alias("speed_variance")
    )

# Rilevamento anomalie
def detect_congestion(df):
    return df.filter(
        (df.avg_speed < 20) & 
        (df.speed_variance > 10)
    ).withColumn("alert_level", 
        when(df.avg_speed < 10, "severe")
        .otherwise("moderate")
    )
```

### 5.3 Livello di Storage

#### MongoDB
- **Versione**: 6.0
- **Configurazione**: Replica Set (3 nodi)
- **Collections**:

| Collection | Descrizione | TTL |
|------------|-------------|-----|
| raw_traffic_data | Dati grezzi sensori | 7 giorni |
| aggregated_hourly | Medie orarie | 30 giorni |
| traffic_patterns | Pattern storici | Permanente |
| incident_reports | Segnalazioni | 90 giorni |

#### Redis
- **Versione**: 7.0
- **Uso**: Cache per dati real-time
- **Strutture dati**:
  - Hash per stato corrente per zona
  - Sorted Set per ranking congestioni
  - Pub/Sub per alert real-time

### 5.4 Livello API

#### REST API (Node.js + Express)
```javascript
// Endpoints principali
GET  /api/v1/traffic/current/:zone
GET  /api/v1/traffic/historical/:zone?from=date&to=date
GET  /api/v1/traffic/prediction/:zone
POST /api/v1/routes/calculate
GET  /api/v1/alerts/active
POST /api/v1/alerts/subscribe
```

### 5.5 Livello Presentazione

#### Dashboard Web
- **Framework**: React 18 + TypeScript
- **Mappe**: Leaflet + OpenStreetMap
- **Grafici**: Chart.js
- **Real-time**: WebSocket per aggiornamenti

## 6. Modello Dati

### Formato Dati Sensore
```json
{
  "sensor_id": "INT_001",
  "location": {
    "lat": 45.464204,
    "lon": 9.189982
  },
  "timestamp": "2024-01-10T10:30:00Z",
  "measurements": {
    "vehicle_count": 45,
    "avg_speed_kmh": 35,
    "occupancy_percentage": 65
  }
}
```

### Formato Dati Aggregati
```json
{
  "zone_id": "CENTRO_01",
  "time_window": {
    "start": "2024-01-10T10:30:00Z",
    "end": "2024-01-10T10:35:00Z"
  },
  "statistics": {
    "avg_speed": 28.5,
    "total_vehicles": 450,
    "congestion_index": 0.72
  },
  "predictions": {
    "next_15min": 0.78,
    "next_30min": 0.65
  }
}
```

## 7. Algoritmi di Analisi

### 7.1 Indice di Congestione
```
CI = 1 - (velocità_attuale / velocità_libero_flusso) * 
     (1 + deviazione_standard / velocità_media)
```

### 7.2 Predizione Traffico
- **Modello**: ARIMA + Random Forest
- **Features**: 
  - Dati storici stessa ora/giorno
  - Condizioni meteo
  - Eventi speciali
  - Pattern stagionali

## 8. Deployment e Scalabilità

### Containerizzazione
```yaml
# docker-compose.yml
version: '3.8'
services:
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    ports:
      - "9092:9092"
  
  spark:
    image: apache/spark:3.4.0
    command: spark-submit /app/traffic-streaming.py
    depends_on:
      - kafka
  
  mongodb:
    image: mongo:6.0
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"
```

### Kubernetes per Produzione
- **Namespace**: traffic-monitoring
- **Deployments**: 
  - kafka-cluster (3 replicas)
  - spark-streaming (2 replicas)
  - api-server (3 replicas)
- **Autoscaling**: HPA basato su CPU/Memory

### Benefici Attesi
- Riduzione tempi percorrenza: -15%
- Diminuzione congestioni: -20%
- Risparmio carburante: ~€50.000/mese per la città
- Riduzione emissioni CO2: -10%


## 9. Riferimenti

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Spark Structured Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [MongoDB Best Practices](https://www.mongodb.com/docs/manual/administration/production-notes/)
- [Traffic Flow Theory](https://ops.fhwa.dot.gov/trafficanalysistools/tat_vol1/vol1_guidelines.pdf)

---

*Progetto creato come compito per il corso "Algoritmi e Strutture Dati per i Big Data"*