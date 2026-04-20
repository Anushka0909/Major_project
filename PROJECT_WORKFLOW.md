# GNN Trade Forecasting System - Complete Project Workflow

## 📋 Executive Summary

This is a **Graph Neural Network (GNN) system** that predicts bilateral trade flows by fusing three data streams: structured trade data (UN Comtrade), macroeconomic indicators (World Bank), and live global news sentiment (GDELT). The system runs in three modes: **Setup & Training**, **Continuous Updates**, and **Real-time API Serving**.

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                          │
│  UN Comtrade │ World Bank APIs │ GDELT DOC API (NEW!) │ CEPII Distances │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING LAYER (ETL)                        │
│  Normalize │ Feature Engineer │ Create Graph Snapshots │ Split Data  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING LAYER                          │
│         NeuralGravity (Additive Log-Space) + CausalGNN               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌─────────┐    ┌────────────┐    ┌──────────┐
    │ Model   │    │ Live Feed  │    │  API     │
    │ Cache   │    │ Injection  │    │ Backend  │
    │ (*.pt)  │    │ (DOC API)  │    │(FastAPI) │
    │         │    │            │    │          │
    └─────────┘    └────────────┘    └──────────┘
                           │                │
                           ▼                ▼
    ┌─────────────────────────────────────────────────┐
    │        PERSISTENCE & ACCELERATION LAYER          │
    │  In-Memory Graph Cache │ Redis │ PostgreSQL     │
    └─────────────────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────┐
    │         PRESENTATION LAYER                       │
    │        Next.js Dashboard + Causal Simulator      │
    └─────────────────────────────────────────────────┘
```

---

## 🔄 Complete Workflow Stages

### **STAGE 1: SETUP & INITIALIZATION** (Run Once)

**Entry Point:** `run.sh`

```bash
./run.sh
```

**Files Involved:**
| Step | File | Purpose | Input | Output |
|------|------|---------|-------|--------|
| 1a | `run.sh` | Main orchestrator | (none) | Checks/creates directories |
| 1b | `create required dirs` | Setup infrastructure | (none) | `logs/`, `models/`, `data/processed/` |

---

### **STAGE 2: DATA PREPROCESSING (ETL)** 

**Entry Point:** `scripts/preprocess_data.py` (called from `run.sh`)

```python
python scripts/preprocess_data.py
```

#### **2.1 Data Ingestion**
| File | Source | What It Does |
|------|--------|-------------|
| `src/data/preprocessing.py::DataPreprocessor.run()` | Reads configs | Orchestrates entire ETL pipeline |
| `configs/pipeline_config.yaml` | Configuration | Specifies data sources, paths, filters |

**Data Sources Loaded:**

| Data Type | File | Format | Key Columns |
|-----------|------|--------|------------|
| **Trade Data** | `data/raw/comtrade/TradeData.csv` | CSV | reporterCode, partnerCode, refYear, primaryValue, flowCode |
| **GDP** | `data/raw/world-bank/API_NY.GDP...csv` | CSV | Country, Year, GDP value |
| **Population** | `data/raw/world-bank/API_SP.POP...csv` | CSV | Country, Year, Population |
| **Inflation** | `data/raw/world-bank/API_FP.CPI...csv` | CSV | Country, Year, Inflation % |
| **Distance** | `data/raw/cepii/dist_cepii.csv` | CSV | iso_o, iso_d, dist, comlang_off, contig |
| **FTAs** | `data/raw/rta/AllRTAs.csv` | CSV | Member countries, RTA Name |

#### **2.2 Feature Engineering & Normalization**
| File | Function | Transforms |
|------|----------|-----------|
| `src/data/loaders_preprocessing.py` | Feature scaling | Logs GDP, normalizes sentiment [0,1] |
| `src/data/country_mapping.py` | Standardizes country names | ISO3 code → Unified names |
| `src/features/sentiment_features.py` | Sentiment scores (if available) | Raw news → Bilateral sentiment |

**Node Features Created (per country-year):**
```
- gdp_log (normalized log GDP)
- pop_log (normalized log population)  
- inflation (normalized inflation %)
- lagged_exports (previous year export volume)
```

**Edge Features Created (for trade pairs):**
```
- distance (log distance between countries)
- is_contiguous (binary: shared border)
- common_language (binary)
- fta_member (binary: same FTA)
- sentiment_score (news sentiment about trade pair)
- lagged_export (previous year bilateral exports)
+ other structural features
```

#### **2.3 Graph Construction & Temporal Splitting**
| File | Function | Output |
|------|----------|--------|
| `src/data/graph_builder.py` | Creates graph snapshots | One graph per year |
| `src/data/loaders.py::GraphDataLoader` | Train/val/test split | 70% train / 15% val / 15% test |

**Splits by Edge Type:**
- Train edges: Historical bilateral trades (70%)
- Validation edges: Mid-period trades (15%)
- Test edges: Most recent trades (15%)

**Output Files Created:**
```
data/processed/
├── nodes.csv              # All country-year records with features
├── edges.csv              # All bilateral trade records with features
├── node_mapping.json      # {country_iso3: node_id}
└── metadata.json          # Dataset stats (num_nodes, edges, years, etc.)
```

**Log Output:**
- `logs/preprocessing.log` - Detailed ETL logs

---

### **STAGE 3: MODEL TRAINING**

**Entry Point:** `scripts/train_model.py` (called from `run.sh`)

```python
python scripts/train_model.py
```

#### **3.1 Model Architecture Loading**
| File | Provides | Details |
|------|----------|---------|
| `configs/model_config.yaml` | Hyperparameters | Layers, heads, dropout, learning rate |
| `src/models/gnn.py::TradeGNN` | Baseline model | 3-layer GAT with 4 attention heads |
| `src/models/causal_gnn.py::CausalTradeGNN` | Causal model | GAT + causal inference for counterfactuals |

**Model Architecture (Baseline TradeGNN):**
```
Input: Node features (4D) + Edge features (10D)
  ↓
GATConv Layer 1: 4D → 128D × 4 heads = 512D
  ↓ (BatchNorm + Dropout)
GATConv Layer 2: 512D → 128D × 4 heads = 512D
  ↓ (BatchNorm + Dropout)
GATConv Layer 3: 512D → 128D × 1 head = 128D
  ↓
MLP Head: 128D → 1D (trade value prediction)
```

#### **3.2 Training Loop**
| File | Component | Function |
|------|-----------|----------|
| `src/models/train.py::main()` | Trainer | Epoch loop, loss computation, checkpoint saving |
| `src/data/loaders.py::GraphDataLoader` | Data pipeline | Batch samples from train/val/test sets |
| `src/utils/logger.py` | Monitoring | Loss tracking, metrics logging |

**Training Pipeline:**
```python
for epoch in range(num_epochs):
    # Forward pass on training data
    train_loss = model(train_graphs)
    
    # Backward pass + optimization
    optimizer.step()
    
    # Validation on held-out data
    val_loss = model(val_graphs)
    
    # Save checkpoint if val_loss improves
    if val_loss < best_val_loss:
        save_model(model, f"models/gnn_working.pt")
```

**Output Files:**
```
models/
├── gnn_working.pt                    # Latest best model (baseline)
├── gnn_YYYYMMDD_HHMMSS_metadata.json # Training metadata
└── causal_gnn_working.pt             # Causal model (if trained)

logs/
└── training.log                      # Training metrics
```

**Key Metrics Tracked:**
- Training loss (MSE between predicted and actual trade flows)
- Validation loss (early stopping criterion)
- Test accuracy (held-out set performance)

---

### **STAGE 4: SENTIMENT DATA PIPELINE** (Optional but Enhanced System)

This stage enriches edge features with news sentiment scores.

**Entry Point:** Called by `scripts/weekly_update.py`

#### **4.1 News Article Fetching (Real-Time vs Batch)**
| Method | File | Source | Logic |
|--------|------|--------|-------|
| **Real-Time** | `src/api/main.py::get_news` | GDELT DOC API | Fetches on-demand when user clicks country |
| **Batch** | `src/pipelines/gdelt_article_scheduler.py` | GDELT DOC API | Periodically fetches for all pairs |
| **Old Batch** | `src/pipelines/gdelt_fetcher.py` | BigQuery | Large scale historical data |

**Query Logic:**
```
SELECT article_url, tone, goldstein_score, num_mentions
FROM gdelt-bq.gdeltv2.events
WHERE EventDate >= start_date AND EventDate <= end_date
  AND (mentions CONTAINS "export" OR mentions CONTAINS "trade")
  AND actors contain relevant country pairs
```

**Output:** Raw articles with GDELT tone scores (-100 to +100)

#### **4.2 Article Content Extraction**
| File | Task | Input | Output |
|------|------|-------|--------|
| `src/pipelines/sentiment_analyzer.py::NewsContentExtractor` | Web scraping | URL from GDELT | Raw article text |

**Extraction Strategy:**
1. Try `newspaper3k` library (newspaper)
2. Fallback to BeautifulSoup if needed
3. Extract title, body text, summary

#### **4.3 Sentiment Analysis**
| File | Model | Scale | Details |
|------|-------|-------|---------|
| `src/pipelines/sentiment_analyzer.py::FinBERTSentimentAnalyzer` | FinBERT | [-1, 1] | Fine-tuned BERT for financial sentiment |

**Sentiment Scoring:**
```
Raw text → FinBERT Tokenizer → BERT Encoding → Classification Head
  ↓
Output: [NEGATIVE, NEUTRAL, POSITIVE] probabilities
  ↓
Score = P(POSITIVE) - P(NEGATIVE)  # Range: [-1, 1]
```

#### **4.4 Bilateral Aggregation**
| File | Function | Aggregation |
|------|----------|-------------|
| `src/pipelines/sentiment_analyzer.py::CompleteSentimentPipeline` | Aggregates | Mean sentiment per (country_pair, time_window) |

**Output Files:**
```
data/raw/sentiment/
├── bilateral_sentiment.csv    # (reporter, partner, date, sentiment_score)
└── gdelt_articles.parquet     # Raw articles with metadata
```

---

### **STAGE 5: WEEKLY AUTOMATED UPDATE CYCLE**

**Entry Points:**
- Manual: `python scripts/weekly_update.py`
- Automated: `python scripts/scheduler_service.py` (runs every Sunday at 2:00 AM)

#### **5.1 Scheduler Service**
| File | Duty | Schedule |
|------|------|----------|
| `scripts/scheduler_service.py` | Job orchestrator | Runs weekly_update.py on fixed schedule |
| `schedule` library | Timing | Cron-like scheduling in Python |

**Schedule Config (hardcoded):**
```python
schedule.every().sunday.at("02:00").do(run_weekly_update)
```

#### **5.2 Weekly Update Pipeline Steps**

**Step 1: Fetch Articles** 
```python
gdelt_fetcher = GDELTArticleFetcher()
df = fetcher.fetch_all_articles(max_per_pair=15)  # Latest news per trade pair
fetcher.save_articles(df)  # Save to data/raw/sentiment/
```

**Step 2: Analyze Sentiment**
```python
sentiment_pipeline = CompleteSentimentPipeline()
sentiment_df = sentiment_pipeline.run()  # Extract + Score articles
sentiment_pipeline.save_results()        # Save to data/raw/sentiment/
```

**Step 3: Preprocess Updated Data**
```python
preprocessor = DataPreprocessor()
nodes, edges, metadata = preprocessor.run()  # Reload, add new sentiment features
# Updates: data/processed/*
```

**Step 4: Optional Model Retraining**
```python
if settings.RETRAIN_ON_UPDATE:
    train_model()  # Re-train on fresh data
    # Saves new: models/gnn_YYYYMMDD_HHMMSS.pt
```

**Output:** 
```json
logs/weekly_updates/
└── weekly_update_YYYYMMDD_HHMMSS.json
{
  "timestamp": "20260401_115848",
  "steps_completed": ["fetch", "sentiment", "preprocess"],
  "articles_fetched": 324,
  "sentiment_scores_added": 250,
  "edges_updated": 1250,
  "errors": []
}
```

---

### **STAGE 6: REAL-TIME API SERVING**

**Entry Point:** `src/api/main.py` (started from `run.sh`)

```bash
# From run.sh or manual start
cd dashboard/src
pnpm dev  # Start frontend on localhost:3000

# In separate terminal
pytest       # Optional: health check
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Backend on localhost:8000
```

#### **6.1 Model Loading & Initialization**
| File | Role | Action |
|------|------|--------|
| `src/api/main.py::@app.on_event("startup")` | Initialization | Called once when API starts |
| `src/models/gnn.py::TradeGNN` | Model class | Instantiated from saved checkpoint |
| `models/gnn_working.pt` | Weights | Loaded into memory |
| `src/data/loaders.py::GraphDataLoader` | Data loader | Initialized for inference-time data |

**Startup Flow (Modern Lifespan):**
```
FastAPI app starts
  ↓
[Lifespan Context Manager]
  ├─ Verify Redis/Postgres
  ├─ Load Causal Model (causal_gnn_working.pt)
  ├─ Initialize GraphDataLoader
  ├─ Load Static Articles CSV
  └─ Pre-cache Temporal Graph Snapshots (~4-6 years)
  ↓
API ready (Ready message in logs)
```

#### **6.2 API Endpoints**

**1. Predictions Endpoint**
```
GET /predictions/export?reporter=IND&years=3&n_partners=10

Purpose: Get top export destinations for a country
Returns:
{
  "reporter": "India",
  "predictions": [
    {
      "partnerCode": "USA",
      "partner": "United States",
      "value": 45250000000,
      "change": +12.5,
      "confidence": 0.87
    },
    ...
  ],
  "generated_at": "2026-04-01T10:30:00Z"
}
```

**2. Alert Endpoint**
```
GET /alerts?severity=HIGH&days=7

Purpose: Get real-time supply chain disruption alerts
Returns:
{
  "alerts": [
    {
      "country_pair": "China-Taiwan",
      "alert_type": "SENTIMENT_SHOCK",
      "severity": "HIGH",
      "message": "Political tension detected (-35 sentiment change)",
      "confidence": 0.92,
      "timestamp": "2026-04-01T08:15:00Z"
    }
  ]
}
```

**3. Dashboard Data Endpoint**
```
GET /dashboard/graph-data?focus_country=IND&year=2025

Purpose: Get network visualization data
Returns:
{
  "nodes": [
    {"id": "IND", "label": "India", "gdp": ..., "exports": ...},
    ...
  ],
  "edges": [
    {"source": "IND", "target": "USA", "weight": 45.2, "sentiment": 0.65},
    ...
  ]
}
```

#### **6.3 Data Persistence**

**PostgreSQL (if configured):**
```
Table: predictions
├── id (PRIMARY KEY)
├── reporter_iso (Country code)
├── partner_iso (Trade partner code)
├── predicted_value (forecasted trade amount)
├── confidence (0-1 score)
├── execution_date (when prediction was made)
└── fiscal_year (target year)

Table: alerts
├── id
├── country_pair
├── alert_type
├── sentiment_score
├── created_at
└── acknowledged
```

**Redis Cache (if configured):**
```
Key Format: "predictions:{reporter}:{year}"
Value: JSON-serialized prediction results
TTL: 24 hours (auto-refresh daily)

Key Format: "sentiment:{iso_a}:{iso_b}:{date}"
Value: Bilateral sentiment score
TTL: 7 days
```

#### **6.4 Request Flow**
```
Client Request (e.g., /predictions/export?reporter=IND)
  ↓
Check Redis cache → if HIT, return cached result
  ↓ (if MISS)
Load test graph from data/processed/
  ↓
Forward graph through GNN model
  ↓
Extract predictions for requested country
  ↓
Compute confidence scores (attention weights)
  ↓
Save to PostgreSQL (if available)
  ↓
Cache in Redis (if available)
  ↓
Return JSON response to client
```

---

### **STAGE 7: DASHBOARD FRONTEND**

**Entry Point:** `dashboard/src/` (Next.js)

**Technology Stack:**
- Framework: Next.js 14 (React)
- State: React hooks
- Styling: Tailwind CSS + PostCSS
- Visualization: Custom D3.js / Three.js (network graphs)

**Directory Structure:**
```
dashboard/src/
├── app/
│   ├── layout.tsx         # Global layout (header, navbar)
│   ├── page.tsx           # Home page
│   ├── globals.css        # Global styles
│   └── alerts/
│       └── page.tsx       # Alerts dashboard
├── components/            # React components
│   ├── TradeNetworkVisualization.tsx  # Network graph
│   ├── PredictionTable.tsx            # Trade predictions table
│   ├── AlertBanner.tsx                # Alert notifications
│   └── ...
├── hooks/                 # Custom React hooks
│   ├── useTradeData.ts    # Fetch from API
│   ├── usePredictions.ts
│   └── ...
├── lib/                   # Utilities
│   ├── api-client.ts      # Calls /api/* endpoints
│   └── ...
└── public/                # Static assets
```

**Key Pages:**

| Page | Route | Data Source | Display |
|------|-------|-------------|---------|
| Dashboard | `/` | `/api/dashboard/graph-data` | Network graph + KPIs |
| Predictions | `/predictions` | `/api/predictions/export` | Trade forecast table |
| Alerts | `/alerts` | `/api/alerts` | Supply chain disruptions |
| Country Detail | `/country/[iso]` | `/api/country-details?iso=` | Deep-dive analytics |

**Data Flow in UI:**
```
User opens dashboard
  ↓
Next.js loads page.tsx
  ↓
useEffect hook calls fetch("/api/dashboard/graph-data")
  ↓
API backend responds with nodes/edges
  ↓
React renders TradeNetworkVisualization component
  ↓
D3.js/Three.js renders interactive network
```

---

## 📊 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
├─────────────────────────────────────────────────────────┤
│  UN Comtrade CSV  │  World Bank APIs  │  GDELT BigQuery  │
│  CEPII Distance   │  WTO RTAs CSV     │  News Sentiment  │
└────────┬──────────────────────────────────────────────────┘
         │
         │ ┌──────────────────────────────────────┐
         │ │ preprocess_data.py (Stage 2)         │
         │ │ ↓ DataPreprocessor.run()             │
         │ └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              PROCESSED DATA (NORMALIZED)                 │
├─────────────────────────────────────────────────────────┤
│  nodes.csv (countries)  │  edges.csv (bilateral trades) │
│  node_mapping.json      │  metadata.json                │
└────────┬──────────────────────────────────────────────────┘
         │
         │ ┌──────────────────────────────────────┐
         │ │ train_model.py (Stage 3)             │
         │ │ ↓ TradeGNN training loop             │
         │ └──────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   TRAINED MODELS                         │
├─────────────────────────────────────────────────────────┤
│  gnn_working.pt (baseline)                              │
│  causal_gnn_working.pt (causal inference)               │
│  Training metadata (hyperparams, metrics)               │
└────────┬──────────────────────────────────────────────────┘
         │
         │ ┌──────────────────────────────────────┐
         │ │ scheduler_service.py (Stage 5)       │
         │ │ ↓ Runs weekly_update.py              │
         │ │ ├─→ gdelt_fetcher.py (new articles)  │
         │ │ ├─→ sentiment_analyzer.py (scores)   │
         │ │ ├─→ preprocess_data.py (refresh)     │
         │ │ └─→ train_model.py (retrain)         │
         │ └──────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌──────────────────────┐       ┌──────────────────────┐
│   Backend API        │       │ PostgreSQL / Redis   │
│ (main.py - Stage 6)  │       │ (Persistent Store)   │
│ - Load models        │       │ - Predictions cache  │
│ - Make predictions   │───────│ - Alert history      │
│ - Compute alerts     │       │ - User preferences   │
└──────────┬───────────┘       └──────────────────────┘
           │
           │ REST Endpoints:
           │ - /predictions/export
           │ - /alerts
           │ - /dashboard/graph-data
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│         Next.js Dashboard Frontend (Stage 7)             │
│  - Network Visualization                                 │
│  - Prediction Tables                                     │
│  - Alert Notifications                                   │
│  - Country Deep-Dives                                    │
└──────────────────────────────────────────────────────────┘
```

---

## 🔑 Master File Reference

### **Configuration Files**
| File | Purpose | Key Settings |
|------|---------|--------------|
| `configs/model_config.yaml` | GNN hyperparameters | layers, heads, dropout, learning_rate |
| `configs/pipeline_config.yaml` | Data sources & paths | Data source configs, BigQuery settings |
| `configs/features.yaml` | Feature definitions | Node/edge feature engineering |
| `.env` | Environment variables | GCP_PROJECT_ID, DB_URL, REDIS_URL |

### **Data Layer**
| File | Stage | Function |
|------|-------|----------|
| `src/data/preprocessing.py` | 2 | Main ETL orchestrator |
| `src/data/graph_builder.py` | 2 | Creates temporal graph snapshots |
| `src/data/loaders.py` | 2,6 | GraphDataLoader for train/inference |
| `src/data/country_mapping.py` | 2 | Country name standardization |
| `src/features/sentiment_features.py` | 2,4 | Sentiment feature engineering |

### **Model Layer**
| File | Role | Details |
|------|------|---------|
| `src/models/gnn.py` | Model | TradeGNN class (baseline) |
| `src/models/causal_gnn.py` | Model | CausalTradeGNN (counterfactuals) |
| `src/models/train.py` | Training | Training loop, optimization |
| `src/models/inference.py` | Inference | Prediction logic |
| `src/models/simulation.py` | Simulation | Scenario analysis |

### **Pipeline Layer**
| File | Process | Trigger |
|------|---------|---------|
| `src/pipelines/gdelt_fetcher.py` | News fetching | Weekly schedule |
| `src/pipelines/sentiment_analyzer.py` | Sentiment scoring | Weekly schedule |
| `src/pipelines/gdelt_article_scheduler.py` | Job orchestration | Weekly schedule |
| `src/pipelines/feature_updater.py` | Feature refresh | Weekly schedule |

### **API Layer**
| File | Role | Details |
|------|------|---------|
| `src/api/main.py` | Backend | FastAPI app, endpoints |
| `src/api/redis_cache.py` | Caching | Redis operations |
| `src/api/postgres_db.py` | Database | PostgreSQL operations |

### **Utility Layer**
| File | Function |
|------|----------|
| `src/utils/config.py` | Configuration management |
| `src/utils/database.py` | Database connections |
| `src/utils/logger.py` | Structured logging |
| `src/utils/helpers.py` | Common utilities |

### **Operational Scripts**
| Script | Purpose | Entry Point | When to Run |
|--------|---------|-------------|------------|
| `run.sh` | Master orchestrator | Initial setup | Once at deployment |
| `scripts/preprocess_data.py` | ETL pipeline | Called by run.sh | Every new data dump |
| `scripts/train_model.py` | Model training | Called by run.sh | Every new data version |
| `scripts/weekly_update.py` | Automated refresh | Called by scheduler | Every Sunday 2 AM |
| `scripts/scheduler_service.py` | Background job | Manual start | Running in background |
| `scripts/quickstart.py` | Health check | Manual testing | Verify setup |

---

## ⚙️ Execution Sequences

### **Sequence 1: Initial Setup (One-Time)**
```
1. $ ./run.sh
   ├─ Creates dirs (logs, models, data/processed)
   ├─ Checks Python venv
   ├─ Calls: scripts/preprocess_data.py (Stage 2)
   │  └─ Outputs: nodes.csv, edges.csv, metadata.json
   ├─ Calls: scripts/train_model.py (Stage 3)
   │  └─ Outputs: models/gnn_working.pt
   └─ Starts API backend (Stage 6)
      └─ Serves on localhost:8000
      
2. $ cd dashboard/src && pnpm dev (Stage 7)
   └─ Frontend on localhost:3000
```

### **Sequence 2: Weekly Automated Update**
```
[TIMER: Every Sunday 2:00 AM]
  ↓
scheduler_service.py triggers:
  ├─ Step 1: gdelt_fetcher.py
  │  └─ Queries BigQuery → data/raw/sentiment/gdelt_articles.parquet
  ├─ Step 2: sentiment_analyzer.py
  │  ├─ Extracts article text
  │  ├─ Scores with FinBERT
  │  └─ Saves: bilateral_sentiment.csv
  ├─ Step 3: preprocess_data.py
  │  ├─ Reloads all data + new sentiment
  │  └─ Updates: nodes.csv, edges.csv
  ├─ Step 4: (Optional) train_model.py
  │  └─ Retrains on fresh data
  └─ Logs results: logs/weekly_updates/weekly_update_*.json
```

### **Sequence 3: Request-Response Cycle (Real-Time)**
```
User: Visits dashboard.com → GET /
  ↓ (Frontend loads page.tsx)
useEffect calls: GET /api/dashboard/graph-data?year=2025
  ↓ (Backend receives request)
src/api/main.py::@app.get("/dashboard/graph-data")
  ├─ Check Redis cache → HIT?
  │  └─ Return cached result
  └─ MISS: Compute fresh predictions
     ├─ Load graph from data/processed/
     ├─ Forward through GNN model
     ├─ Extract node/edge embeddings
     ├─ Cache in Redis (TTL: 24h)
     └─ Return JSON
  ↓ (Frontend receives JSON)
React re-renders TradeNetworkVisualization
  ↓
D3.js/Three.js renders interactive network
  ↓
User views dashboard with trade predictions
```

---

## 🎯 Key Features & Their Files

### **Feature: Trade Flow Prediction**
| Component | File | Input | Output |
|-----------|------|-------|--------|
| Historical data | `data/processed/edges.csv` | Bilateral trades | Edge features |
| Model inference | `src/models/gnn.py::forward()` | Graph + features | Predicted values |
| API response | `src/api/main.py::@app.get("/predictions/*")` | Country code | JSON predictions |

### **Feature: Real-Time Alerts**
| Component | File | Input | Output |
|-----------|------|-------|--------|
| Article fetching | `src/pipelines/gdelt_fetcher.py` | BigQuery | Raw articles |
| Sentiment scoring | `src/pipelines/sentiment_analyzer.py` | Article text | Sentiment scores |
| Alert logic | `src/pipelines/alert_engine.py` | Sentiment changes | Alert objects |
| Alert serving | `src/api/main.py::@app.get("/alerts")` | Filters | JSON alerts |

### **Feature: Dashboard Visualization**
| Component | File | Input | Output |
|-----------|------|-------|--------|
| Data aggregation | `src/api/main.py::dashboard_graph_data()` | Processed data | Node/edge JSON |
| Frontend layout | `dashboard/src/app/layout.tsx` | Static HTML | UI shell |
| Network viz | `dashboard/src/components/TradeNetworkVisualization.tsx` | Node/edge data | Interactive graph |

---

## 📁 Directory Purpose Summary

```
anushka/
│
├── configs/                    ← Configuration hub (YAML files)
│   ├── model_config.yaml      # GNN hyperparameters
│   ├── pipeline_config.yaml   # Data sources & paths
│   └── features.yaml          # Feature definitions
│
├── data/                       ← Data lake
│   ├── raw/                   # Source data (Comtrade, WB, GDELT)
│   │   ├── comtrade/
│   │   ├── world-bank/
│   │   ├── cepii/
│   │   ├── rta/
│   │   └── sentiment/
│   └── processed/             # ETL outputs (Stage 2)
│       ├── nodes.csv          # Country features
│       ├── edges.csv          # Trade relationships
│       ├── node_mapping.json
│       └── metadata.json
│
├── models/                     ← Model checkpoints (Stage 3)
│   ├── gnn_working.pt         # Active baseline model
│   ├── causal_gnn_working.pt  # Causal variant
│   └── gnn_*.pt               # Archived versions
│
├── scripts/                    ← Operational entry points
│   ├── preprocess_data.py     # Stage 2: ETL
│   ├── train_model.py         # Stage 3: Training
│   ├── weekly_update.py       # Stage 5: Update cycle
│   ├── scheduler_service.py   # Stage 5: Background job
│   └── ... (diagnostics, setup)
│
├── src/                        ← Core library (stages 2-7)
│   ├── data/                  # Data processing pipeline
│   │   ├── preprocessing.py   # Main ETL (Stage 2)
│   │   ├── graph_builder.py
│   │   ├── loaders.py         # Data loaders
│   │   └── country_mapping.py
│   ├── models/                # GNN implementation (Stage 3, 6)
│   │   ├── gnn.py             # Baseline model
│   │   ├── causal_gnn.py      # Causal model
│   │   ├── train.py           # Training loop
│   │   ├── inference.py       # Prediction
│   │   └── simulation.py      # Scenario analysis
│   ├── pipelines/             # Automation (Stage 5)
│   │   ├── gdelt_fetcher.py   # News fetching
│   │   ├── sentiment_analyzer.py # Scoring
│   │   ├── gdelt_article_scheduler.py
│   │   ├── alert_engine.py
│   │   └── feature_updater.py
│   ├── api/                   # Backend (Stage 6)
│   │   ├── main.py            # FastAPI app
│   │   ├── redis_cache.py     # Caching layer
│   │   └── postgres_db.py     # Database layer
│   └── utils/                 # Utilities (all stages)
│       ├── config.py
│       ├── logger.py
│       ├── database.py
│       └── helpers.py
│
├── dashboard/                  ← Frontend (Stage 7)
│   └── src/
│       ├── app/               # Next.js pages
│       ├── components/        # React components
│       ├── hooks/             # Custom hooks
│       ├── lib/               # Utilities
│       ├── public/            # Static assets
│       ├── package.json       # Node dependencies
│       └── ...
│
├── docker/                     ← Container setup
│   ├── Dockerfile.api         # Backend container
│   ├── Dockerfile.dashboard   # Frontend container
│   └── docker-compose.yml     # Orchestration
│
├── logs/                       ← Execution logs
│   └── weekly_updates/        # Update cycle results
│
├── run.sh                      ← Master orchestrator
├── requirements.txt            ← Python dependencies
├── database_schema.sql         ← DB schema
└── README.md                   ← Project docs
```

---

## 🚀 Quick Start Guide

### **To Set Up & Train (First Time):**
```bash
cd /Users/anushkapatil/Documents/anushka
./run.sh
```
This runs: Setup → Preprocess → Train → Start API

### **To Run Weekly Updates Manually:**
```bash
python scripts/weekly_update.py
```

### **To Run Background Scheduler:**
```bash
python scripts/scheduler_service.py
# Keep running (Ctrl+C to stop)
```

### **To Start Frontend:**
```bash
cd dashboard/src
pnpm dev
# Visit http://localhost:3000
```

### **To Test API:**
```bash
curl http://localhost:8000/predictions/export?reporter=IND&years=3
curl http://localhost:8000/alerts?severity=HIGH
```

---

## 📌 Critical Files at Each Stage

| Stage | Critical Files | If Missing | Recovery |
|-------|----------------|-----------|----------|
| **Setup** | `run.sh`, `requirements.txt` | Can't start | Re-clone repo |
| **Preprocessing** | Data sources in `data/raw/` | Pipeline fails | Download from source APIs |
| **Training** | `configs/model_config.yaml` | Can't train | Rebuild from template |
| **Models** | `models/gnn_working.pt` | API won't start | Run `train_model.py` |
| **Sentiment** | `GOOGLE_APPLICATION_CREDENTIALS` env var | Sentiment disabled | Set GCP credentials |
| **API** | `.env` with DB/Redis URLs | Limited functionality | Run without cache |
| **Frontend** | `dashboard/src/package.json` | Frontend won't build | `pnpm install` |

---

## 🔍 Debugging Checklist

**When preprocessing fails:**
- Check: Are all raw data files present in `data/raw/`?
- Check: `configs/pipeline_config.yaml` paths correct?
- Check: `src/data/country_mapping.py` has all countries?
- Look: `logs/preprocessing.log` for specifics

**When model training fails:**
- Check: `data/processed/nodes.csv` and `edges.csv` exist?
- Check: `configs/model_config.yaml` valid?
- Check: Sufficient GPU/RAM for batch size?
- Look: `logs/training.log`

**When API won't start:**
- Check: `models/gnn_working.pt` exists?
- Check: All Python dependencies installed?
- Check: Port 8000 not already in use?
- Look: Backend console output

**When dashboard won't load:**
- Check: API responding to requests?
- Check: `docker-compose.yml` services up?
- Check: Browser console for errors?
- Look: `dashboard/src` build logs

---

## 🏁 Conclusion

This system follows a **modular, batch + real-time hybrid architecture**:
- **Batch processing** (daily/weekly) updates the knowledge base
- **Real-time serving** (API) answers user queries instantly
- **Persistent storage** (PostgreSQL, Redis) caches predictions
- **Interactive UI** (Next.js) visualizes results

Every file has a specific role in one of the 7 stages. Understanding which stage you're in helps diagnose problems quickly.

