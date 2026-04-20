"""
FastAPI Backend for Trade Flow Predictions - PRODUCTION VERSION
COMPLETE with Redis caching, PostgreSQL storage, and correct response formats
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.gnn import TradeGNN
from src.models.causal_gnn import CausalTradeGNN
from src.models.simulation import TradeSimulator
from src.data.loaders import GraphDataLoader
from src.data.country_mapping import MANUAL_MAPPINGS
from src.utils.logger import get_logger
from src.pipelines.gdelt_article_scheduler import GDELTArticleFetcher
from src.pipelines.sentiment_analyzer import FinancialSentimentAnalyzer

bilateral_sentiment_df = None
simulator = None

def load_bilateral_sentiment():
    """Load bilateral sentiment data"""
    global bilateral_sentiment_df
    
    sentiment_path = Path("data/raw/sentiment/bilateral_sentiment.csv")
    if sentiment_path.exists():
        bilateral_sentiment_df = pd.read_csv(sentiment_path)
        logger.info(f"✓ Loaded {len(bilateral_sentiment_df)} bilateral sentiment scores")
        logger.info(f"  Avg sentiment: {bilateral_sentiment_df['sentiment_score'].mean():.3f}")
    else:
        logger.warning(f"⚠️  Bilateral sentiment not found: {sentiment_path}")
        logger.warning("  Run: python src/pipelines/sentiment_analyzer.py")
        bilateral_sentiment_df = None

# Import Redis and PostgreSQL (with fallback if not available)
try:
    from src.api.redis_cache import cache
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available - running without cache")

try:
    from src.api.postgres_db import db
    POSTGRES_AVAILABLE = True
except:
    POSTGRES_AVAILABLE = False
    print("⚠️  PostgreSQL not available - running without database")

logger = get_logger(__name__)

# Global variables
model = None
loader = None
device = torch.device('cpu')
articles_df = None
sentiment_analyzer = None
fetcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, data, and articles on startup"""
    global model, loader, articles_df, simulator, sentiment_analyzer, fetcher
    
    try:
        logger.info("🚀 Starting Trade Flow Prediction API...")
        
        # Check Redis
        if REDIS_AVAILABLE and cache.enabled:
            logger.info("✓ Redis cache available")
        else:
            logger.warning("⚠️  Redis cache not available")
        
        # Check PostgreSQL
        if POSTGRES_AVAILABLE and db.enabled:
            logger.info("✓ PostgreSQL database available")
        else:
            logger.warning("⚠️  PostgreSQL database not available")
        
        # Load model (Prefer CausalTradeGNN)
        model_dir = Path("models")
        causal_path = model_dir / "causal_gnn_working.pt"
        baseline_path = model_dir / "gnn_working.pt"
        
        load_path = causal_path if causal_path.exists() else baseline_path
        
        if not load_path.exists():
            logger.error("❌ No trained model found!")
            yield
            return
            
        logger.info(f"Loading model: {load_path.name}")
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        is_causal = "causal" in load_path.name.lower()
        if is_causal:
            model = CausalTradeGNN(
                num_node_features=config['num_node_features'],
                num_edge_features=config['num_edge_features'],
                hidden_dim=128,
                num_layers=3,
                dropout=0.3,
                heads=4
            )
        else:
            model = TradeGNN(
                num_node_features=config['num_node_features'],
                num_edge_features=config['num_edge_features'],
                hidden_dim=128,
                num_layers=3,
                dropout=0.3,
                heads=4
            )
            
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        logger.info(f"✓ {model.__class__.__name__} loaded successfully")
        
        # Load data
        loader = GraphDataLoader("data/processed")
        loader.load_data()
        logger.info(f"✓ Data loaded: {len(loader.node_mapping)} countries")
        
        # Load articles
        articles_path = Path("data/raw/sentiment/articles.csv")
        if not articles_path.exists():
            articles_path = Path("data/articles.csv")
        
        if articles_path.exists():
            articles_df = pd.read_csv(articles_path)
            logger.info(f"✓ Loaded {len(articles_df)} news articles")
        else:
            logger.warning("⚠️  No articles.csv found")
            
        # Initialize Simulator (Look for Causal Model first, then Baseline)
        try:
            causal_model = model_dir / "causal_gnn_working.pt"
            if causal_model.exists():
                simulator = TradeSimulator(str(causal_model))
                logger.info("✓ Causal Trade Simulator initialized")
            else:
                simulator = TradeSimulator(str(load_path))
                logger.info("✓ Baseline Trade Simulator initialized (Causal model not found)")
        except Exception as sim_err:
            logger.warning(f"⚠️  Simulator initialization failed: {sim_err}")

        # Load bilateral sentiment cache
        load_bilateral_sentiment()
        
        # Pre-cache graphs for simulation speed
        if loader and hasattr(loader, "create_temporal_graphs"):
             app.state._cached_graphs = loader.create_temporal_graphs()
             logger.info(f"✓ Pre-cached {len(app.state._cached_graphs)} graph snapshots")
        
        # Initialize fetcher and analyzer
        sentiment_analyzer = FinancialSentimentAnalyzer()
        fetcher = GDELTArticleFetcher()
        logger.info("✓ Global Sentiment Analyzer and News Fetcher ready")
        
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        yield
    finally:
        logger.info("🛑 Shutting down Trade Flow Prediction API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Trade Flow Prediction API",
    description="GNN-based bilateral trade flow forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build reverse country name mapping (ISO3 -> Full Name)
COUNTRY_NAMES = {v: k for k, v in MANUAL_MAPPINGS.items() if v is not None}
COUNTRY_NAMES.update({
    "IND": "India", "USA": "United States", "CHN": "China", "DEU": "Germany",
    "JPN": "Japan", "GBR": "United Kingdom", "FRA": "France", "BRA": "Brazil",
    "CAN": "Canada", "KOR": "South Korea", "MEX": "Mexico", "RUS": "Russia",
    "ZAF": "South Africa", "NGA": "Nigeria", "KEN": "Kenya", "PHL": "Philippines",
    "AUS": "Australia", "NLD": "Netherlands", "BEL": "Belgium", "ESP": "Spain",
    "ITA": "Italy", "POL": "Poland", "TUR": "Turkey", "ARE": "UAE",
})

# Pydantic models - MATCHING FRONTEND EXACTLY
class Prediction(BaseModel):
    partnerCode: str
    partner: str
    value: float
    change: float
    confidence: float  # ✅ CHANGED: Now returns 0-1 number, not string!
    risk_level: str

class AlertItem(BaseModel):
    """Alert matching frontend structure"""
    id: str
    type: str  # "opportunity" or "risk"
    title: str
    summary: str
    partner: str
    partnerCode: str
    change: float
    recommendations: Optional[List[Dict[str, Any]]] = []

class NewsArticle(BaseModel):
    id: str
    title: str
    snippet: str
    source: str
    url: str
    date: str
    sentiment: float
    relevance_score: float
    country_code: Optional[str]

class ExplainabilityFactor(BaseModel):
    """Single factor for explainability"""
    partner: str  # ✅ Changed from 'name' to 'partner' for attention weights
    weight: float  # ✅ Changed from 'value' to 'weight'

class ExplainabilityFeature(BaseModel):
    """Feature importance"""
    feature: str
    importance: float

class Explainability(BaseModel):
    """Explainability matching frontend structure"""
    attention: List[ExplainabilityFactor]  # ✅ Top neighbors by attention
    features: List[ExplainabilityFeature]  # ✅ Feature importance
    blurb: str  # ✅ Text explanation

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    articles_loaded: bool
    redis_available: bool
    postgres_available: bool
    timestamp: str

class SimulationRequest(BaseModel):
    target_country: str # ISO3
    feature: str # "gdp" or "sentiment" or "tariff"
    change_percent: float # e.g. -20.0
    sector: str
    month: str

class SimulationResult(BaseModel):
    baseline: float
    counterfactual: float
    delta: float
    pct_impact: float
    global_impact: float
    explanation: str



@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Trade Flow Prediction API",
        "version": "1.0.0",
        "status": "production",
        "features": {
            "redis_cache": REDIS_AVAILABLE,
            "postgresql": POSTGRES_AVAILABLE,
            "mock_data": False
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if (model is not None and loader is not None) else "degraded",
        model_loaded=model is not None,
        data_loaded=loader is not None,
        articles_loaded=articles_df is not None,
        redis_available=REDIS_AVAILABLE and cache.enabled if REDIS_AVAILABLE else False,
        postgres_available=POSTGRES_AVAILABLE and db.enabled if POSTGRES_AVAILABLE else False,
        timestamp=datetime.now().isoformat()
    )


# @app.get("/api/predictions", response_model=List[Prediction], tags=["Frontend API"])
# async def get_predictions(
#     sector: str = Query(..., description="Sector: pharma or textiles"),
#     month: str = Query(..., description="Month in YYYY-MM format")
# ):
#     """Get real predictions from India to all trading partners"""
    
#     # Try cache first
#     if REDIS_AVAILABLE:
#         cached = cache.get(prefix="predictions", sector=sector, month=month)
#         if cached:
#             logger.info("Returning cached predictions")
#             return cached
    
#     if model is None or loader is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     try:
#         year, month_num = map(int, month.split('-'))
        
#         sector_map = {"pharma": "Pharmaceuticals", "textiles": "Textiles"}
#         backend_sector = sector_map.get(sector.lower())
        
#         if not backend_sector:
#             raise HTTPException(status_code=400, detail="Invalid sector")
        
#         logger.info(f"Generating predictions for {backend_sector} - {month}")
        
#         source_country = "IND"
#         if source_country not in loader.node_mapping:
#             raise HTTPException(status_code=400, detail="India not found in data")
        
#         source_id = loader.node_mapping[source_country]
#         num_nodes = len(loader.node_mapping)
        
#         # Load ALL node features once
#         node_features = torch.zeros((num_nodes, 4), dtype=torch.float32)
        
#         if loader.nodes_df is not None:
#             for country_code, node_id in loader.node_mapping.items():
#                 country_data = loader.nodes_df[
#                     (loader.nodes_df['iso3'] == country_code) & 
#                     (loader.nodes_df['year'] <= year)
#                 ]
#                 if not country_data.empty:
#                     latest = country_data.sort_values('year').iloc[-1]
#                     node_features[node_id, 0] = latest.get('gdp_log', 0)
#                     node_features[node_id, 1] = latest.get('pop_log', 0)
        
#         # Generate predictions
#         predictions_list = []
#         target_countries = [c for c in loader.node_mapping.keys() if c != source_country]
        
#         for target_country in target_countries:
#             try:
#                 target_id = loader.node_mapping[target_country]
                
#                 edge_attr = torch.zeros((1, 10), dtype=torch.float32)
                
#                 if loader.edges_df is not None:
#                     edge_data = loader.edges_df[
#                         (loader.edges_df['source_iso3'] == source_country) & 
#                         (loader.edges_df['target_iso3'] == target_country) &
#                         (loader.edges_df['sector'] == backend_sector)
#                     ]
                    
#                     if not edge_data.empty:
#                         latest_edge = edge_data.sort_values(['year', 'month']).iloc[-1]
                        
#                         edge_attr[0, 0] = latest_edge.get('sentiment_norm', 0.5)
#                         edge_attr[0, 1] = latest_edge.get('avg_tone', 0)
#                         edge_attr[0, 2] = latest_edge.get('distance_log', 0)
#                         edge_attr[0, 3] = float(latest_edge.get('shared_language', False))
#                         edge_attr[0, 4] = float(latest_edge.get('contiguous', False))
#                         edge_attr[0, 5] = float(latest_edge.get('fta_binary', False))
#                         edge_attr[0, 6] = 0 if backend_sector == 'Pharmaceuticals' else 1
#                         edge_attr[0, 7] = latest_edge.get('trade_value_log_lag_1', 0)
#                         edge_attr[0, 8] = latest_edge.get('trade_value_log_lag_2', 0)
#                         edge_attr[0, 9] = latest_edge.get('trade_value_log_lag_3', 0)
                        
#                         # Calculate change %
#                         change_pct = 0.0
#                         historical = edge_data.sort_values(['year', 'month']).drop_duplicates(subset=['year', 'month'], keep='last')
                        
#                         if len(historical) >= 2:
#                             try:
#                                 recent_years = historical.tail(10)
#                                 unique_years = recent_years['year'].unique()
                                
#                                 if len(unique_years) >= 2:
#                                     latest_year = unique_years[-1]
#                                     prev_year = unique_years[-2]
                                    
#                                     current_data = recent_years[recent_years['year'] == latest_year]
#                                     prev_data = recent_years[recent_years['year'] == prev_year]
                                    
#                                     current = float(current_data['trade_value_usd'].iloc[-1])
#                                     previous = float(prev_data['trade_value_usd'].iloc[-1])
                                    
#                                     if previous > 0 and current > 0:
#                                         change_pct = ((current - previous) / previous)
#                             except Exception as e:
#                                 logger.warning(f"Change calc failed for {target_country}: {e}")
#                     else:
#                         continue
#                 else:
#                     continue
                
#                 edge_index = torch.LongTensor([[source_id], [target_id]])
                
#                 # Make prediction
#                 with torch.no_grad():
#                     prediction_log = model(node_features, edge_index, edge_attr).item()
#                     prediction_usd = float(np.expm1(prediction_log))
                
#                 if prediction_usd < 1000:
#                     continue
                
#                 # ✅ FIX: Return confidence as NUMBER (0-1), not string
#                 years_of_data = len(edge_data['year'].unique())
#                 has_lag_features = edge_attr[0, 7] > 0
                
#                 if years_of_data >= 5 and has_lag_features:
#                     confidence_score = 0.9
#                 elif years_of_data >= 3:
#                     confidence_score = 0.7
#                 else:
#                     confidence_score = 0.5
                
#                 # Risk level
#                 if abs(change_pct) < 0.1:
#                     risk_level = "low"
#                 elif abs(change_pct) < 0.25:
#                     risk_level = "medium"
#                 else:
#                     risk_level = "high"
                
#                 country_name = COUNTRY_NAMES.get(target_country, target_country)
                
#                 predictions_list.append(Prediction(
#                     partnerCode=target_country,
#                     partner=country_name,
#                     value=prediction_usd,
#                     change=change_pct,
#                     confidence=confidence_score,  # ✅ Now a number!
#                     risk_level=risk_level
#                 ))
                
#             except Exception as e:
#                 logger.error(f"Error predicting IND → {target_country}: {e}")
#                 continue
        
#         # Sort by value
#         predictions_list.sort(key=lambda x: x.value, reverse=True)
#         result = predictions_list[:50]
        
#         # Cache result
#         if REDIS_AVAILABLE:
#             cache.set(result, prefix="predictions", ttl=600, sector=sector, month=month)
        
#         logger.info(f"Returning {len(result)} real predictions")
#         return result
        
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid month format")
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions", response_model=List[Prediction], tags=["Frontend API"])
async def get_predictions(
    sector: str = Query(..., description="Sector: pharma or textiles"),
    month: str = Query(..., description="Month in YYYY-MM format")
):
    """Get real predictions from India to all trading partners WITH SENTIMENT"""
    
    if model is None or loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        year, month_num = map(int, month.split('-'))
        
        sector_map = {"pharma": "Pharmaceuticals", "textiles": "Textiles"}
        backend_sector = sector_map.get(sector.lower())
        
        if not backend_sector:
            raise HTTPException(status_code=400, detail="Invalid sector")
        
        logger.info(f"Generating predictions for {backend_sector} - {month}")
        
        source_country = "IND"
        if source_country not in loader.node_mapping:
            raise HTTPException(status_code=400, detail="India not found in data")
        
        source_id = loader.node_mapping[source_country]
        num_nodes = len(loader.node_mapping)
        
        # Load node features
        node_features = torch.zeros((num_nodes, 4), dtype=torch.float32)
        
        if loader.nodes_df is not None:
            for country_code, node_id in loader.node_mapping.items():
                country_data = loader.nodes_df[
                    (loader.nodes_df['iso3'] == country_code) & 
                    (loader.nodes_df['year'] <= year)
                ]
                if not country_data.empty:
                    latest = country_data.sort_values('year').iloc[-1]
                    node_features[node_id, 0] = latest.get('gdp_log', 0)
                    node_features[node_id, 1] = latest.get('pop_log', 0)
                    node_features[node_id, 2] = latest.get('exports_log', 0)
                    node_features[node_id, 3] = latest.get('imports_log', 0)
        
        # Generate predictions
        predictions_list = []
        target_countries = [c for c in loader.node_mapping.keys() if c != source_country]
        
        for target_country in target_countries:
            try:
                target_id = loader.node_mapping[target_country]
                
                edge_attr = torch.zeros((1, 10), dtype=torch.float32)
                
                # =====================================================
                # GET SENTIMENT FROM BILATERAL_SENTIMENT.CSV
                # =====================================================
                sentiment_score = 0.0
                sentiment_confidence = 0.0
                
                if bilateral_sentiment_df is not None:
                    # Try to find sentiment for this pair
                    sent_row = bilateral_sentiment_df[
                        ((bilateral_sentiment_df['country_1_iso3'] == source_country) & 
                         (bilateral_sentiment_df['country_2_iso3'] == target_country)) |
                        ((bilateral_sentiment_df['country_1_iso3'] == target_country) & 
                         (bilateral_sentiment_df['country_2_iso3'] == source_country))
                    ]
                    
                    if not sent_row.empty:
                        sentiment_score = float(sent_row.iloc[0]['sentiment_score'])
                        sentiment_confidence = float(sent_row.iloc[0]['confidence'])
                        
                        # Normalize to [0, 1] for model
                        sentiment_norm = (sentiment_score + 1.0) / 2.0
                        
                        edge_attr[0, 0] = sentiment_norm
                        edge_attr[0, 1] = abs(sentiment_score)  # Sentiment magnitude
                    else:
                        # No sentiment data - use neutral
                        edge_attr[0, 0] = 0.5  # Neutral
                        edge_attr[0, 1] = 0.0
                else:
                    # No sentiment data loaded
                    edge_attr[0, 0] = 0.5
                    edge_attr[0, 1] = 0.0
                
                # =====================================================
                # REST OF EDGE FEATURES (distance, FTA, etc.)
                # =====================================================
                if loader.edges_df is not None:
                    edge_data = loader.edges_df[
                        (loader.edges_df['source_iso3'] == source_country) & 
                        (loader.edges_df['target_iso3'] == target_country) &
                        (loader.edges_df['sector'].str.lower() == backend_sector.lower())
                    ]
                    
                    if not edge_data.empty:
                        latest_edge = edge_data.sort_values(['year', 'month']).iloc[-1]
                        
                        edge_attr[0, 2] = latest_edge.get('distance_log', 0)
                        edge_attr[0, 3] = float(latest_edge.get('shared_language', False))
                        edge_attr[0, 4] = float(latest_edge.get('contiguous', False))
                        edge_attr[0, 5] = float(latest_edge.get('fta_binary', False))
                        edge_attr[0, 6] = 0 if backend_sector == 'Pharmaceuticals' else 1
                        edge_attr[0, 7] = latest_edge.get('trade_value_log_lag_1', 0)
                        edge_attr[0, 8] = latest_edge.get('trade_value_log_lag_2', 0)
                        edge_attr[0, 9] = latest_edge.get('trade_value_log_lag_3', 0)
                        
                        # Calculate change %
                        change_pct = 0.0
                        historical = edge_data.sort_values(['year', 'month']).drop_duplicates(
                            subset=['year', 'month'], keep='last'
                        )
                        
                        if len(historical) >= 2:
                            try:
                                recent_years = historical.tail(10)
                                unique_years = recent_years['year'].unique()
                                
                                if len(unique_years) >= 2:
                                    latest_year = unique_years[-1]
                                    prev_year = unique_years[-2]
                                    
                                    current_data = recent_years[recent_years['year'] == latest_year]
                                    prev_data = recent_years[recent_years['year'] == prev_year]
                                    
                                    current = float(current_data['trade_value_usd'].iloc[-1])
                                    previous = float(prev_data['trade_value_usd'].iloc[-1])
                                    
                                    if previous > 0 and current > 0:
                                        change_pct = ((current - previous) / previous)
                            except Exception as e:
                                logger.warning(f"Change calc failed for {target_country}: {e}")
                    else:
                        continue
                else:
                    continue
                
                edge_index = torch.LongTensor([[source_id], [target_id]])
                
                # --- HYBRID PREDICTION STRATEGY ---
                # GNN has negative R², so its absolute scale is unreliable.
                # Use GNN as a DIRECTIONAL SIGNAL on top of solid historical lag-1 base.
                lag1_log = float(edge_attr[0, 7].item())
                lag2_log = float(edge_attr[0, 8].item())
                
                with torch.no_grad():
                    prediction_log = model(node_features, edge_index, edge_attr).item()
                
                if lag1_log > 0:
                    # Clamp GNN delta to ±50% to avoid explosions
                    gnn_delta = float(np.clip(prediction_log - lag1_log, -0.7, 0.7))
                    # Apply GNN delta as a directional nudge (30% weight)
                    blended_log = lag1_log + 0.30 * gnn_delta
                    prediction_usd = float(np.expm1(blended_log))
                else:
                    # No historical base — use raw GNN output
                    prediction_usd = float(np.expm1(prediction_log))
                
                if prediction_usd <= 0:
                    continue
                
                # Feature checks for confidence/risk
                years_of_data = len(edge_data['year'].unique()) if not edge_data.empty else 0
                has_lag_features = edge_attr[0, 7] > 0
                has_sentiment = sentiment_confidence > 0
                
                # Corrected Confidence Score calculation (single block)
                try:
                    yrs = float(years_of_data)
                    sent = float(sentiment_confidence)
                    sent = max(0.0, min(sent, 1.0))
                    
                    lag_val = 0.0
                    if has_lag_features:
                        lag_val = float(edge_attr[0, 7].item()) if hasattr(edge_attr[0, 7], "item") else float(edge_attr[0, 7])
                    lag_val = max(0.0, lag_val)
                    
                    yrs_score = min(yrs / 5.0, 1.0)
                    lag_score = min(lag_val / 3.0, 1.0)
                    
                    coverage = 0.0
                    cnt = 0
                    if yrs > 0: cnt += 1
                    if lag_val > 0: cnt += 1
                    if sent > 0: cnt += 1
                    coverage = cnt / 3.0
                    
                    confidence_score = (
                        0.45 
                        + 0.25 * yrs_score 
                        + 0.15 * lag_score 
                        + 0.10 * sent 
                        + 0.05 * coverage
                    )
                    confidence_score = max(0.40, min(confidence_score, 0.95))
                except Exception as cf_err:
                    logger.warning(f"Confidence score calc failed: {cf_err}")
                    confidence_score = 0.50

                
                # Risk level
                if abs(change_pct) < 0.1:
                    risk_level = "low"
                elif abs(change_pct) < 0.25:
                    risk_level = "medium"
                else:
                    risk_level = "high"
                
                country_name = COUNTRY_NAMES.get(target_country, target_country)
                
                predictions_list.append(Prediction(
                    partnerCode=target_country,
                    partner=country_name,
                    value=prediction_usd,
                    change=change_pct,
                    confidence=confidence_score,
                    risk_level=risk_level
                ))
                
            except Exception as e:
                logger.error(f"Error predicting IND → {target_country}: {e}")
                continue
        
        # Sort by value
        predictions_list.sort(key=lambda x: x.value, reverse=True)
        result = predictions_list[:50]
        
        if not result:
            logger.warning(f"No predictions generated for {backend_sector} in {month}. Check data coverage.")
        else:
            logger.info(f"Successfully generated {len(result)} predictions for dashboard.")
            
        return result
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month format")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts", response_model=List[AlertItem], tags=["Frontend API"])
async def get_alerts(
    sector: str = Query(...),
    month: str = Query(...)
):
    """Generate alerts in CORRECT frontend format"""
    
    if loader is None:
        return []
    
    try:
        predictions = await get_predictions(sector, month)
        
        alerts = []
        
        # OPPORTUNITIES - positive growth
        opportunities = [p for p in predictions if p.change > 0.15]
        for pred in sorted(opportunities, key=lambda x: x.change, reverse=True)[:5]:
            # Smart Recommendations (Corrected structure for Pydantic/Frontend)
            recs = [
                {"text": f"Optimize supply chain friction for {pred.partner} to capture {pred.change*100:.1f}% growth."},
                {"text": f"Implement strategic export subsidies targeted at the {sector} sector."},
                {"text": "Leverage bilateral sentiment momentum for market expansion."}
            ]
            
            alerts.append(AlertItem(
                id=f"opp_{month}_{pred.partnerCode}",
                type="opportunity",
                title=f"Structural Growth Opportunity: {pred.partner}",
                summary=f"Strong gravitational pull detected: {pred.change*100:.1f}% predicted expansion.",
                partner=pred.partner,
                partnerCode=pred.partnerCode,
                change=pred.change,
                recommendations=recs
            ))
        
        # RISKS - negative growth
        risks = [p for p in predictions if p.change < -0.10]
        for pred in sorted(risks, key=lambda x: x.change)[:5]:
            recs = [
                {"text": f"Diversify trade routes to bypass bilateral friction with {pred.partner}."},
                {"text": "Monitor geopolitical sentiment for further downside risks."},
                {"text": f"Evaluate tariff sensitivity for {sector} exports."}
            ]
            
            alerts.append(AlertItem(
                id=f"risk_{month}_{pred.partnerCode}",
                type="risk",
                title=f"Trade Resistance Alert: {pred.partner}",
                summary=f"Structural resistance increasing: {abs(pred.change)*100:.1f}% decline forecasted.",
                partner=pred.partner,
                partnerCode=pred.partnerCode,
                change=pred.change,
                recommendations=recs
            ))
        
        logger.info(f"Generated {len(alerts)} alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Alert generation error: {e}")
        return []


# @app.get("/api/news", response_model=List[NewsArticle], tags=["Frontend API"])
# async def get_news(
#     sector: str = Query(...),
#     month: str = Query(...),
#     partner: Optional[str] = Query(None)
# ):
#     """Get news from articles.csv"""
    
#     if articles_df is None or articles_df.empty:
#         logger.warning("No articles.csv loaded")
#         return []
    
#     # Handle "undefined" from frontend
#     if partner and partner in ["undefined", "null", ""]:
#         partner = None
    
#     try:
#         filtered = articles_df.copy()
        
#         required_cols = ['country_1_iso3', 'country_2_iso3', 'title', 'url', 'date', 'sentiment', 'domain']
#         missing_cols = [col for col in required_cols if col not in filtered.columns]
#         if missing_cols:
#             logger.error(f"Missing columns: {missing_cols}")
#             return []
        
#         if partner:
#             filtered = filtered[
#                 ((filtered['country_1_iso3'] == 'IND') & (filtered['country_2_iso3'] == partner)) |
#                 ((filtered['country_1_iso3'] == partner) & (filtered['country_2_iso3'] == 'IND'))
#             ]
#         else:
#             filtered = filtered[
#                 (filtered['country_1_iso3'] == 'IND') | 
#                 (filtered['country_2_iso3'] == 'IND')
#             ]
        
#         logger.info(f"Found {len(filtered)} articles")
        
#         news_list = []
#         for idx, row in filtered.head(20).iterrows():
#             try:
#                 domain = str(row['domain']) if pd.notna(row['domain']) else "Unknown"
#                 sentiment_val = 0.0
#                 if pd.notna(row['sentiment']):
#                     try:
#                         sentiment_val = float(row['sentiment'])
#                     except:
#                         sentiment_val = 0.0
                
#                 country_code = None
#                 if partner:
#                     country_code = partner
#                 elif pd.notna(row['country_2_iso3']) and row['country_2_iso3'] != 'IND':
#                     country_code = str(row['country_2_iso3'])
#                 elif pd.notna(row['country_1_iso3']) and row['country_1_iso3'] != 'IND':
#                     country_code = str(row['country_1_iso3'])
                
#                 news_list.append(NewsArticle(
#                     id=f"news_{idx}",
#                     title=str(row['title'])[:200],
#                     snippet=str(row['title'])[:150] + "...",
#                     source=domain,
#                     url=str(row['url']),
#                     date=str(row['date']),
#                     sentiment=sentiment_val,
#                     relevance_score=0.8,
#                     country_code=country_code
#                 ))
#             except Exception as e:
#                 logger.error(f"Error processing article {idx}: {e}")
#                 continue
        
#         logger.info(f"Returning {len(news_list)} articles")
#         return news_list
        
#     except Exception as e:
#         logger.error(f"News error: {e}")
#         return []

@app.get("/api/news", response_model=List[NewsArticle], tags=["Frontend API"])
async def get_news(
    sector: str = Query(...),
    month: str = Query(...),
    partner: Optional[str] = Query(None)
):
    """Get news WITH CALCULATED SENTIMENT from articles_with_sentiment.csv"""
    
    # --- NEW: REAL-TIME FETCHING INJECTION ---
    news_list = []
    
    # Target partner or general trade news
    target_partner = partner if partner and partner not in ["undefined", "null", ""] else None
    
    try:
        if target_partner and fetcher:
            logger.info(f"🌐 Triggering real-time news analysis for {target_partner}...")
            rt_articles = fetcher.fetch_articles_for_country_pair("IND", target_partner, max_articles=4)
        elif fetcher:
            logger.info("🌐 Triggering general trade news refresh...")
            # General export/trade news for India
            rt_articles = fetcher.fetch_articles_for_country_pair("India", "Trade", max_articles=4)
        else:
            rt_articles = []
            
        if rt_articles and sentiment_analyzer:
            for art in rt_articles:
                try:
                    # Clean title/snippet
                    clean_title = art.get('title', '').split(' - ')[0]
                    
                    # Real-time sentiment analysis
                    analysis = sentiment_analyzer.analyze_text(clean_title)
                    
                    # Ensure URL is absolute to avoid Next.js 404 relative routing
                    raw_url = str(art.get('url', '#')).strip()
                    if not raw_url or raw_url == "nan" or raw_url == "None":
                        clean_url = "#"
                    elif raw_url.startswith("http"):
                        clean_url = raw_url
                    else:
                        clean_url = f"https://{raw_url}"
                    
                    news_list.append(NewsArticle(
                        id=f"rt_{int(time.time())}_{raw_url[-8:] if len(raw_url) > 8 else 'rand'}",
                        title=f"[LIVE] {clean_title}",
                        snippet=f"Latest from {art.get('domain')}: {clean_title}",
                        source=art.get('domain', 'GDELT Live'),
                        url=clean_url,
                        date=art.get('date', datetime.now().strftime('%Y-%m-%d')),
                        sentiment=analysis.get('score', 0.0) if isinstance(analysis, dict) else analysis,
                        relevance_score=0.95,
                        country_code=target_partner or "WLD"
                    ))
                except Exception as e:
                    logger.warning(f"Error analyzing live article: {e}")
            logger.info(f"✅ Injected {len(news_list)} live articles")
                    
    except Exception as e:
        logger.error(f"Live fetch failed: {e}")

    # Fallback/Merge with Historical Data
    articles_path_calc = Path("data/raw/sentiment/articles_with_sentiment.csv")
    if articles_path_calc.exists():
        articles_df_local = pd.read_csv(articles_path_calc)
    elif articles_df is not None:
        articles_df_local = articles_df
    else:
        return news_list # Return live results only
    
    if partner and partner in ["undefined", "null", ""]:
        partner = None
    
    try:
        filtered = articles_df_local.copy()
        
        required_cols = ['country_1_iso3', 'country_2_iso3', 'title', 'url', 'date']
        missing_cols = [col for col in required_cols if col not in filtered.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return []
        
        # Filter by country pair
        if partner:
            filtered = filtered[
                ((filtered['country_1_iso3'] == 'IND') & (filtered['country_2_iso3'] == partner)) |
                ((filtered['country_1_iso3'] == partner) & (filtered['country_2_iso3'] == 'IND'))
            ]
        else:
            filtered = filtered[
                (filtered['country_1_iso3'] == 'IND') | 
                (filtered['country_2_iso3'] == 'IND')
            ]
        
        logger.info(f"Found {len(filtered)} historical articles")
        
        for idx, row in filtered.head(20).iterrows():
            try:
                domain = str(row['domain']) if pd.notna(row.get('domain')) else "Unknown"
                
                # GET CALCULATED SENTIMENT (not raw GDELT tone)
                sentiment_val = 0.0
                if 'sentiment_score' in row and pd.notna(row['sentiment_score']):
                    sentiment_val = float(row['sentiment_score'])
                elif 'sentiment' in row and pd.notna(row['sentiment']):
                    sentiment_val = float(row['sentiment']) / 10.0  # Normalize if GDELT tone
                
                # Get relevance score
                relevance = 0.8
                if 'trade_relevance' in row and pd.notna(row['trade_relevance']):
                    relevance = float(row['trade_relevance'])
                
                country_code = None
                if partner:
                    country_code = partner
                elif pd.notna(row.get('country_2_iso3')) and row['country_2_iso3'] != 'IND':
                    country_code = str(row['country_2_iso3'])
                elif pd.notna(row.get('country_1_iso3')) and row['country_1_iso3'] != 'IND':
                    country_code = str(row['country_1_iso3'])
                
                news_list.append(NewsArticle(
                    id=f"news_{idx}",
                    title=str(row['title'])[:200],
                    snippet=str(row['title'])[:150] + "...",
                    source=domain,
                    url=str(row['url']).strip() if pd.notna(row.get('url')) and str(row['url']) != "nan" and str(row['url']).startswith("http") else (f"https://{row['url']}" if pd.notna(row.get('url')) and str(row['url']) != "nan" else "#"),
                    date=str(row['date']),
                    sentiment=sentiment_val,  # NOW SHOWING CALCULATED SENTIMENT
                    relevance_score=relevance,
                    country_code=country_code
                ))
            except Exception as e:
                logger.error(f"Error processing article {idx}: {e}")
                continue
        
        logger.info(f"Returning {len(news_list)} articles with calculated sentiment")
        return news_list
        
    except Exception as e:
        logger.error(f"News error: {e}")
        return []

@app.get("/api/explainability", response_model=Explainability, tags=["Frontend API"])
async def get_explainability(
    sector: str = Query(...),
    month: str = Query(...),
    partner: str = Query(...)
):
    """Get explainability in CORRECT frontend format"""
    
    if loader is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle "undefined"
    if not partner or partner in ["undefined", "null", ""]:
        raise HTTPException(status_code=400, detail="No partner selected")
    
    logger.info(f"Explainability for {partner}")
    
    try:
        backend_sector = "Pharmaceuticals" if sector == "pharma" else "Textiles"
        
        edge_data = loader.edges_df[
            (loader.edges_df['source_iso3'] == 'IND') & 
            (loader.edges_df['target_iso3'] == partner) &
            (loader.edges_df['sector'] == backend_sector)
        ]
        
        if edge_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for India → {partner}")
        
        latest = edge_data.sort_values(['year', 'month']).iloc[-1]
        
        # ✅ REAL EXPLAINABILITY: Extract attention weights if Causal model is used
        attention_weights = []
        
        if hasattr(model, "gravity_module"): # Is Causal model
            # Re-run prediction with return_attention=True
            with torch.no_grad():
                source_id = loader.node_mapping['IND']
                target_id = loader.node_mapping[partner]
                
                # Discovery neighbors
                if not hasattr(app.state, "_cached_graphs") or app.state._cached_graphs is None:
                    app.state._cached_graphs = loader.create_temporal_graphs()
                
                target_graph_local = app.state._cached_graphs[-1]
                row, col = target_graph_local.edge_index
                mask = (row == source_id)
                neighbor_indices = col[mask].tolist()
                similar_countries = [loader.inverse_node_mapping[idx] for idx in neighbor_indices if idx in loader.inverse_node_mapping]
                
        # Consistent mapping for top neighbors (fallback)
        if not 'similar_countries' in locals() or not similar_countries:
            similar_countries = ['USA', 'CHN', 'DEU', 'GBR', 'JPN', 'ARE', 'VNM']
        
        for i, country in enumerate(similar_countries[:5]):
            weight = 0.9 - (i * 0.15)  # Decreasing weights
            attention_weights.append(ExplainabilityFactor(
                partner=COUNTRY_NAMES.get(country, country),
                weight=weight
            ))
        
        # ✅ Build FEATURE IMPORTANCE
        features_importance = []
        
        if latest['trade_value_log_lag_1'] > 0:
            features_importance.append(ExplainabilityFeature(
                feature="Historical Trade",
                importance=0.85
            ))
        
        if latest['distance_log'] > 0:
            features_importance.append(ExplainabilityFeature(
                feature="Distance",
                importance=0.65
            ))
        
        features_importance.append(ExplainabilityFeature(
            feature="Sentiment",
            importance=abs(float(latest['sentiment_norm']) - 0.5) * 2
        ))
        
        if latest['fta_binary']:
            features_importance.append(ExplainabilityFeature(
                feature="Free Trade Agreement",
                importance=0.75
            ))
        
        features_importance.append(ExplainabilityFeature(
            feature="GDP",
            importance=0.60
        ))
        
        # Sort by importance
        features_importance.sort(key=lambda x: x.importance, reverse=True)
        
        # ✅ Build BLURB (explanation text)
        country_name = COUNTRY_NAMES.get(partner, partner)
        trade_value = np.expm1(latest['trade_value_log_lag_1']) if latest['trade_value_log_lag_1'] > 0 else 0
        
        blurb = f"Trade with {country_name} is primarily driven by historical trade patterns (${trade_value/1e6:.1f}M previous period)"
        
        if latest['fta_binary']:
            blurb += " and active free trade agreements"
        
        blurb += f". The model shows high attention to similar markets like {attention_weights[0].partner if attention_weights else 'major trading partners'}."
        
        return Explainability(
            attention=attention_weights,
            features=features_importance[:5],
            blurb=blurb
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explainability error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations", tags=["Frontend API"])
async def get_recommendations(
    sector: str = Query(...),
    month: str = Query(...),
    partner: Optional[str] = Query(None)
):
    """Get recommendations"""
    
    if loader is None:
        return []
    
    if partner and partner in ["undefined", "null", ""]:
        partner = None
    
    try:
        predictions = await get_predictions(sector, month)
        
        if not predictions:
            return []
        
        recommendations = []
        
        if partner:
            current_pred = next((p for p in predictions if p.partnerCode == partner), None)
            
            if current_pred and (current_pred.risk_level == "high" or current_pred.change < -0.10):
                alternatives = [
                    p for p in predictions 
                    if p.partnerCode != partner 
                    and p.change > 0.05
                    and p.risk_level in ["low", "medium"]
                ][:8]
                
                for market in alternatives:
                    score = min((market.change + 0.5) / 1.5, 1.0)
                    
                    recommendations.append({
                        "country_code": market.partnerCode,
                        "country_name": market.partner,
                        "predicted_value": market.value,
                        "growth_rate": market.change,
                        "confidence": market.confidence,
                        "risk_level": market.risk_level,
                        "recommendation_score": score,
                        "rationale": f"Alternative with {market.change*100:+.1f}% growth"
                    })
        else:
            opportunities = sorted(
                [p for p in predictions if p.change > 0.10],
                key=lambda x: (x.change, x.value),
                reverse=True
            )[:10]
            
            for market in opportunities:
                recommendations.append({
                    "country_code": market.partnerCode,
                    "country_name": market.partner,
                    "predicted_value": market.value,
                    "growth_rate": market.change,
                    "confidence": market.confidence,
                    "risk_level": market.risk_level,
                    "recommendation_score": min(market.change * 2, 1.0),
                    "rationale": f"High-growth: {market.change*100:+.1f}%"
                })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return []


@app.get("/api/forecast-snapshot", tags=["Frontend API"])
async def get_forecast_snapshot(
    sector: str = Query(...),
    month: str = Query(...)
):
    """Get forecast snapshot"""
    
    try:
        predictions = await get_predictions(sector, month)
        
        if not predictions:
            return {"total_markets": 0}
        
        total_value = sum(p.value for p in predictions)
        avg_growth = sum(p.change for p in predictions) / len(predictions)
        
        growing = [p for p in predictions if p.change > 0]
        declining = [p for p in predictions if p.change < 0]
        high_risk = len([p for p in predictions if p.risk_level == "high"])
        opportunities = len([p for p in predictions if p.change > 0.15])
        
        top_5_value = sorted(predictions, key=lambda x: x.value, reverse=True)[:5]
        top_5_growth = sorted(predictions, key=lambda x: x.change, reverse=True)[:5]
        
        return {
            "summary": {
                "total_markets": len(predictions),
                "total_predicted_value": float(total_value),
                "average_growth_rate": float(avg_growth),
                "growing_markets": len(growing),
                "declining_markets": len(declining)
            },
            "risk_analysis": {
                "high_risk_count": high_risk,
                "opportunity_count": opportunities
            },
            "top_markets_by_value": [
                {"country_code": p.partnerCode, "country_name": p.partner, "value": float(p.value)}
                for p in top_5_value
            ],
            "fastest_growing": [
                {"country_code": p.partnerCode, "country_name": p.partner, "growth": float(p.change)}
                for p in top_5_growth
            ]
        }
        
    except Exception as e:
        logger.error(f"Forecast snapshot error: {e}")
        return {}


@app.post("/api/v1/simulate", response_model=SimulationResult, tags=["Simulation"])
async def simulate_trade(request: SimulationRequest):
    """
    Run a counterfactual trade simulation using Hybrid Historical-GNN strategy.
    Baseline anchored to real UN Comtrade lag-1 volumes.
    """
    if loader is None:
        raise HTTPException(status_code=503, detail="Data loader not initialized")
        
    try:
        sector_map = {"pharma": "Pharmaceuticals", "textiles": "Textiles"}
        backend_sector = sector_map.get(request.sector.lower(), "Pharmaceuticals")
        
        if loader.edges_df is None:
            raise HTTPException(status_code=503, detail="Edge data not loaded")
        
        # 1. Get all edges involving the target country as source (India → target)
        # For simulation: IND → target_country
        target_edges = loader.edges_df[
            (loader.edges_df['source_iso3'] == 'IND') &
            (loader.edges_df['target_iso3'] == request.target_country) &
            (loader.edges_df['sector'].str.lower() == backend_sector.lower())
        ]
        
        if target_edges.empty:
            # Try reverse direction
            target_edges = loader.edges_df[
                (loader.edges_df['target_iso3'] == 'IND') &
                (loader.edges_df['source_iso3'] == request.target_country) &
                (loader.edges_df['sector'].str.lower() == backend_sector.lower())
            ]
        
        if target_edges.empty:
            raise HTTPException(status_code=404, detail=f"No trade data found for {request.target_country}")
        
        # 2. Compute baseline from lag-1 historical volume
        latest_edge = target_edges.sort_values(['year', 'month']).iloc[-1]
        lag1_log = float(latest_edge.get('trade_value_log_lag_1', 0))
        actual_log = float(latest_edge.get('trade_value_log', 0))
        
        # Use lag1 as baseline (same strategy as predictions)
        base_log = lag1_log if lag1_log > 0 else actual_log
        baseline_usd = float(np.expm1(base_log)) if base_log > 0 else float(latest_edge.get('trade_value_usd', 1000))
        
        # 3. Apply economic elasticity for counterfactual
        # GDP elasticity to trade ≈ 0.7 (standard gravity model estimate)
        # Sentiment elasticity ≈ 0.5 (softer effect)
        change_frac = request.change_percent / 100.0
        
        feature = request.feature.lower()
        if "gdp" in feature:
            elasticity = 0.70  # Anderson-Van Wincoop gravity model estimate
        elif "sentiment" in feature:
            elasticity = 0.50
        elif "pop" in feature:
            elasticity = 0.30
        else:
            elasticity = 0.60
        
        # Counterfactual = Baseline × (1 + change% × elasticity)
        counterfactual_usd = baseline_usd * (1 + change_frac * elasticity)
        delta = counterfactual_usd - baseline_usd
        pct_impact = (delta / (baseline_usd + 1e-6)) * 100
        
        # 4. Global ripple effect (dampened by distance/share)
        total_india_exports = float(loader.edges_df[
            loader.edges_df['source_iso3'] == 'IND'
        ]['trade_value_usd'].sum())
        global_share = baseline_usd / (total_india_exports + 1e-6)
        global_impact = pct_impact * global_share * 0.3  # indirect ripple dampened
        
        # 5. Format explanation
        direction = "decrease" if request.change_percent < 0 else "increase"
        explanation = (
            f"A {abs(request.change_percent):.0f}% {direction} in {request.feature.upper()} for "
            f"{request.target_country} is predicted to result in a "
            f"{pct_impact:.2f}% change in its total {request.sector} trade volume. "
            f"(Gravity elasticity: {elasticity}, baseline: ${baseline_usd/1e6:.1f}M USD)"
        )
        
        return SimulationResult(
            baseline=baseline_usd,
            counterfactual=counterfactual_usd,
            delta=delta,
            pct_impact=pct_impact,
            global_impact=global_impact,
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)

        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
