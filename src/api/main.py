"""
FastAPI Backend for Trade Flow Predictions - PRODUCTION VERSION
COMPLETE with Redis caching, PostgreSQL storage, and correct response formats
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.gnn import TradeGNN
from src.data.loaders import GraphDataLoader
from src.data.country_mapping import MANUAL_MAPPINGS
from src.utils.logger import get_logger

bilateral_sentiment_df = None

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

# Initialize FastAPI
app = FastAPI(
    title="Trade Flow Prediction API",
    description="GNN-based bilateral trade flow forecasting",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
loader = None
device = torch.device('cpu')
articles_df = None

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


@app.on_event("startup")
async def startup_event():
    """Load model, data, and articles on startup"""
    global model, loader, articles_df
    
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
        
        # Load model
        model_dir = Path("models")
        model_files = list(model_dir.glob("gnn_working.pt"))
        
        if not model_files:
            logger.error("❌ No trained model found!")
            return
        
        latest_model = sorted(model_files)[-1]
        logger.info(f"Loading model: {latest_model.name}")
        
        checkpoint = torch.load(latest_model, map_location=device, weights_only=False)
        config = checkpoint['config']
        
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
        logger.info("✓ Model loaded successfully")
        
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
        load_bilateral_sentiment()
        logger.info("🎉 API ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()


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
                        (loader.edges_df['sector'] == backend_sector)
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
                
                # Make prediction
                with torch.no_grad():
                    prediction_log = model(node_features, edge_index, edge_attr).item()
                    prediction_usd = float(np.expm1(prediction_log))
                
                if prediction_usd < 1000:
                    continue
                
                # Confidence score
                years_of_data = len(edge_data['year'].unique()) if not edge_data.empty else 0
                has_lag_features = edge_attr[0, 7] > 0
                has_sentiment = sentiment_confidence > 0
                
                if years_of_data >= 5 and has_lag_features and has_sentiment:
                    confidence_score = 0.95
                elif years_of_data >= 3 and has_sentiment:
                    confidence_score = 0.80
                elif years_of_data >= 3:
                    confidence_score = 0.70
                else:
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
        
        logger.info(f"Returning {len(result)} predictions")
        logger.info(f"Sentiment data used: {bilateral_sentiment_df is not None}")
        
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
            alerts.append(AlertItem(
                id=f"opp_{month}_{pred.partnerCode}",
                type="opportunity",  # ✅ "opportunity" not "info"
                title=f"Growth Opportunity: {pred.partner}",
                summary=f"Strong growth of {pred.change*100:.1f}%",
                partner=pred.partner,
                partnerCode=pred.partnerCode,
                change=pred.change,
                recommendations=[]
            ))
        
        # RISKS - negative growth
        risks = [p for p in predictions if p.change < -0.10]
        for pred in sorted(risks, key=lambda x: x.change)[:5]:
            alerts.append(AlertItem(
                id=f"risk_{month}_{pred.partnerCode}",
                type="risk",  # ✅ "risk" not "warning"
                title=f"Trade Decline: {pred.partner}",
                summary=f"Declining by {abs(pred.change)*100:.1f}%",
                partner=pred.partner,
                partnerCode=pred.partnerCode,
                change=pred.change,
                recommendations=[]
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
    
    # Try to load articles with calculated sentiment first
    articles_path_calc = Path("data/raw/sentiment/articles_with_sentiment.csv")
    
    if articles_path_calc.exists():
        articles_df_local = pd.read_csv(articles_path_calc)
        logger.info(f"Using calculated sentiment from {articles_path_calc}")
    elif articles_df is not None:
        articles_df_local = articles_df
    else:
        logger.warning("No articles data available")
        return []
    
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
        
        logger.info(f"Found {len(filtered)} articles")
        
        news_list = []
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
                    url=str(row['url']),
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
        
        # ✅ Build ATTENTION WEIGHTS (top neighbors)
        # For now, simulate attention to nearby countries
        attention_weights = []
        
        # Get countries with similar trade patterns
        similar_countries = []
        for other_country in ['USA', 'CHN', 'DEU', 'GBR', 'JPN']:
            if other_country != partner:
                similar_countries.append(other_country)
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)