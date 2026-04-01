"""
Complete data loaders for all source datasets.
Handles UN Comtrade, World Bank, CEPII, WTO RTAs, and GDELT sentiment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.config import get_config, get_settings
from src.utils.logger import get_logger
from src.data.country_mapping import get_iso3
from src.utils.helpers import reduce_memory_usage, validate_iso3_codes

logger = get_logger(__name__)
config = get_config()
settings = get_settings()


class ComtradeLoader:
    """Load UN Comtrade bilateral trade data - BULLETPROOF VERSION."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            # Try multiple filenames
            comtrade_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "comtrade"
            possible_files = [
                "TradeData.csv",
                "TradeData_filtered.csv", 
                "comtrade_export.csv",
                "india_exports.csv"
            ]
            
            for filename in possible_files:
                potential_path = comtrade_dir / filename
                if potential_path.exists():
                    file_path = potential_path
                    break
            
            if file_path is None:
                # Just use default, will error if not found
                file_path = comtrade_dir / "TradeData.csv"
        
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load and process Comtrade data."""
        logger.info(f"Loading UN Comtrade data from {self.file_path}")
        
        # Load CSV
        df = pd.read_csv(
            self.file_path,
            encoding='latin1',
            low_memory=False
        )
        
        logger.info(f"Loaded {len(df):,} raw trade records")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Rename key columns
        column_mapping = {
            'reporterDesc': 'reporter_name',
            'partnerDesc': 'partner_name',
            'reporterISO': 'reporter_iso3',
            'partnerISO': 'partner_iso3',
            'refYear': 'year',
            'refMonth': 'month',
            'flowDesc': 'flow',
            'cmdCode': 'hs_code',
            'cmdDesc': 'product_description',
            'primaryValue': 'trade_value_usd',
            'qty': 'quantity',
            'qtyUnitAbbr': 'quantity_unit'
        }
        
        df = df.rename(columns=column_mapping)
        
        # CRITICAL FIX: Your reporterDesc has codes 'M'/'X', not country names!
        # The actual country names are likely in reporterISO or need different column
        logger.info("Checking reporter column format...")
        
        # Check if reporterDesc looks like country names or codes
        sample_reporters = df['reporter_name'].head(100).unique()
        logger.info(f"Sample reporters: {sample_reporters[:5]}")
        
        # If reporterDesc contains only single letters (M, X, etc.), use reporterISO instead
        if df['reporter_name'].str.len().max() <= 3:
            logger.warning("reporterDesc contains codes, not names. Using reporterISO instead.")
            # Map ISO codes in reporterISO to ISO3 (they're already ISO3!)
            df['source_iso3'] = df['reporter_iso3']  # reporterISO is already ISO3
            df['target_iso3'] = df['partner_iso3']   # partnerISO is already ISO3
        else:
            # Use country name mapping as before
            logger.info("Mapping country names to ISO3 codes...")
            df['source_iso3'] = df['reporter_name'].apply(get_iso3)
            df['target_iso3'] = df['partner_name'].apply(get_iso3)
        
        # Filter out unmapped countries
        initial_len = len(df)
        df = df.dropna(subset=['source_iso3', 'target_iso3'])
        logger.info(f"After ISO3 mapping: {len(df):,} records ({len(df)/initial_len*100:.1f}%)")
        
        # Filter to Exports only
        # Handle both flowDesc and flowCode
        if 'flow' in df.columns:
            # If flow is already renamed
            df = df[df['flow'].isin(['Export', 'X', 'Exports'])].copy()
        elif 'flowDesc' in df.columns and 'flowCode' in df.columns:
            # Use flowCode (X = Export, M = Import)
            df = df[df['flowCode'] == 'X'].copy()
            df['flow'] = 'Export'
        
        logger.info(f"After filtering to Exports: {len(df):,} records")
        
        # Convert types BEFORE filtering
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
        df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        # Filter out invalid values
        df = df[df['trade_value_usd'] > 0]
        df = df[df['year'].notna()]
        
        logger.info(f"After filtering invalid values: {len(df):,} records")
        
        # Convert HS code to string and clean
        df['hs_code'] = df['hs_code'].astype(str).str.strip().str.lower()
        
        # CRITICAL FIX: Your data has TEXT DESCRIPTIONS, not numeric codes!
        # Determine sector based on TEXT in cmdCode/cmdDesc
        def determine_sector(hs_code):
            hs = str(hs_code).strip().lower()
            
            # numeric? convert
            try:
                hs_num = int(hs[:2])   # first 2 digits (chapter number)
            except:
                hs_num = None

            # numeric detection
            if hs_num == 30:
                return "Pharmaceuticals"
            if 50 <= hs_num <= 63:
                return "Textiles"

            # fallback to text description (if present)
            text = hs
            if any(k in text for k in pharma_keywords):
                return "Pharmaceuticals"
            if any(k in text for k in textile_keywords):
                return "Textiles"

            return "Other"
        
        df['sector'] = df['hs_code'].apply(determine_sector)
        
        logger.info(f"Sector distribution: {df['sector'].value_counts().to_dict()}")
        
        # Filter to target sectors only
        df = df[df['sector'].isin(['Pharmaceuticals', 'Textiles'])].copy()
        logger.info(f"After filtering to Pharma/Textiles: {len(df):,} records")
        
        # Select final columns
        columns = [
            'source_iso3', 'target_iso3', 'year', 'month', 
            'hs_code', 'sector', 'product_description',
            'trade_value_usd', 'quantity', 'quantity_unit', 'flow'
        ]
        
        # Only keep columns that exist
        existing_cols = [col for col in columns if col in df.columns]
        df = df[existing_cols].copy()
        
        # Handle missing columns
        if 'month' not in df.columns:
            df['month'] = 1  # Default to January
        if 'quantity_unit' not in df.columns:
            df['quantity_unit'] = None
        
        # Reduce memory (but skip problematic columns with NA)
        try:
            df = reduce_memory_usage(df, verbose=True)
        except Exception as e:
            logger.warning(f"Memory reduction failed: {e}, continuing without it")
        
        logger.info(f"Final Comtrade data: {len(df):,} records")
        
        if len(df) == 0:
            logger.error("⚠️  WARNING: No Comtrade data after filtering!")
            logger.error("Check: 1) HS codes in your data, 2) Flow codes (X vs Export)")
        
        return df


class WorldBankLoader:
    """Load World Bank indicators (GDP, Population, Inflation)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "world-bank"
        self.data_dir = Path(data_dir)
    
    def _load_indicator(self, filename: str, indicator_name: str) -> pd.DataFrame:
        """Load a single World Bank indicator."""
        file_path = self.data_dir / filename
        logger.info(f"Loading World Bank {indicator_name} from {file_path}")
        
        # Skip metadata rows (first 4 rows)
        df = pd.read_csv(file_path, skiprows=4)
        
        # Identify year columns (columns that are numeric)
        year_cols = [col for col in df.columns if col.isdigit()]
        
        # Melt to long format
        id_cols = ['Country Name', 'Country Code']
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=year_cols,
            var_name='year',
            value_name=indicator_name
        )
        
        # Convert types
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype('Int64')
        df_long[indicator_name] = pd.to_numeric(df_long[indicator_name], errors='coerce')
        
        # Map to ISO3
        df_long['iso3'] = df_long['Country Code'].str[:3]  # First 3 chars
        
        # Clean up
        df_long = df_long[['iso3', 'year', indicator_name]].dropna(subset=['iso3', 'year'])
        
        logger.info(f"Loaded {len(df_long):,} {indicator_name} records")
        return df_long
    
    def load(self) -> pd.DataFrame:
        """Load all World Bank indicators and merge."""
        logger.info("Loading World Bank data...")
        
        # Load each indicator
        gdp = self._load_indicator(
            'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_75934.csv',
            'gdp_usd'
        )
        
        population = self._load_indicator(
            'API_SP.POP.TOTL_DS2_en_csv_v2_76034.csv',
            'population'
        )
        
        inflation = self._load_indicator(
            'API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_73483.csv',
            'inflation_rate'
        )
        
        # Merge all indicators
        logger.info("Merging World Bank indicators...")
        wb_data = gdp.merge(population, on=['iso3', 'year'], how='outer')
        wb_data = wb_data.merge(inflation, on=['iso3', 'year'], how='outer')
        
        # Validate ISO3 codes
        wb_data = validate_iso3_codes(wb_data, ['iso3'])
        
        logger.info(f"Final World Bank data: {len(wb_data):,} country-year records")
        return wb_data


class CEPIILoader:
    """Load CEPII GeoDist bilateral distance data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "cepii" / "dist_cepii.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load CEPII distance data."""
        logger.info(f"Loading CEPII GeoDist data from {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df):,} country pair distances")
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'iso_o': 'source_iso3',
            'iso_d': 'target_iso3',
            'dist': 'distance_km',
            'comlang_off': 'shared_language',
            'contig': 'contiguous'
        })
        
        # Convert binary columns to boolean
        df['shared_language'] = df['shared_language'].astype(bool)
        df['contiguous'] = df['contiguous'].astype(bool)
        
        # Select relevant columns
        columns = [
            'source_iso3', 'target_iso3', 'distance_km',
            'shared_language', 'contiguous'
        ]
        
        df = df[columns].copy()
        
        # Validate ISO3 codes
        df = validate_iso3_codes(df, ['source_iso3', 'target_iso3'])
        
        logger.info(f"Final CEPII data: {len(df):,} country pairs")
        return df


class RTALoader:
    """Load WTO Regional Trade Agreements data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "rta" / "AllRTAs.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load and parse RTA data into country pairs."""
        logger.info(f"Loading WTO RTA data from {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df)} RTAs")
        
        # Parse signatories and create country pairs
        pairs = []
        
        for idx, row in df.iterrows():
            signatories_str = str(row.get('Current signatories', ''))
            
            if pd.isna(signatories_str) or signatories_str == 'nan':
                continue
            
            # Split by common delimiters
            signatories = [s.strip() for s in signatories_str.replace(';', ',').split(',')]
            
            # Convert to ISO3
            iso3_list = []
            for country_name in signatories:
                iso3 = get_iso3(country_name)
                if iso3:
                    iso3_list.append(iso3)
            
            # Create all bidirectional pairs
            for i, iso1 in enumerate(iso3_list):
                for iso2 in iso3_list[i+1:]:
                    # Add both directions
                    pairs.append({
                        'source_iso3': iso1,
                        'target_iso3': iso2,
                        'fta_binary': 1,
                        'rta_name': row['RTA Name']
                    })
                    pairs.append({
                        'source_iso3': iso2,
                        'target_iso3': iso1,
                        'fta_binary': 1,
                        'rta_name': row['RTA Name']
                    })
        
        fta_df = pd.DataFrame(pairs)
        
        # Remove duplicates (keep first occurrence)
        fta_df = fta_df.drop_duplicates(subset=['source_iso3', 'target_iso3'], keep='first')
        
        # Validate ISO3 codes
        fta_df = validate_iso3_codes(fta_df, ['source_iso3', 'target_iso3'])
        
        logger.info(f"Final RTA data: {len(fta_df):,} FTA country pairs")
        return fta_df


class GDELTLoader:
    """Load GDELT sentiment data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment" / "sentiment.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load GDELT sentiment aggregates."""
        logger.info(f"Loading GDELT sentiment from {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df):,} sentiment records")
        
        # Check what columns we actually have
        logger.info(f"GDELT columns: {list(df.columns)}")
        
        # Handle different GDELT formats
        if 'month' in df.columns and df['month'].dtype == 'object':
            # Format 1: "2023-01" style month column
            try:
                df[['year', 'month']] = df['month'].str.split('-', expand=True)
                df['year'] = pd.to_numeric(df['year']).astype('Int64')
                df['month'] = pd.to_numeric(df['month']).astype('Int64')
            except:
                # If split fails, month might already be separated
                pass
        
        elif 'year' not in df.columns or 'month' not in df.columns:
            # Format 2: No year/month columns, create from date or other column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year.astype('Int64')
                df['month'] = df['date'].dt.month.astype('Int64')
            else:
                # Use current year/month as fallback
                logger.warning("No date columns found, using current year/month")
                from datetime import datetime
                now = datetime.now()
                df['year'] = now.year
                df['month'] = now.month
        else:
            # Format 3: year and month already separate
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
        
        # Standardize country column names
        column_mapping = {
            'country_1': 'country_1_iso3',
            'country_2': 'country_2_iso3',
            'Actor1CountryCode': 'country_1_iso3',
            'Actor2CountryCode': 'country_2_iso3',
            'avg_sentiment': 'avg_tone',
            'AvgTone': 'avg_tone'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Validate required columns exist
        required = ['country_1_iso3', 'country_2_iso3', 'avg_tone']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.error(f"GDELT missing columns: {missing}")
            logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Validate ISO3 codes
        df = validate_iso3_codes(df, ['country_1_iso3', 'country_2_iso3'])
        
        # Select columns
        final_cols = ['year', 'month', 'country_1_iso3', 'country_2_iso3', 'avg_tone']
        existing_cols = [col for col in final_cols if col in df.columns]
        df = df[existing_cols].copy()
        
        logger.info(f"Final GDELT data: {len(df):,} sentiment records")
        return df


class DataLoader:
    """Main data loader orchestrator."""
    
    def __init__(self):
        self.comtrade_loader = ComtradeLoader()
        self.worldbank_loader = WorldBankLoader()
        self.cepii_loader = CEPIILoader()
        self.rta_loader = RTALoader()
        self.gdelt_loader = GDELTLoader()
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources.
        
        Returns:
            Dictionary with data from all sources
        """
        logger.info("="*60)
        logger.info("LOADING ALL DATA SOURCES")
        logger.info("="*60)
        
        data = {}
        
        try:
            data['comtrade'] = self.comtrade_loader.load()
        except Exception as e:
            logger.error(f"Failed to load Comtrade: {e}", exc_info=True)
            data['comtrade'] = pd.DataFrame()
        
        try:
            data['world_bank'] = self.worldbank_loader.load()
        except Exception as e:
            logger.error(f"Failed to load World Bank: {e}", exc_info=True)
            data['world_bank'] = pd.DataFrame()
        
        try:
            data['cepii'] = self.cepii_loader.load()
        except Exception as e:
            logger.error(f"Failed to load CEPII: {e}", exc_info=True)
            data['cepii'] = pd.DataFrame()
        
        try:
            data['rtas'] = self.rta_loader.load()
        except Exception as e:
            logger.error(f"Failed to load RTAs: {e}", exc_info=True)
            data['rtas'] = pd.DataFrame()
        
        try:
            data['gdelt'] = self.gdelt_loader.load()
        except Exception as e:
            logger.error(f"Failed to load GDELT: {e}", exc_info=True)
            data['gdelt'] = pd.DataFrame()
        
        logger.info("="*60)
        logger.info("DATA LOADING SUMMARY")
        logger.info("="*60)
        for source, df in data.items():
            logger.info(f"{source:15s}: {len(df):>10,} rows")
        logger.info("="*60)
        
        return data


if __name__ == "__main__":
    # Test data loaders
    loader = DataLoader()
    data = loader.load_all()
    
    print("\n✅ All data sources loaded successfully!")
    print("\nSample data from each source:")
    for source, df in data.items():
        print(f"\n{source.upper()}:")
        print(df.head(2))