"""
Database configuration and query utilities for NLP analytics
Reads database connection information from environment variables (preferred)
or falls back to parsing docker-compose.yml
Supports .env files for local development

Key Functions:
- get_all_records(): Get all records from database
- get_records_above_percentile(): Filter records above percentile threshold
- get_records_by_category(): Get records by category value
- get_score_distribution(): Get distribution statistics for score columns
- get_score_histogram_bins(): Get histogram data for scores
- insert_job_with_embedding(): Insert job with vector embedding
- find_similar_jobs(): Find similar jobs using cosine similarity
"""
import os
import psycopg2
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from app-streamlit directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    # Also try project root
    project_root = os.path.join(current_dir, "..", "..")
    env_path_root = os.path.join(project_root, ".env")
    if os.path.exists(env_path_root):
        load_dotenv(env_path_root)
except ImportError:
    # python-dotenv not available, skip .env loading
    pass

# Try to import yaml for docker-compose parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try to import OpenAI for embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


#  ORIGINAL_COLUMNS = [
#         "CHECK_NUMBER", "BENECUSTID_i", "BeneCustSegment", "Bene_ER", "BeneSegment", "BENECOUNTRY_i", "BENECOUNTRY_HR", "BYORDERCUSTID_i", "ByorderCustSegment", "Byorder_ER", "Byorder_ER_Segment", "BYORDERCOUNTRY_i", "BYORDERCOUNTRY_HR", "BENEBANKCUSTID_i", "BeneBankCustSegment", "BENEBANKCOUNTRY_i", "BENEBANKCOUNTRY_HR", "BYORDERBANKCUSTID_i", "ByorderBankCustSegment", "BYORDERBANKCOUNTRY_i", "BYORDERBANKCOUNTRY_HR", "INTERMEDIARYCUSTID_i", "INTERMEDIARY_segment", "SEND_BANK_COUNTRY_i", "SEND_BANK_COUNTRY_HR", "INTERMEDIARYCUSTID2_i", "INTERMEDIARY2_segment", "REC_BANK_COUNTRY_I", "REC_BANK_COUNTRY_HR", "INTERMEDIARYCUSTID3_i", "INTERMEDIARY3_segment", "INTERMEDIARYCOUNTRY3_I", "INTERMEDIARYCOUNTRY3_HR", "INTERMEDIARYCUSTID4_i", "INTERMEDIARY4_segment", "INTERMEDIARYCOUNTRY4_i", "INTERMEDIARYCOUNTRY4_HR", "executing_party_i", "executing_party_Segment", "TRANSACTION_CDI_CODE", "TRANSACTION_CDI_CODE", "PRIMARY_MEDIUM_DESC", "SECONDARY_MEDIUM_DESC", "MECHANISM_DESC",
#     ]

RISK_OVER_TIMES = [
"TRANSACTION_KEY", "DATE_KEY", "CURRENCY_AMOUNT", 
'hbos_anomaly_score','pca_isolation_forest_score','hbos_pca_isolation_forest_score'
]

SUMMARY_TABLE_COLUMNS = [
"TRANSACTION_KEY", "DATE_KEY", "CURRENCY_AMOUNT", 
'byorder_type','byorder_id','beneficiary_type','beneficiary_id',
'byorder_to_bene',
'MECHANISM_DESC',
'hbos_anomaly_score','pca_isolation_forest_score','hbos_pca_isolation_forest_score'
]

SELECTED_COLUMNS = [
    "TRANSACTION_KEY", "DATE_KEY", "CURRENCY_AMOUNT", 
    'beneficiary_type','beneficiary_id','beneficiary_segment','byorder_type','byorder_id','byorder_segment','byorder_to_bene',
    'unknown_high_risk_flag',

    # High-priority risk indicators
    'cross_border', 'sender_high_risk_country', 'receiver_high_risk_country',
    
    # Amount pattern features (high anomaly indicators)
    'is_round_any',
    
    # Intermediary and layering features (high risk indicators)
    'count_hit_intermediary',
    
    # Behavioral pattern features (anomaly indicators)
    'byorder_to_bene_repetitive_amount_flag',
    
    # High-value transaction flags
    'total_above_100K_flag', 'total_above_500K_flag', 'total_above_1M_flag',

    # Mechanism Description Analysis
    'MECHANISM_DESC_INT',
    'MECHANISM_DESC',
    
    'rule_base_risk_score','hbos_anomaly_score','pca_isolation_forest_score','hbos_pca_isolation_forest_score'
]

def get_db_config_from_env() -> Optional[Dict[str, Any]]:
    """
    Get database configuration from environment variables
    Returns a dictionary with database connection information, or None if not all required vars are set
    """
    # Try to get DATABASE_URL first (most complete)
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        # Parse DATABASE_URL format: postgresql://user:password@host:port/database
        try:
            # Remove postgresql:// prefix
            url_part = database_url.replace('postgresql://', '')
            # Split by @ to separate credentials from host
            creds_part, host_part = url_part.split('@')
            user, password = creds_part.split(':')
            # Split host part
            if ':' in host_part:
                host_port, database = host_part.split('/')
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_part.split('/')[0]
                port = 5432
                database = host_part.split('/')[1]
            
            return {
                'user': user,
                'password': password,
                'host': host,
                'port': port,
                'database': database,
                'source': 'DATABASE_URL'
            }
        except Exception:
            # If parsing fails, continue to individual env vars
            pass
    
    # Try individual environment variables
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    database = os.getenv('POSTGRES_DB')
    host = os.getenv('POSTGRES_HOST')
    port = os.getenv('POSTGRES_PORT')
    
    if all([user, password, database, host, port]):
        return {
            'user': user,
            'password': password,
            'host': host,
            'port': int(port),
            'database': database,
            'source': 'ENV_VARS'
        }
    
    return None


def get_docker_compose_path() -> str:
    """
    Get the path to docker-compose.yml file
    Returns the path relative to the project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from app-streamlit/functions to project root
    project_root = os.path.join(current_dir, "..", "..")
    docker_compose_path = os.path.join(project_root, "docker-compose.yml")
    return os.path.abspath(docker_compose_path)


def parse_docker_compose() -> Dict[str, Any]:
    """
    Parse docker-compose.yml and extract database configuration
    Returns a dictionary with database connection information
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to parse docker-compose.yml. Install with: pip install PyYAML")
    
    docker_compose_path = get_docker_compose_path()
    
    if not os.path.exists(docker_compose_path):
        raise FileNotFoundError(
            f"docker-compose.yml not found at {docker_compose_path}"
        )
    
    with open(docker_compose_path, 'r') as f:
        compose_data = yaml.safe_load(f)
    
    # Extract postgres service configuration
    postgres_service = compose_data.get('services', {}).get('nlp-postgres', {})
    
    # Extract environment variables
    env_vars = postgres_service.get('environment', {})
    
    # Extract port mapping (format: "45432:5432")
    ports = postgres_service.get('ports', [])
    host_port = None
    container_port = None
    if ports:
        port_mapping = ports[0].split(':')
        host_port = int(port_mapping[0])  # External port (45432)
        container_port = int(port_mapping[1])  # Internal port (5432)
    
    # Extract database credentials
    db_config = {
        'user': env_vars.get('POSTGRES_USER', 'admin'),
        'password': env_vars.get('POSTGRES_PASSWORD', 'PassW0rd'),
        'database': env_vars.get('POSTGRES_DB', 'db'),
        'container_name': postgres_service.get('container_name', 'nlp-postgres'),
        'host_port': host_port,
        'container_port': container_port or 5432,
        'network': postgres_service.get('networks', ['nlp-net'])[0] if postgres_service.get('networks') else 'nlp-net',
        'source': 'docker-compose.yml'
    }
    
    return db_config


def get_db_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables (preferred) or docker-compose.yml (fallback)
    Returns cached configuration dictionary
    """
    if not hasattr(get_db_config, '_cached_config'):
        # Try environment variables first
        env_config = get_db_config_from_env()
        if env_config:
            get_db_config._cached_config = env_config
        else:
            # Fall back to docker-compose.yml
            compose_config = parse_docker_compose()
            get_db_config._cached_config = compose_config
    
    return get_db_config._cached_config


def get_database_url(use_host: bool = False) -> str:
    """
    Get SQLAlchemy database URL
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        Database connection URL string
    """
    config = get_db_config()
    
    # If config came from environment variables, use it directly
    if config.get('source') in ['DATABASE_URL', 'ENV_VARS']:
        host = config['host']
        port = config['port']
    else:
        # Config came from docker-compose.yml
        if use_host or not _is_running_in_docker():
            # Connection from host machine
            host = 'localhost'
            port = config.get('host_port') or 45432
        else:
            # Connection from within Docker network
            host = config.get('container_name', 'nlp-postgres')
            port = config.get('container_port', 5432)
    
    user = config['user']
    password = config['password']
    database = config['database']
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _is_running_in_docker() -> bool:
    """
    Check if code is running inside a Docker container
    """
    return os.path.exists('/.dockerenv') or os.path.exists('/proc/self/cgroup')


def get_db_connection_params(use_host: bool = False) -> Dict[str, Any]:
    """
    Get database connection parameters as dictionary
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        Dictionary with connection parameters
    """
    config = get_db_config()
    
    # If config came from environment variables, use it directly
    if config.get('source') in ['DATABASE_URL', 'ENV_VARS']:
        host = config['host']
        port = config['port']
    else:
        # Config came from docker-compose.yml
        if use_host or not _is_running_in_docker():
            host = 'localhost'
            port = config.get('host_port') or 45432
        else:
            host = config.get('container_name', 'nlp-postgres')
            port = config.get('container_port', 5432)
    
    return {
        'host': host,
        'port': port,
        'user': config['user'],
        'password': config['password'],
        'database': config['database']
    }


def create_db_engine(use_host: bool = False):
    """
    Create SQLAlchemy engine for database connections
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        SQLAlchemy engine
    """
    database_url = get_database_url(use_host)
    return create_engine(database_url)


def get_db_session(use_host: bool = False):
    """
    Get database session using SQLAlchemy
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        SQLAlchemy session
    """
    engine = create_db_engine(use_host)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def execute_query(query: str, use_host: bool = False, return_df: bool = True) -> Optional[pd.DataFrame]:
    """
    Execute a SQL query and return results as DataFrame
    Args:
        query: SQL query string
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
        return_df: If True, return results as pandas DataFrame. If False, return raw results
    Returns:
        pandas DataFrame with query results, or None if return_df is False
    """
    engine = create_db_engine(use_host)
    
    try:
        with engine.connect() as connection:
            if return_df:
                result_df = pd.read_sql_query(text(query), connection)
                return result_df
            else:
                result = connection.execute(text(query))
                return result.fetchall()
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")


def get_psycopg2_connection(use_host: bool = False):
    """
    Get a raw psycopg2 database connection
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        psycopg2 connection object
    """
    params = get_db_connection_params(use_host)
    return psycopg2.connect(**params)


def test_connection(use_host: bool = False) -> bool:
    """
    Test database connection
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
                  If False, use container name/host from config (for connections within Docker network)
    Returns:
        True if connection successful, False otherwise
    """
    try:
        engine = create_db_engine(use_host)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


# Example usage functions
def get_all_tables(use_host: bool = False) -> pd.DataFrame:
    """
    Get list of all tables in the database
    """
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """
    return execute_query(query, use_host=use_host)


def get_table_schema(table_name: str, use_host: bool = False) -> pd.DataFrame:
    """
    Get schema information for a specific table
    """
    query = f"""
    SELECT 
        column_name,
        data_type,
        character_maximum_length,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = '{table_name}'
    ORDER BY ordinal_position;
    """
    return execute_query(query, use_host=use_host)


def get_table_row_count(table_name: str, use_host: bool = False) -> int:
    """
    Get row count for a specific table
    """
    query = f"SELECT COUNT(*) as count FROM {table_name};"
    result = execute_query(query, use_host=use_host)
    return result['count'].iloc[0] if result is not None and len(result) > 0 else 0


def get_query_where_byorder_to_bene(byorder_to_bene_value: str, table_name: str = "transactions") -> str:
    """
    Generate a SQL query string to select transactions where byorder_to_bene equals the specified value
    Args:
        byorder_to_bene_value: The value to filter by in the byorder_to_bene column
        table_name: Name of the table to query (default: "transactions")
    Returns:
        SQL query string
    """
    columns_str = ', '.join(f'"{col}"' for col in SELECTED_COLUMNS)
    query = f"SELECT {columns_str} FROM {table_name} WHERE byorder_to_bene = '{byorder_to_bene_value}';"
    return query


def get_transactions_by_byorder_to_bene(byorder_to_bene_value: str, use_host: bool = False, table_name: str = "transactions") -> pd.DataFrame:
    """
    Execute a query to get all transactions where byorder_to_bene equals the specified value
    Args:
        byorder_to_bene_value: The value to filter by in the byorder_to_bene column
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        pandas DataFrame with matching transactions
    """
    query = get_query_where_byorder_to_bene(byorder_to_bene_value, table_name)
    return execute_query(query, use_host=use_host)


def get_similar_transactions(transaction_key: str, use_host: bool = False, table_name: str = "transactions", limit: int = 10) -> pd.DataFrame:
    """
    Get similar transactions based on the same byorder_to_bene relationship, excluding the current transaction
    Args:
        transaction_key: The TRANSACTION_KEY of the reference transaction
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
        limit: Maximum number of similar transactions to return
    Returns:
        pandas DataFrame with similar transactions
    """
    # First get the byorder_to_bene value for the reference transaction
    query_get_key = f'SELECT byorder_to_bene FROM {table_name} WHERE "TRANSACTION_KEY" = \'{transaction_key}\';'
    key_df = execute_query(query_get_key, use_host=use_host)
    
    if key_df is None or key_df.empty:
        return pd.DataFrame()
    
    byorder_to_bene_value = key_df.iloc[0]['byorder_to_bene']
    
    # Get all transactions with the same byorder_to_bene, excluding the current transaction
    columns_str = ', '.join(f'"{col}"' for col in SELECTED_COLUMNS)
    query = f'SELECT {columns_str} FROM {table_name} WHERE byorder_to_bene = \'{byorder_to_bene_value}\' AND "TRANSACTION_KEY" != \'{transaction_key}\' ORDER BY "DATE_KEY" DESC LIMIT {limit};'
    
    return execute_query(query, use_host=use_host)


def get_score_distribution(use_host: bool = False, table_name: str = "transactions") -> pd.DataFrame:
    """
    Get distribution statistics for all score columns in the transactions table
    Returns statistics like count, mean, std, min, max, and percentiles for each score column
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        pandas DataFrame with distribution statistics for each score column
    """
    score_columns = [
        'rule_base_risk_score',
        'hbos_anomaly_score', 
        'pca_isolation_forest_score',
        'hbos_pca_isolation_forest_score'
    ]
    
    # Build query to get statistics for each score column
    stats_queries = []
    for col in score_columns:
        stats_queries.append(f"""
            SELECT 
                '{col}' as score_type,
                COUNT({col}) as count,
                AVG({col}) as mean,
                STDDEV({col}) as std,
                MIN({col}) as min,
                MAX({col}) as max,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) as q25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {col}) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) as q75,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {col}) as q95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {col}) as q99
            FROM {table_name}
            WHERE {col} IS NOT NULL
        """)
    
    # Combine all queries with UNION ALL
    combined_query = " UNION ALL ".join(stats_queries)
    
    return execute_query(combined_query, use_host=use_host)


def get_anomaly_score_histogram_bins(score_column: str = 'hbos_pca_isolation_forest_score', 
                                   bins: int = 50, use_host: bool = False, 
                                   table_name: str = "transactions") -> pd.DataFrame:
    """
    Get histogram bin data for anomaly score distribution with specified number of bins
    Args:
        score_column: Name of the score column to analyze (default: 'hbos_pca_isolation_forest_score')
        bins: Number of bins for the histogram (default: 50)
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        pandas DataFrame with bin ranges and counts for histogram plotting
    """
    query = f"""
    WITH score_stats AS (
        SELECT 
            MIN("{score_column}") as min_score,
            MAX("{score_column}") as max_score
        FROM {table_name}
        WHERE "{score_column}" IS NOT NULL
    ),
    bin_data AS (
        SELECT 
            "{score_column}",
            CASE 
                WHEN (SELECT max_score - min_score FROM score_stats) = 0 THEN 0
                ELSE FLOOR(
                    ("{score_column}" - (SELECT min_score FROM score_stats)) / 
                    ((SELECT max_score - min_score FROM score_stats) / {bins})
                )::integer
            END as bin_index
        FROM {table_name}
        WHERE "{score_column}" IS NOT NULL
    ),
    bin_ranges AS (
        SELECT 
            bin_index,
            (SELECT min_score FROM score_stats) + 
            (bin_index * ((SELECT max_score - min_score FROM score_stats) / {bins})) as bin_start,
            (SELECT min_score FROM score_stats) + 
            ((bin_index + 1) * ((SELECT max_score - min_score FROM score_stats) / {bins})) as bin_end,
            COUNT(*) as count
        FROM bin_data
        GROUP BY bin_index
        ORDER BY bin_index
    )
    SELECT 
        bin_index,
        ROUND(bin_start::numeric, 4) as bin_start,
        ROUND(bin_end::numeric, 4) as bin_end,
        ROUND(((bin_start + bin_end) / 2)::numeric, 4) as bin_center,
        count
    FROM bin_ranges
    ORDER BY bin_index;
    """
    
    return execute_query(query, use_host=use_host)


def get_all_transactions(use_host: bool = False, table_name: str = "transactions") -> pd.DataFrame:
    """
    Get all transactions from the database
    Args:
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        pandas DataFrame with all transactions
    """
    columns_str = ', '.join(f'"{col}"' for col in SELECTED_COLUMNS)
    query = f"SELECT {columns_str} FROM {table_name}"
    return execute_query(query, use_host=use_host)


def get_transactions_above_threshold(score_column: str, threshold: float, use_host: bool = False) -> pd.DataFrame:
    """
    Get transactions where the specified score column is above the given percentile threshold
    Args:
        score_column: Name of the score column to filter by (e.g., 'hbos_anomaly_score')
        threshold: Threshold value for filtering
        use_host: If True, use localhost with host port (for connections from host machine)
    Returns:
        pandas DataFrame with filtered transactions
    """
    columns_str = ', '.join(f'"{col}"' for col in SUMMARY_TABLE_COLUMNS)
    query = f"""
    SELECT {columns_str}
    FROM transactions
    WHERE "{score_column}" >= {threshold}
    ORDER BY "{score_column}" DESC;
    """
    result = execute_query(query, use_host=use_host)
    
    if result is not None and len(result) > 0:
        # Apply memory-efficient dtypes
        for col in result.columns:
            if col == "CURRENCY_AMOUNT":
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
            elif col == "DATE_KEY":
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif col in ["BENECOUNTRY_HR", "BYORDERCOUNTRY_HR", "Check_flag"]:
                result[col] = result[col].astype("category")
            elif col in ["hbos_anomaly_score", "pca_isolation_forest_score", "hbos_pca_isolation_forest_score", "rule_base_risk_score"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

        # This ensures consistent decile ranks across all transactions
        hbos_pca_if_col = 'hbos_pca_isolation_forest_score'
        if hbos_pca_if_col in result:
            # Get global decile thresholds for consistent ranking across all pages
            global_thresholds = get_decile_thresholds(hbos_pca_if_col, use_host=use_host)
            
            if global_thresholds and len(global_thresholds) == 11:
                # Apply global thresholds to assign decile ranks
                score_values = pd.to_numeric(result[hbos_pca_if_col], errors='coerce')
                decile_ranks = pd.Series(index=result.index, dtype=int)
                
                # DR1: Highest risk (90th-100th percentile) - scores >= 90th percentile
                decile_ranks[score_values >= global_thresholds[9]] = 1
                # DR2: (80th-90th percentile)
                decile_ranks[(score_values >= global_thresholds[8]) & (score_values < global_thresholds[9])] = 2
                # DR3: (70th-80th percentile)
                decile_ranks[(score_values >= global_thresholds[7]) & (score_values < global_thresholds[8])] = 3
                # DR4: (60th-70th percentile)
                decile_ranks[(score_values >= global_thresholds[6]) & (score_values < global_thresholds[7])] = 4
                # DR5: (50th-60th percentile)
                decile_ranks[(score_values >= global_thresholds[5]) & (score_values < global_thresholds[6])] = 5
                # DR6: (40th-50th percentile)
                decile_ranks[(score_values >= global_thresholds[4]) & (score_values < global_thresholds[5])] = 6
                # DR7: (30th-40th percentile)
                decile_ranks[(score_values >= global_thresholds[3]) & (score_values < global_thresholds[4])] = 7
                # DR8: (20th-30th percentile)
                decile_ranks[(score_values >= global_thresholds[2]) & (score_values < global_thresholds[3])] = 8
                # DR9: (10th-20th percentile)
                decile_ranks[(score_values >= global_thresholds[1]) & (score_values < global_thresholds[2])] = 9
                # DR10: Lowest risk (0th-10th percentile) - scores < 10th percentile
                decile_ranks[score_values < global_thresholds[1]] = 10
                
                # Fill NaN values with 10 (lowest risk)
                decile_ranks = decile_ranks.fillna(10).astype(int)
                # Convert to DR1, DR2, ..., DR10 format
                result['hbos_pca_if_decile_rank'] = 'DR' + decile_ranks.astype(str)
            else:
                result['hbos_pca_if_decile_rank'] = 'DR10'
    
    return result


def get_transactions_above_threshold_all_columns(score_column: str, threshold: float, use_host: bool = False, table_name: str = "transactions") -> pd.DataFrame:
    """
    Get all columns from transactions where the specified score column is above the given threshold
    This function returns ALL columns from the database table, not just a subset
    Args:
        score_column: Name of the score column to filter by (e.g., 'hbos_anomaly_score')
        threshold: Threshold value for filtering
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        pandas DataFrame with all columns for filtered transactions
    """
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE "{score_column}" >= {threshold}
    ORDER BY "{score_column}" DESC;
    """
    result = execute_query(query, use_host=use_host)
    
    if result is not None and len(result) > 0:
        # Apply memory-efficient dtypes for common columns
        for col in result.columns:
            if col == "CURRENCY_AMOUNT":
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
            elif col == "DATE_KEY":
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif col in ["hbos_anomaly_score", "pca_isolation_forest_score", "hbos_pca_isolation_forest_score", "rule_base_risk_score"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
    
    return result


def get_model_column(selected_model: str) -> str:
    """
    Map model name to column name
    Args:
        selected_model: Model name (e.g., 'HBOS', 'PCA+IF', 'HBOS & PCA+IF')
    Returns:
        Column name for the model
    """
    model_mapping = {
        "HBOS": "hbos_anomaly_score",
        "PCA+IF": "pca_isolation_forest_score", 
        "HBOS & PCA+IF": "hbos_pca_isolation_forest_score"
    }
    return model_mapping.get(selected_model, "hbos_anomaly_score")


def get_total_transactions_above_percentile(score_column: str, percentile: float, use_host: bool = False) -> int:
    """
    Get the total count of transactions that have a score >= the given percentile threshold
    Args:
        score_column: Name of the score column to filter by (e.g., 'hbos_anomaly_score')
        percentile: Percentile threshold (e.g., 90.0 for 90th percentile)
        use_host: If True, use localhost with host port (for connections from host machine)
    Returns:
        Total count of transactions above the percentile threshold
    """
    query = f"""
    WITH score_stats AS (
        SELECT
            PERCENTILE_CONT({percentile}/100.0) WITHIN GROUP (ORDER BY "{score_column}") as threshold
        FROM transactions
        WHERE "{score_column}" IS NOT NULL
    )
    SELECT COUNT(*) as total_count
    FROM transactions, score_stats
    WHERE "{score_column}" >= score_stats.threshold
    """

    result = execute_query(query, use_host=use_host)
    if result is not None and len(result) > 0:
        return int(result.iloc[0]['total_count'])
    return []


def get_transaction_counts(score_column: str, high_risk_percentile: float, critical_percentile: float) -> Dict[str, Any]:
    """
    Get counts of critical, high risk, and normal transactions based on model and percentiles
    Args:
        selected_model: Model name (e.g., 'HBOS', 'PCA+IF', 'HBOS & PCA+IF')
        high_risk_percentile: Percentile threshold for high risk (e.g., 90.0)
        critical_percentile: Percentile threshold for critical risk (e.g., 95.0)
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        Dictionary with counts and thresholds: {'critical': int, 'high_risk': int, 'normal': int, 'high_risk_threshold': float, 'critical_threshold': float}
    """
    # score_column = get_model_column(selected_model)

    query = f"""
    WITH score_stats AS (
        SELECT
            PERCENTILE_CONT({high_risk_percentile}/100.0) WITHIN GROUP (ORDER BY "{score_column}") as high_risk_threshold,
            PERCENTILE_CONT({critical_percentile}/100.0) WITHIN GROUP (ORDER BY "{score_column}") as critical_threshold,
            PERCENTILE_CONT({high_risk_percentile}/100.0) WITHIN GROUP (ORDER BY "hbos_anomaly_score") as hbos_anomaly_score_high_risk_threshold,
            PERCENTILE_CONT({critical_percentile}/100.0) WITHIN GROUP (ORDER BY "hbos_anomaly_score") as hbos_anomaly_score_critical_threshold,
            PERCENTILE_CONT({high_risk_percentile}/100.0) WITHIN GROUP (ORDER BY "pca_isolation_forest_score") as pca_isolation_forest_score_high_risk_threshold,
            PERCENTILE_CONT({critical_percentile}/100.0) WITHIN GROUP (ORDER BY "pca_isolation_forest_score") as pca_isolation_forest_score_critical_threshold,
            PERCENTILE_CONT({high_risk_percentile}/100.0) WITHIN GROUP (ORDER BY "hbos_pca_isolation_forest_score") as hbos_pca_isolation_forest_score_high_risk_threshold,
            PERCENTILE_CONT({critical_percentile}/100.0) WITHIN GROUP (ORDER BY "hbos_pca_isolation_forest_score") as hbos_pca_isolation_forest_score_critical_threshold
        FROM transactions
        WHERE "{score_column}" IS NOT NULL
    )
    SELECT 
        SUM(CASE WHEN "{score_column}" >= (SELECT critical_threshold FROM score_stats) THEN 1 ELSE 0 END) as critical_count,
        SUM(CASE WHEN "{score_column}" >= (SELECT high_risk_threshold FROM score_stats) AND "{score_column}" < (SELECT critical_threshold FROM score_stats) THEN 1 ELSE 0 END) as high_risk_count,
        SUM(CASE WHEN "{score_column}" < (SELECT high_risk_threshold FROM score_stats) THEN 1 ELSE 0 END) as normal_count,
        COUNT(*) as total_count,
        (SELECT high_risk_threshold FROM score_stats) as high_risk_threshold,
        (SELECT critical_threshold FROM score_stats) as critical_threshold,
        (SELECT hbos_anomaly_score_high_risk_threshold FROM score_stats) as hbos_anomaly_score_high_risk_threshold,
        (SELECT hbos_anomaly_score_critical_threshold FROM score_stats) as hbos_anomaly_score_critical_threshold,
        (SELECT pca_isolation_forest_score_high_risk_threshold FROM score_stats) as pca_isolation_forest_score_high_risk_threshold,
        (SELECT pca_isolation_forest_score_critical_threshold FROM score_stats) as pca_isolation_forest_score_critical_threshold,
        (SELECT hbos_pca_isolation_forest_score_high_risk_threshold FROM score_stats) as hbos_pca_isolation_forest_score_high_risk_threshold,
        (SELECT hbos_pca_isolation_forest_score_critical_threshold FROM score_stats) as hbos_pca_isolation_forest_score_critical_threshold
    FROM transactions
    WHERE "{score_column}" IS NOT NULL
    """

    result = execute_query(query, use_host=True)
    if result is not None and len(result) > 0:
        row = result.iloc[0]
        total = int(row['total_count'])
        critical_count = int(row['critical_count'])
        high_risk_count = int(row['high_risk_count'])
        normal_count = int(row['normal_count'])
        
        return {
            'critical': critical_count,
            'high_risk': high_risk_count,
            'normal': normal_count,
            'critical_pct': round((critical_count / total) * 100, 2) if total > 0 else 0.0,
            'high_risk_pct': round((high_risk_count / total) * 100, 2) if total > 0 else 0.0,
            'normal_pct': round((normal_count / total) * 100, 2) if total > 0 else 0.0,
            'high_risk_threshold': row['high_risk_threshold'],
            'critical_threshold': row['critical_threshold'],
            'hbos_anomaly_score_high_risk_threshold': row['hbos_anomaly_score_high_risk_threshold'],
            'hbos_anomaly_score_critical_threshold': row['hbos_anomaly_score_critical_threshold'],
            'pca_isolation_forest_score_high_risk_threshold': row['pca_isolation_forest_score_high_risk_threshold'],
            'pca_isolation_forest_score_critical_threshold': row['pca_isolation_forest_score_critical_threshold'],
            'hbos_pca_isolation_forest_score_high_risk_threshold': row['hbos_pca_isolation_forest_score_high_risk_threshold'],
            'hbos_pca_isolation_forest_score_critical_threshold': row['hbos_pca_isolation_forest_score_critical_threshold'],
        }
    return {'critical': 0, 'high_risk': 0, 'normal': 0, 'critical_pct': 0.0, 'high_risk_pct': 0.0, 'normal_pct': 0.0, 'high_risk_threshold': 0.0, 'critical_threshold': 0.0}


def get_decile_thresholds(score_column: str, use_host: bool = False, table_name: str = "transactions") -> list:
    """
    Get decile thresholds from the full dataset for consistent ranking
    Args:
        score_column: Name of the score column to analyze
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
    Returns:
        List of 11 threshold values [0th, 10th, 20th, ..., 100th percentile]
    """
    query = f"""
    SELECT 
        PERCENTILE_CONT(0.0) WITHIN GROUP (ORDER BY "{score_column}") as p0,
        PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY "{score_column}") as p10,
        PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY "{score_column}") as p20,
        PERCENTILE_CONT(0.3) WITHIN GROUP (ORDER BY "{score_column}") as p30,
        PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY "{score_column}") as p40,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{score_column}") as p50,
        PERCENTILE_CONT(0.6) WITHIN GROUP (ORDER BY "{score_column}") as p60,
        PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY "{score_column}") as p70,
        PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY "{score_column}") as p80,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY "{score_column}") as p90,
        PERCENTILE_CONT(1.0) WITHIN GROUP (ORDER BY "{score_column}") as p100
    FROM {table_name}
    WHERE "{score_column}" IS NOT NULL
    """
    
    result = execute_query(query, use_host=use_host)
    if result is not None and len(result) > 0:
        row = result.iloc[0]
        return [row[f'p{i}'] for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    return []


def get_transaction_per_page(score_column: str, threshold: float, item_per_page: int = 50,
                              page: int = 1, use_host: bool = True) -> dict:
    """
    Get filtered transactions above percentile threshold, ordered by score descending
    Args:
        score_column: Name of the score column to filter by (e.g., 'hbos_anomaly_score')
        percentile: Percentile threshold (e.g., 90.0 for 90th percentile)
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
        item_per_page: Maximum number of rows to return (default: 50)
        page: Page number for pagination (default: 1)
    Returns:
        Dictionary with 'data' containing list of transaction records and 'total' containing total count
    """
    # Include the score_column in the output if not already present
    columns = SUMMARY_TABLE_COLUMNS.copy()
    if score_column not in columns:
        columns.append(score_column)
    columns_str = ', '.join(f'"{col}"' for col in columns)
    offset = (page - 1) * item_per_page

    # First, get the total count
    count_query = f"""
    SELECT COUNT(*) as total_count
    FROM transactions
    WHERE "{score_column}" >= {threshold}
    """
    
    count_result = execute_query(count_query, use_host=use_host)
    total_count = int(count_result.iloc[0]['total_count']) if count_result is not None and len(count_result) > 0 else 0

    # Then get the paginated data
    query = f"""
    SELECT {columns_str}
    FROM transactions
    WHERE "{score_column}" >= {threshold}
    ORDER BY "{score_column}" DESC
    LIMIT {item_per_page} OFFSET {offset}
    """

    result = execute_query(query, use_host=use_host)

    if result is not None and len(result) > 0:

        # Apply memory-efficient dtypes
        for col in result.columns:
            if col == "CURRENCY_AMOUNT":
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
            elif col == "DATE_KEY":
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif col in ["BENECOUNTRY_HR", "BYORDERCOUNTRY_HR", "Check_flag"]:
                result[col] = result[col].astype("category")
            elif col in ["hbos_anomaly_score", "pca_isolation_forest_score", "hbos_pca_isolation_forest_score", "rule_base_risk_score"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")


        # This ensures consistent decile ranks across all transactions
        hbos_pca_if_col = 'hbos_pca_isolation_forest_score'
        if hbos_pca_if_col in result:
            # Get global decile thresholds for consistent ranking across all pages
            global_thresholds = get_decile_thresholds(hbos_pca_if_col, use_host=use_host)
            
            if global_thresholds and len(global_thresholds) == 11:
                # Apply global thresholds to assign decile ranks
                score_values = pd.to_numeric(result[hbos_pca_if_col], errors='coerce')
                decile_ranks = pd.Series(index=result.index, dtype=int)
                
                # DR1: Highest risk (90th-100th percentile) - scores >= 90th percentile
                decile_ranks[score_values >= global_thresholds[9]] = 1
                # DR2: (80th-90th percentile)
                decile_ranks[(score_values >= global_thresholds[8]) & (score_values < global_thresholds[9])] = 2
                # DR3: (70th-80th percentile)
                decile_ranks[(score_values >= global_thresholds[7]) & (score_values < global_thresholds[8])] = 3
                # DR4: (60th-70th percentile)
                decile_ranks[(score_values >= global_thresholds[6]) & (score_values < global_thresholds[7])] = 4
                # DR5: (50th-60th percentile)
                decile_ranks[(score_values >= global_thresholds[5]) & (score_values < global_thresholds[6])] = 5
                # DR6: (40th-50th percentile)
                decile_ranks[(score_values >= global_thresholds[4]) & (score_values < global_thresholds[5])] = 6
                # DR7: (30th-40th percentile)
                decile_ranks[(score_values >= global_thresholds[3]) & (score_values < global_thresholds[4])] = 7
                # DR8: (20th-30th percentile)
                decile_ranks[(score_values >= global_thresholds[2]) & (score_values < global_thresholds[3])] = 8
                # DR9: (10th-20th percentile)
                decile_ranks[(score_values >= global_thresholds[1]) & (score_values < global_thresholds[2])] = 9
                # DR10: Lowest risk (0th-10th percentile) - scores < 10th percentile
                decile_ranks[score_values < global_thresholds[1]] = 10
                
                # Fill NaN values with 10 (lowest risk)
                decile_ranks = decile_ranks.fillna(10).astype(int)
                # Convert to DR1, DR2, ..., DR10 format
                result['hbos_pca_if_decile_rank'] = 'DR' + decile_ranks.astype(str)
            else:
                result['hbos_pca_if_decile_rank'] = 'DR10'

        return {
            'data': result.to_dict('records'),
            'total': total_count
        }

    return {
        'data': [],
        'total': total_count
    }


def get_similar_transactions(index_key: str, use_host: bool = False) -> pd.DataFrame:
    """
    Get similar transactions based on the same byorder_to_bene relationship, excluding the current transaction
    Args:
        transaction_key: The TRANSACTION_KEY of the reference transaction
        use_host: If True, use localhost with host port (for connections from host machine)
        table_name: Name of the table to query (default: "transactions")
        limit: Maximum number of similar transactions to return
    Returns:
        pandas DataFrame with similar transactions
    """

    
    # Get all transactions with the same byorder_to_bene, excluding the current transaction
    columns_str = ', '.join(f'"{col}"' for col in SELECTED_COLUMNS)
    query = f'SELECT {columns_str} FROM transactions WHERE byorder_to_bene = \'{index_key}\' ORDER BY "DATE_KEY" DESC;'
    
    return execute_query(query, use_host=use_host)


# New functions for job matching with embeddings

def generate_openai_embedding(text: str) -> Optional[list]:
    """Generate OpenAI embedding for text (1536 dimensions)"""
    if not OPENAI_AVAILABLE:
        return None
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    if not client.api_key:
        return None
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def insert_job_with_embedding(job_id: str, job_text: str, company: str = None, title: str = None, embedding: list = None):
    """Insert job with vector embedding into database"""
    if embedding is None:
        embedding = generate_openai_embedding(job_text)
        if embedding is None:
            return False
    
    engine = create_db_engine()
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            # Assuming table name 'jobs' with columns: id, text, company, title, embedding
            query = text("""
                INSERT INTO jobs (id, text, company, title, embedding) 
                VALUES (:id, :text, :company, :title, :embedding)
                ON CONFLICT (id) DO UPDATE SET 
                    text = EXCLUDED.text,
                    company = EXCLUDED.company,
                    title = EXCLUDED.title,
                    embedding = EXCLUDED.embedding
            """)
            conn.execute(query, {
                'id': job_id,
                'text': job_text,
                'company': company,
                'title': title,
                'embedding': embedding
            })
            conn.commit()
        return True
    except Exception as e:
        print(f"Error inserting job: {e}")
        return False

def find_similar_jobs(resume_embedding: list, top_k: int = 10) -> list:
    """Find top-k similar jobs using cosine similarity"""
    engine = create_db_engine()
    if engine is None:
        return []
    
    try:
        with engine.connect() as conn:
            # Use pgvector for cosine similarity
            query = text("""
                SELECT id, text, company, title, 
                       1 - (embedding <=> :embedding) as similarity
                FROM jobs
                ORDER BY embedding <=> :embedding
                LIMIT :top_k
            """)
            result = conn.execute(query, {
                'embedding': resume_embedding,
                'top_k': top_k
            })
            
            jobs = []
            for row in result:
                jobs.append({
                    'id': row[0],
                    'text': row[1],
                    'company': row[2],
                    'title': row[3],
                    'similarity': float(row[4])
                })
            return jobs
    except Exception as e:
        print(f"Error finding similar jobs: {e}")
        return []

def batch_insert_jobs(jobs_df: pd.DataFrame):
    """Batch insert jobs with embeddings"""
    success_count = 0
    for _, row in jobs_df.iterrows():
        job_id = str(row.get('id', row.name))
        job_text = row.get('job_text_cleaned', row.get('Description', ''))
        company = row.get('Company', None)
        title = row.get('Job Title', None)
        
        if insert_job_with_embedding(job_id, job_text, company, title):
            success_count += 1
    
    return success_count

def setup_jobs_table():
    """Create jobs table with vector extension if it doesn't exist"""
    engine = create_db_engine()
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create jobs table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    company TEXT,
                    text TEXT,
                    embedding vector(768),  -- SBERT dimension
                    word2vec_embedding vector(300),  -- Word2Vec dimension
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for vector search
                CREATE INDEX IF NOT EXISTS jobs_embedding_idx ON jobs USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS jobs_word2vec_embedding_idx ON jobs USING ivfflat (word2vec_embedding vector_cosine_ops);
            """))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error setting up jobs table: {e}")
        return False

def insert_job_with_multiple_embeddings(job_id: str, job_text: str, company: str = None, title: str = None, 
                                       sbert_embedding: list = None, word2vec_embedding: list = None):
    """Insert job with multiple vector embeddings into database"""
    engine = create_db_engine()
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            query = text("""
                INSERT INTO jobs (id, title, company, text, embedding, word2vec_embedding) 
                VALUES (:id, :title, :company, :text, :embedding, :word2vec_embedding)
                ON CONFLICT (id) DO UPDATE SET 
                    title = EXCLUDED.title,
                    company = EXCLUDED.company,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    word2vec_embedding = EXCLUDED.word2vec_embedding
            """)
            conn.execute(query, {
                'id': job_id,
                'title': title,
                'company': company,
                'text': job_text,
                'embedding': sbert_embedding,
                'word2vec_embedding': word2vec_embedding
            })
            conn.commit()
        return True
    except Exception as e:
        print(f"Error inserting job: {e}")
        return False

def batch_insert_jobs_with_embeddings(jobs_data: list):
    """Batch insert multiple jobs with embeddings into database"""
    engine = create_db_engine()
    if engine is None:
        return False

    try:
        with engine.connect() as conn:
            for job in jobs_data:
                # Build dynamic query based on which embeddings are available
                columns = ['id', 'title', 'company', 'text']
                values = [job['id'], job['title'], job['company'], job['text']]
                placeholders = [':id', ':title', ':company', ':text']

                update_parts = [
                    'title = EXCLUDED.title',
                    'company = EXCLUDED.company',
                    'text = EXCLUDED.text'
                ]

                if job.get('embedding') is not None:
                    columns.append('embedding')
                    values.append('[' + ','.join(map(str, job['embedding'])) + ']')
                    placeholders.append(':embedding')
                    update_parts.append('embedding = EXCLUDED.embedding')

                if job.get('word2vec_embedding') is not None:
                    columns.append('word2vec_embedding')
                    values.append('[' + ','.join(map(str, job['word2vec_embedding'])) + ']')
                    placeholders.append(':word2vec_embedding')
                    update_parts.append('word2vec_embedding = EXCLUDED.word2vec_embedding')

                columns_str = ', '.join(columns)
                placeholders_str = ', '.join(placeholders)
                update_str = ', '.join(update_parts)

                query = text(f"""
                    INSERT INTO jobs ({columns_str})
                    VALUES ({placeholders_str})
                    ON CONFLICT (id) DO UPDATE SET {update_str}
                """)

                # Create parameter dict
                params = dict(zip([col for col in columns], values))

                conn.execute(query, params)

            conn.commit()
        return True
    except Exception as e:
        print(f"Error batch inserting jobs: {e}")
        return False

def find_similar_jobs_vector(resume_embedding: list, embedding_type: str = 'sbert', top_k: int = 10) -> list:
    """Find top-k similar jobs using vector similarity search"""
    engine = create_db_engine()
    if engine is None:
        return []
    
    try:
        with engine.connect() as conn:
            # Choose embedding column based on type
            embedding_column = 'embedding' if embedding_type == 'sbert' else 'word2vec_embedding'
            
            # Convert embedding to string format for pgvector
            embedding_str = '[' + ','.join(map(str, resume_embedding)) + ']'
            
            # Use pgvector for cosine similarity
            query = text(f"""
                SELECT id, title, company, text, 
                       1 - ({embedding_column} <=> :embedding) as similarity
                FROM jobs
                WHERE {embedding_column} IS NOT NULL
                ORDER BY {embedding_column} <=> :embedding
                LIMIT :top_k
            """)
            result = conn.execute(query, {
                'embedding': embedding_str,
                'top_k': top_k
            })
            
            jobs = []
            for row in result:
                jobs.append({
                    'id': row[0],
                    'title': row[1],
                    'company': row[2],
                    'text': row[3],
                    'similarity': float(row[4])
                })
            return jobs
    except Exception as e:
        print(f"Error finding similar jobs: {e}")
        return []

def populate_job_embeddings(compute_embeddings: bool = True):
    """Populate database with job embeddings from combined_data.json

    Args:
        compute_embeddings: If True, compute and store embeddings. If False, just import job data.
    """
    # Setup table
    if not setup_jobs_table():
        print("Failed to setup jobs table")
        return False

    # Load job data - use workspace path from session state or default
    try:
        import streamlit as st
        workspace_path = st.session_state.get('workspace_path')
    except:
        workspace_path = None

    if not workspace_path:
        # Fallback: assume workspace is at project root
        workspace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")

    json_path = os.path.join(workspace_path, "Data", "combined_data.json")

    if not os.path.exists(json_path):
        print(f"Job data not found at {json_path}")
        return False

    with open(json_path, 'r') as f:
        job_data = json.load(f)
    jobs_df = pd.DataFrame(job_data)

    # Filter valid jobs
    valid_jobs = jobs_df[jobs_df['Description'].notna() & (jobs_df['Description'] != '')].copy()
    job_texts = valid_jobs['Description'].tolist()

    print(f"Processing {len(valid_jobs)} jobs...")

    # Initialize embeddings
    sbert_embeddings = None
    w2v_embeddings = None

    if compute_embeddings:
        try:
            # Import here to avoid circular imports
            from functions.nlp_models import (
                load_sbert_model, compute_job_embeddings_sbert,
                load_trained_word2vec_model, compute_job_embeddings_w2v,
                simple_tokenize
            )
        except ImportError:
            print("Could not import NLP models - falling back to data-only import")
            compute_embeddings = False

    if compute_embeddings:
        # Compute SBERT embeddings in batches for better memory management
        print("Computing SBERT embeddings... (this may take 30-60 minutes for 30k jobs)")
        sbert_model = load_sbert_model()
        if sbert_model:
            # Process in smaller batches to avoid memory issues
            batch_size = 1000
            sbert_embeddings = []
            for i in range(0, len(job_texts), batch_size):
                batch_texts = job_texts[i:i + batch_size]
                batch_embeddings = compute_job_embeddings_sbert(batch_texts, sbert_model)
                sbert_embeddings.extend(batch_embeddings)
                print(f"Processed SBERT batch {i//batch_size + 1}/{(len(job_texts) + batch_size - 1)//batch_size}")
            print(f"SBERT embeddings computed for {len(job_texts)} jobs")

        # Compute Word2Vec embeddings
        print("Computing Word2Vec embeddings... (this should be faster)")
        w2v_model = load_trained_word2vec_model()
        if w2v_model:
            w2v_embeddings = compute_job_embeddings_w2v(job_texts, w2v_model)
            print(f"Word2Vec embeddings computed for {len(job_texts)} jobs")
    else:
        print("Skipping embedding computation - importing job data only")

    # Prepare batch data
    batch_data = []
    for idx, (_, row) in enumerate(valid_jobs.iterrows()):
        job_id = str(row.get('id', idx))
        job_text = row.get('Description', '')
        company = row.get('Company', None)
        title = row.get('Job Title', None)

        sbert_emb = sbert_embeddings[idx].tolist() if sbert_embeddings is not None and idx < len(sbert_embeddings) else None
        w2v_emb = w2v_embeddings[idx].tolist() if w2v_embeddings is not None and idx < len(w2v_embeddings) else None

        batch_data.append({
            'id': job_id,
            'title': title,
            'company': company,
            'text': job_text,
            'embedding': sbert_emb,
            'word2vec_embedding': w2v_emb
        })

    # Batch insert in chunks of 10,000 for better progress tracking
    chunk_size = 10000
    success_count = 0
    total_jobs = len(batch_data)

    for i in range(0, total_jobs, chunk_size):
        chunk = batch_data[i:i + chunk_size]
        if batch_insert_jobs_with_embeddings(chunk):
            success_count += len(chunk)
            progress = (success_count / total_jobs) * 100
            print(f"Inserted {success_count}/{total_jobs} jobs ({progress:.1f}%)...")
        else:
            print(f"Failed to insert chunk starting at {i}")
            return False

    print(f"Successfully inserted {success_count}/{total_jobs} jobs")
    if compute_embeddings:
        print("Embeddings computed and stored for vector search")
    else:
        print("Job data imported - embeddings can be computed later for faster searches")
    return success_count > 0


def update_job_embeddings_for_missing():
    """Update embeddings for jobs that don't have them yet"""
    try:
        from functions.nlp_models import (
            load_sbert_model, compute_job_embeddings_sbert,
            load_trained_word2vec_model, compute_job_embeddings_w2v
        )
    except ImportError:
        print("Could not import NLP models")
        return False

    engine = create_db_engine()
    if engine is None:
        return False

    try:
        with engine.connect() as conn:
            # Get jobs without embeddings
            result = conn.execute(text("""
                SELECT id, text FROM jobs
                WHERE embedding IS NULL OR word2vec_embedding IS NULL
                LIMIT 10000
            """))
            jobs_without_embeddings = result.fetchall()

            if not jobs_without_embeddings:
                print("All jobs already have embeddings")
                return True

            print(f"Found {len(jobs_without_embeddings)} jobs without embeddings")

            job_ids = [row[0] for row in jobs_without_embeddings]
            job_texts = [row[1] for row in jobs_without_embeddings]

            # Compute SBERT embeddings
            sbert_embeddings = None
            sbert_model = load_sbert_model()
            if sbert_model:
                print("Computing SBERT embeddings for missing jobs...")
                sbert_embeddings = compute_job_embeddings_sbert(job_texts, sbert_model)

            # Compute Word2Vec embeddings
            w2v_embeddings = None
            w2v_model = load_trained_word2vec_model()
            if w2v_model:
                print("Computing Word2Vec embeddings for missing jobs...")
                w2v_embeddings = compute_job_embeddings_w2v(job_texts, w2v_model)

            # Update jobs in batches
            batch_size = 1000
            success_count = 0

            for i in range(0, len(job_ids), batch_size):
                batch_ids = job_ids[i:i + batch_size]
                batch_sbert = sbert_embeddings[i:i + batch_size] if sbert_embeddings else None
                batch_w2v = w2v_embeddings[i:i + batch_size] if w2v_embeddings else None

                # Prepare update data
                for j, job_id in enumerate(batch_ids):
                    update_data = {}
                    if batch_sbert is not None:
                        update_data['embedding'] = '[' + ','.join(map(str, batch_sbert[j])) + ']'
                    if batch_w2v is not None:
                        update_data['word2vec_embedding'] = '[' + ','.join(map(str, batch_w2v[j])) + ']'

                    if update_data:
                        # Build dynamic update query
                        set_parts = []
                        params = {'job_id': job_id}

                        if 'embedding' in update_data:
                            set_parts.append('embedding = :embedding')
                            params['embedding'] = update_data['embedding']
                        if 'word2vec_embedding' in update_data:
                            set_parts.append('word2vec_embedding = :word2vec_embedding')
                            params['word2vec_embedding'] = update_data['word2vec_embedding']

                        set_clause = ', '.join(set_parts)
                        query = text(f"UPDATE jobs SET {set_clause} WHERE id = :job_id")

                        try:
                            conn.execute(query, params)
                            success_count += 1
                        except Exception as e:
                            print(f"Error updating job {job_id}: {e}")

                print(f"Updated batch {i//batch_size + 1}, total updated: {success_count}/{len(job_ids)}")

            conn.commit()
            print(f"Successfully updated embeddings for {success_count} jobs")
            return True

    except Exception as e:
        print(f"Error updating job embeddings: {e}")
        return False

def populate_job_embeddings_from_df(jobs_df: pd.DataFrame):
    """Populate database with job embeddings from a DataFrame"""
    try:
        # Import here to avoid circular imports
        from functions.nlp_models import (
            load_sbert_model, compute_job_embeddings_sbert,
            load_trained_word2vec_model, compute_job_embeddings_w2v,
            simple_tokenize
        )
    except ImportError:
        print("Could not import NLP models")
        return False
    
    # Setup table
    if not setup_jobs_table():
        print("Failed to setup jobs table")
        return False
    
    # Filter valid jobs - use job_text_cleaned if available, else Description
    text_col = 'job_text_cleaned' if 'job_text_cleaned' in jobs_df.columns else 'Description'
    valid_jobs = jobs_df[jobs_df[text_col].notna() & (jobs_df[text_col] != '')].copy()
    job_texts = valid_jobs[text_col].tolist()
    
    print(f"Processing {len(valid_jobs)} jobs from DataFrame...")
    
    # Compute SBERT embeddings
    print("Computing SBERT embeddings...")
    sbert_model = load_sbert_model()
    sbert_embeddings = None
    if sbert_model:
        sbert_embeddings = compute_job_embeddings_sbert(job_texts, sbert_model)
    
    # Compute Word2Vec embeddings
    print("Computing Word2Vec embeddings...")
    w2v_model = load_trained_word2vec_model()
    w2v_embeddings = None
    if w2v_model:
        w2v_embeddings = compute_job_embeddings_w2v(job_texts, w2v_model)
    
    # Prepare batch data
    batch_data = []
    for idx, (_, row) in enumerate(valid_jobs.iterrows()):
        job_id = str(row.get('job_id', row.get('id', idx)))
        job_text = row.get(text_col, '')
        company = row.get('Company', None)
        title = row.get('Job Title', None)
        
        sbert_emb = sbert_embeddings[idx].tolist() if sbert_embeddings is not None else None
        w2v_emb = w2v_embeddings[idx].tolist() if w2v_embeddings is not None else None
        
        batch_data.append({
            'id': job_id,
            'title': title,
            'company': company,
            'text': job_text,
            'embedding': sbert_emb,
            'word2vec_embedding': w2v_emb
        })
    
    # Batch insert in chunks of 50,000
    chunk_size = 50000
    success_count = 0
    for i in range(0, len(batch_data), chunk_size):
        chunk = batch_data[i:i + chunk_size]
        if batch_insert_jobs_with_embeddings(chunk):
            success_count += len(chunk)
            print(f"Inserted {success_count}/{len(valid_jobs)} jobs...")
        else:
            print(f"Failed to insert chunk starting at {i}")
            return False
    
    print(f"Successfully inserted {success_count}/{len(valid_jobs)} jobs with embeddings")
    return success_count > 0