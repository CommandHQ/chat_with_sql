"""
Data Indexing and Optimization Module for Chat with SQL

This module provides comprehensive optimization features including:
- Query result caching with TTL
- Database index analysis and recommendations
- Query performance monitoring
- Connection pooling management
"""

import hashlib
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool


class QueryCache:
    """
    In-memory cache for SQL query results with TTL support.
    Reduces database load by caching frequently accessed queries.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        """
        Initialize the query cache.

        Args:
            ttl_seconds: Time-to-live for cached entries (default: 5 minutes)
            max_size: Maximum number of cached queries (default: 100)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _generate_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key for a query."""
        query_normalized = re.sub(r'\s+', ' ', query.strip().lower())
        key_str = f"{query_normalized}:{str(params) if params else ''}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieve cached query result if available and not expired.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, params)

        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expires_at']:
                self.hits += 1
                entry['last_accessed'] = datetime.now()
                entry['access_count'] += 1
                return entry['result']
            else:
                # Expired entry
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, query: str, result: Any, params: Optional[Dict] = None) -> None:
        """
        Store query result in cache.

        Args:
            query: SQL query string
            result: Query result to cache
            params: Query parameters
        """
        key = self._generate_key(query, params)

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_oldest()

        self.cache[key] = {
            'result': result,
            'query': query,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.ttl_seconds),
            'last_accessed': datetime.now(),
            'access_count': 1
        }

    def _evict_oldest(self) -> None:
        """Evict the least recently accessed entry."""
        if not self.cache:
            return

        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_accessed']
        )
        del self.cache[oldest_key]

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }


class QueryPerformanceMonitor:
    """
    Monitor and analyze query performance.
    Tracks execution times and identifies slow queries.
    """

    def __init__(self, slow_query_threshold_ms: float = 1000.0):
        """
        Initialize the performance monitor.

        Args:
            slow_query_threshold_ms: Threshold for slow queries in milliseconds
        """
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'total_time_ms': 0.0,
                'min_time_ms': float('inf'),
                'max_time_ms': 0.0,
                'slow_count': 0
            }
        )
        self.recent_queries: List[Dict[str, Any]] = []
        self.max_recent = 50

    def _normalize_query(self, query: str) -> str:
        """Normalize query for grouping statistics."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        # Replace literal values with placeholders for grouping
        query = re.sub(r"'[^']*'", "'?'", query)
        query = re.sub(r'\b\d+\b', '?', query)
        return query

    def record_query(self, query: str, execution_time_ms: float,
                    result_count: int = 0, cached: bool = False) -> None:
        """
        Record a query execution.

        Args:
            query: SQL query string
            execution_time_ms: Execution time in milliseconds
            result_count: Number of rows returned
            cached: Whether result was from cache
        """
        normalized = self._normalize_query(query)
        stats = self.query_stats[normalized]

        stats['count'] += 1
        stats['total_time_ms'] += execution_time_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], execution_time_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], execution_time_ms)
        stats['query_template'] = normalized

        if execution_time_ms > self.slow_query_threshold_ms:
            stats['slow_count'] += 1

        # Store recent query
        self.recent_queries.append({
            'query': query,
            'normalized': normalized,
            'execution_time_ms': execution_time_ms,
            'result_count': result_count,
            'cached': cached,
            'timestamp': datetime.now(),
            'is_slow': execution_time_ms > self.slow_query_threshold_ms
        })

        # Keep only recent queries
        if len(self.recent_queries) > self.max_recent:
            self.recent_queries.pop(0)

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest query patterns."""
        slow_queries = []

        for normalized, stats in self.query_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time_ms'] / stats['count']
                if avg_time > self.slow_query_threshold_ms or stats['slow_count'] > 0:
                    slow_queries.append({
                        'query_template': normalized,
                        'avg_time_ms': avg_time,
                        'max_time_ms': stats['max_time_ms'],
                        'execution_count': stats['count'],
                        'slow_count': stats['slow_count']
                    })

        # Sort by average time descending
        slow_queries.sort(key=lambda x: x['avg_time_ms'], reverse=True)
        return slow_queries[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        total_queries = sum(s['count'] for s in self.query_stats.values())
        total_time = sum(s['total_time_ms'] for s in self.query_stats.values())
        total_slow = sum(s['slow_count'] for s in self.query_stats.values())

        avg_time = (total_time / total_queries) if total_queries > 0 else 0

        return {
            'total_queries': total_queries,
            'unique_query_patterns': len(self.query_stats),
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'slow_queries_count': total_slow,
            'slow_query_rate': f"{(total_slow / total_queries * 100):.2f}%" if total_queries > 0 else "0%"
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent queries."""
        return self.recent_queries[-limit:][::-1]


class IndexAnalyzer:
    """
    Analyze database queries and recommend indexes for optimization.
    """

    def __init__(self, engine: sqlalchemy.engine.Engine):
        """
        Initialize the index analyzer.

        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self.inspector = inspect(engine)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and suggest potential indexes.

        Args:
            query: SQL query to analyze

        Returns:
            Analysis results with index recommendations
        """
        recommendations = []
        query_upper = query.upper()

        # Extract table names
        tables = self._extract_tables(query)

        # Extract WHERE clause columns
        where_columns = self._extract_where_columns(query)

        # Extract JOIN columns
        join_columns = self._extract_join_columns(query)

        # Extract ORDER BY columns
        order_columns = self._extract_order_columns(query)

        # Get existing indexes
        existing_indexes = {}
        for table in tables:
            try:
                existing_indexes[table] = self._get_table_indexes(table)
            except Exception:
                existing_indexes[table] = []

        # Generate recommendations
        for table in tables:
            table_indexes = existing_indexes.get(table, [])
            indexed_columns = {col for idx in table_indexes for col in idx['columns']}

            # Recommend indexes for WHERE columns
            for col in where_columns.get(table, []):
                if col not in indexed_columns:
                    recommendations.append({
                        'type': 'WHERE_CLAUSE',
                        'table': table,
                        'columns': [col],
                        'reason': f'Column {col} used in WHERE clause but not indexed',
                        'priority': 'HIGH',
                        'suggested_index': f'idx_{table}_{col}'
                    })

            # Recommend indexes for JOIN columns
            for col in join_columns.get(table, []):
                if col not in indexed_columns:
                    recommendations.append({
                        'type': 'JOIN',
                        'table': table,
                        'columns': [col],
                        'reason': f'Column {col} used in JOIN but not indexed',
                        'priority': 'HIGH',
                        'suggested_index': f'idx_{table}_{col}_join'
                    })

            # Recommend indexes for ORDER BY columns
            for col in order_columns.get(table, []):
                if col not in indexed_columns:
                    recommendations.append({
                        'type': 'ORDER_BY',
                        'table': table,
                        'columns': [col],
                        'reason': f'Column {col} used in ORDER BY but not indexed',
                        'priority': 'MEDIUM',
                        'suggested_index': f'idx_{table}_{col}_order'
                    })

        return {
            'query': query,
            'tables_analyzed': list(tables),
            'existing_indexes': existing_indexes,
            'recommendations': recommendations,
            'recommendation_count': len(recommendations)
        }

    def _extract_tables(self, query: str) -> set:
        """Extract table names from query."""
        tables = set()

        # Match FROM clause
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.update(re.findall(from_pattern, query, re.IGNORECASE))

        # Match JOIN clauses
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.update(re.findall(join_pattern, query, re.IGNORECASE))

        return tables

    def _extract_where_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in WHERE clauses."""
        columns_by_table = defaultdict(list)

        # Simple pattern matching for WHERE conditions
        where_pattern = r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE | re.DOTALL)

        if where_match:
            where_clause = where_match.group(1)
            # Match table.column or column patterns
            col_pattern = r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]'
            for table, col in re.findall(col_pattern, where_clause):
                if table:
                    columns_by_table[table].append(col)
                else:
                    # If no table specified, add to all tables
                    tables = self._extract_tables(query)
                    for t in tables:
                        columns_by_table[t].append(col)

        return columns_by_table

    def _extract_join_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in JOIN conditions."""
        columns_by_table = defaultdict(list)

        # Match JOIN ... ON conditions
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+[a-zA-Z_][a-zA-Z0-9_]*\s+)?ON\s+([^WHERE|JOIN]+)'

        for table, condition in re.findall(join_pattern, query, re.IGNORECASE):
            # Extract columns from condition
            col_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
            for tbl, col in re.findall(col_pattern, condition):
                columns_by_table[tbl].append(col)

        return columns_by_table

    def _extract_order_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract columns used in ORDER BY clauses."""
        columns_by_table = defaultdict(list)

        order_pattern = r'ORDER BY\s+(.+?)(?:LIMIT|$)'
        order_match = re.search(order_pattern, query, re.IGNORECASE | re.DOTALL)

        if order_match:
            order_clause = order_match.group(1)
            # Match table.column or column patterns
            col_pattern = r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\.)?([a-zA-Z_][a-zA-Z0-9_]*)'
            for table, col in re.findall(col_pattern, order_clause):
                if table:
                    columns_by_table[table].append(col)
                else:
                    tables = self._extract_tables(query)
                    for t in tables:
                        columns_by_table[t].append(col)

        return columns_by_table

    def _get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get existing indexes for a table."""
        try:
            indexes = self.inspector.get_indexes(table_name)
            return [
                {
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx.get('unique', False)
                }
                for idx in indexes
            ]
        except Exception:
            return []

    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics about a table including row count and indexes."""
        try:
            with self.engine.connect() as conn:
                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()

                # Get table size
                size_query = text("""
                    SELECT
                        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb,
                        ROUND((data_length / 1024 / 1024), 2) AS data_size_mb,
                        ROUND((index_length / 1024 / 1024), 2) AS index_size_mb
                    FROM information_schema.TABLES
                    WHERE table_schema = DATABASE()
                    AND table_name = :table_name
                """)
                result = conn.execute(size_query, {"table_name": table_name})
                size_info = result.fetchone()

                indexes = self._get_table_indexes(table_name)

                return {
                    'table_name': table_name,
                    'row_count': row_count,
                    'size_mb': float(size_info[0]) if size_info else 0,
                    'data_size_mb': float(size_info[1]) if size_info else 0,
                    'index_size_mb': float(size_info[2]) if size_info else 0,
                    'indexes': indexes,
                    'index_count': len(indexes)
                }
        except Exception as e:
            return {
                'table_name': table_name,
                'error': str(e)
            }


class DataOptimizer:
    """
    Main data optimization coordinator.
    Integrates caching, performance monitoring, and index analysis.
    """

    def __init__(self, connection_string: str,
                 enable_cache: bool = True,
                 enable_monitoring: bool = True,
                 enable_pooling: bool = True,
                 cache_ttl: int = 300,
                 pool_size: int = 5):
        """
        Initialize the data optimizer.

        Args:
            connection_string: Database connection string
            enable_cache: Enable query result caching
            enable_monitoring: Enable performance monitoring
            enable_pooling: Enable connection pooling
            cache_ttl: Cache TTL in seconds
            pool_size: Connection pool size
        """
        self.connection_string = connection_string
        self.enable_cache = enable_cache
        self.enable_monitoring = enable_monitoring

        # Initialize components
        if enable_cache:
            self.cache = QueryCache(ttl_seconds=cache_ttl)
        else:
            self.cache = None

        if enable_monitoring:
            self.monitor = QueryPerformanceMonitor()
        else:
            self.monitor = None

        # Create engine with optional pooling
        if enable_pooling:
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                connect_args={"connect_timeout": 10}
            )
        else:
            self.engine = create_engine(
                connection_string,
                connect_args={"connect_timeout": 10}
            )

        self.index_analyzer = IndexAnalyzer(self.engine)

    def execute_query(self, query: str, params: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute a query with optimization features.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Tuple of (result, metadata)
        """
        start_time = time.time()
        cached = False
        result = None

        # Try cache first
        if self.enable_cache and self._is_cacheable(query):
            result = self.cache.get(query, params)
            if result is not None:
                cached = True

        # Execute query if not cached
        if result is None:
            with self.engine.connect() as conn:
                db_result = conn.execute(text(query), params or {})
                result = db_result.fetchall()

            # Cache the result
            if self.enable_cache and self._is_cacheable(query):
                self.cache.set(query, result, params)

        execution_time_ms = (time.time() - start_time) * 1000

        # Record performance
        if self.enable_monitoring:
            self.monitor.record_query(
                query=query,
                execution_time_ms=execution_time_ms,
                result_count=len(result) if result else 0,
                cached=cached
            )

        metadata = {
            'execution_time_ms': execution_time_ms,
            'cached': cached,
            'row_count': len(result) if result else 0
        }

        return result, metadata

    def _is_cacheable(self, query: str) -> bool:
        """Determine if a query should be cached."""
        query_upper = query.upper().strip()
        # Only cache SELECT queries
        if not query_upper.startswith('SELECT'):
            return False
        # Don't cache queries with non-deterministic functions
        non_deterministic = ['NOW()', 'CURRENT_TIMESTAMP', 'RAND()', 'UUID()']
        return not any(func in query_upper for func in non_deterministic)

    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query performance analysis.

        Args:
            query: SQL query to analyze

        Returns:
            Analysis report with recommendations
        """
        # Get index recommendations
        index_analysis = self.index_analyzer.analyze_query(query)

        # Get query plan (EXPLAIN)
        query_plan = self._get_query_plan(query)

        return {
            'query': query,
            'index_analysis': index_analysis,
            'query_plan': query_plan,
            'timestamp': datetime.now().isoformat()
        }

    def _get_query_plan(self, query: str) -> Optional[List[Dict]]:
        """Get MySQL EXPLAIN output for query."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"EXPLAIN {query}"))
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return {'error': str(e)}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_enabled': self.enable_cache,
            'monitoring_enabled': self.enable_monitoring
        }

        if self.enable_cache:
            report['cache_stats'] = self.cache.get_stats()

        if self.enable_monitoring:
            report['performance_stats'] = self.monitor.get_stats()
            report['slow_queries'] = self.monitor.get_slow_queries(limit=5)
            report['recent_queries'] = self.monitor.get_recent_queries(limit=10)

        return report

    def clear_cache(self) -> None:
        """Clear query cache."""
        if self.cache:
            self.cache.clear()

    def dispose(self) -> None:
        """Dispose of engine resources."""
        self.engine.dispose()
