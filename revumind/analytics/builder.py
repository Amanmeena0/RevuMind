"""
Analytics Builder Layer
======================
Implements pre-computation of analytical summary tables from raw reviews database.
Reduces Streamlit rendering latency by pre-aggregating heavy metrics.
"""

import logging
import sqlite3
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnalyticsBuilder:
    """
    Handles ETL process of converting raw review metrics into pre-computed summary tables.
    """

    def __init__(self, db_path: str = "revumind.db"):
        self.db_path = db_path
        self._setup_connection_pragmas()

    def _setup_connection_pragmas(self):
        """
        Pre-configures SQLite database pragmas for optimized analytical processing.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            cursor.execute("PRAGMA cache_size=-2000000;")  # Allocate ~2GB memory cache
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to set connection pragmas: {e}")
        finally:
            conn.close()

    def get_connection(self):
        """
        Returns a configured sqlite3 connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def rebuild_all_summaries(self):
        """
        Executes analytical queries to rebuild summary tables from raw data.
        """
        start_time = time.time()
        logger.info("Initializing rebuild of analytical summaries...")

        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # 1. Product Summary Rebuild
            logger.info("Rebuilding product_summary...")
            cursor.execute("DELETE FROM product_summary;")
            cursor.execute("""
                INSERT INTO product_summary (product_id, total_reviews, average_stars, average_helpfulness, positive_count, neutral_count, negative_count, last_updated)
                SELECT 
                    r.product_id,
                    COUNT(r.id) as total_reviews,
                    AVG(CAST(r.score AS FLOAT)) as average_stars,
                    AVG(COALESCE(r.predicted_helpfulness, 0.5)) as average_helpfulness,
                    SUM(CASE WHEN r.score >= 4 THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN r.score = 3 THEN 1 ELSE 0 END) as neutral_count,
                    SUM(CASE WHEN r.score <= 2 THEN 1 ELSE 0 END) as negative_count,
                    datetime('now') as last_updated
                FROM reviews r
                GROUP BY r.product_id;
            """)

            # 2. Monthly Sentiment Rebuild
            logger.info("Rebuilding monthly_sentiment...")
            cursor.execute("DELETE FROM monthly_sentiment;")
            cursor.execute("""
                INSERT INTO monthly_sentiment (month, product_id, total_reviews, positive_count, neutral_count, negative_count)
                SELECT 
                    strftime('%Y-%m', r.review_time) as month,
                    r.product_id,
                    COUNT(r.id) as total_reviews,
                    SUM(CASE WHEN r.score >= 4 THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN r.score = 3 THEN 1 ELSE 0 END) as neutral_count,
                    SUM(CASE WHEN r.score <= 2 THEN 1 ELSE 0 END) as negative_count
                FROM reviews r
                WHERE r.review_time IS NOT NULL
                GROUP BY month, r.product_id;
            """)

            # 3. Aspect Summary Rebuild
            logger.info("Rebuilding aspect_summary...")
            cursor.execute("DELETE FROM aspect_summary;")
            cursor.execute("""
                INSERT INTO aspect_summary (product_id, aspect_term, positive_count, neutral_count, negative_count, avg_confidence)
                SELECT 
                    r.product_id,
                    a.aspect_term,
                    SUM(CASE WHEN a.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN a.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                    SUM(CASE WHEN a.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                    AVG(a.confidence) as avg_confidence
                FROM reviews r
                JOIN aspect_sentiments a ON r.id = a.review_id
                GROUP BY r.product_id, a.aspect_term;
            """)

            # 4. Complaint Summary Rebuild
            logger.info("Rebuilding complaint_summary...")
            cursor.execute("DELETE FROM complaint_summary;")
            cursor.execute("""
                INSERT INTO complaint_summary (product_id, topic_name, complaint_count, severity_score)
                SELECT 
                    r.product_id,
                    COALESCE(t.name, 'General') as topic_name,
                    COUNT(r.id) as complaint_count,
                    (COUNT(r.id) * (5.0 - AVG(CAST(r.score AS FLOAT)))) as severity_score
                FROM reviews r
                LEFT JOIN topics t ON r.topic_id = t.topic_id
                WHERE r.score <= 2
                GROUP BY r.product_id, topic_name;
            """)

            conn.commit()
            duration = time.time() - start_time
            logger.info(f"Analytical Summary tables rebuild completed successfully in {duration:.2f} seconds.")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to rebuild analytical summaries: {e}")
            raise e
        finally:
            conn.close()


if __name__ == "__main__":
    builder = AnalyticsBuilder()
    builder.rebuild_all_summaries()
