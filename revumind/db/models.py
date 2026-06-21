"""
Database ORM Models
===================
Defines the database schema mapping reviews, embeddings, topics,
entities, and aspect sentiments to relational tables.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    String,
    Text,
)
from sqlalchemy.orm import relationship

# Local Base import
from revumind.core.database import Base

# Try to import pgvector for PostgreSQL deployment
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import-not-found]

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class Product(Base):
    """
    Metadata representation of a product.
    """

    __tablename__ = "products"

    id = Column(String(50), primary_key=True)
    brand = Column(String(100), nullable=True)
    category = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="product", cascade="all, delete-orphan")


class Review(Base):
    """
    Core review data, including original scores, preprocessed texts,
    topic IDs, and helpfulness metrics.
    """

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), index=True)
    user_id = Column(String(50), nullable=True)
    profile_name = Column(String(255), nullable=True)
    score = Column(Integer, CheckConstraint("score >= 1 AND score <= 5"))
    helpfulness_numerator = Column(Integer, default=0)
    helpfulness_denominator = Column(Integer, default=0)

    # Store continuous target and predicted helpfulness scores
    helpfulness_score = Column(Float, default=0.0)
    predicted_helpfulness = Column(Float, nullable=True)

    review_time = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    review_text = Column(Text, nullable=True)
    clean_review_text = Column(Text, nullable=False)
    topic_id = Column(Integer, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    product = relationship("Product", back_populates="reviews")
    embedding = relationship(
        "ReviewEmbedding", back_populates="review", uselist=False, cascade="all, delete-orphan"
    )
    entities = relationship("Entity", back_populates="review", cascade="all, delete-orphan")
    aspect_sentiments = relationship(
        "AspectSentiment", back_populates="review", cascade="all, delete-orphan"
    )


class ReviewEmbedding(Base):
    """
    Separate table to store high-dimensional embeddings.
    Isolating embeddings prevents performance degradation during standard tabular queries.
    """

    __tablename__ = "review_embeddings"

    review_id = Column(Integer, ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True)

    # Define vector column dynamically
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384), nullable=False)
    else:
        # Fallback representation for SQLite
        embedding = Column(PickleType, nullable=False)

    # Relationships
    review = relationship("Review", back_populates="embedding")


class Entity(Base):
    """
    Token-level named entities (Brands, Products, Features) extracted by spaCy.
    """

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    entity_text = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False)  # e.g. 'BRAND', 'PRODUCT', 'FEATURE'
    confidence = Column(Float, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)

    # Relationships
    review = relationship("Review", back_populates="entities")


class AspectSentiment(Base):
    """
    Aspect-opinion polarities (e.g. Battery -> Positive) extracted by DeBERTa ABSA.
    """

    __tablename__ = "aspect_sentiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey("reviews.id", ondelete="CASCADE"), index=True)
    aspect_term = Column(String(255), nullable=False)
    sentiment_label = Column(String(20), nullable=False)  # 'positive', 'negative', 'neutral'
    confidence = Column(Float, nullable=True)

    # Relationships
    review = relationship("Review", back_populates="aspect_sentiments")


class Topic(Base):
    """
    Topic mappings and keywords created by BERTopic.
    """

    __tablename__ = "topics"

    topic_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    keywords = Column(JSON, nullable=False)  # Array of strings / word-score objects
    created_at = Column(DateTime, default=datetime.utcnow)


class Summary(Base):
    """
    BART executive summaries generated for a product cohort.
    """

    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), index=True)
    cohort_type = Column(
        String(50), nullable=False
    )  # e.g. 'all', 'positive', 'negative', 'topic_3'
    cohort_value = Column(String(100), nullable=True)
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    product = relationship("Product", back_populates="summaries")


class ProductSummary(Base):
    """
    Materialized aggregate metrics for each product.
    """
    __tablename__ = "product_summary"

    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), primary_key=True)
    total_reviews = Column(Integer, nullable=False)
    average_stars = Column(Float)
    average_helpfulness = Column(Float)
    positive_count = Column(Integer)
    neutral_count = Column(Integer)
    negative_count = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product")


class MonthlySentiment(Base):
    """
    Materialized aggregated monthly sentiment volume per product.
    """
    __tablename__ = "monthly_sentiment"

    month = Column(String(7), primary_key=True)  # YYYY-MM
    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), primary_key=True)
    total_reviews = Column(Integer)
    positive_count = Column(Integer)
    neutral_count = Column(Integer)
    negative_count = Column(Integer)

    product = relationship("Product")


class AspectSummary(Base):
    """
    Materialized aggregated aspect term sentiment per product.
    """
    __tablename__ = "aspect_summary"

    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), primary_key=True)
    aspect_term = Column(String(255), primary_key=True)
    positive_count = Column(Integer)
    neutral_count = Column(Integer)
    negative_count = Column(Integer)
    avg_confidence = Column(Float)

    product = relationship("Product")


class ComplaintSummary(Base):
    """
    Materialized aggregated complaints (low ratings) by topic per product.
    """
    __tablename__ = "complaint_summary"

    product_id = Column(String(50), ForeignKey("products.id", ondelete="CASCADE"), primary_key=True)
    topic_name = Column(String(255), primary_key=True)
    complaint_count = Column(Integer)
    severity_score = Column(Float)

    product = relationship("Product")

