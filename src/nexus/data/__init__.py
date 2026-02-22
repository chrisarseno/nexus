"""
Nexus Data Ingestion & Processing System

Multi-format data ingestion and automated processing pipeline.

This system provides:
- Multi-format data ingestion (text, structured, code, documents, web)
- Automated data processing pipeline
- Internet and web data retrieval
- HuggingFace dataset integration
"""

from .data_ingestion import DataIngestionProcessor
from .auto_data_processor import AutoDataProcessor
from .internet_retriever import InternetKnowledgeRetriever
from .huggingface_loader import HuggingFaceLoader

# Aliases for backward compatibility
DataIngestion = DataIngestionProcessor
InternetRetriever = InternetKnowledgeRetriever

__all__ = [
    "DataIngestion",
    "DataIngestionProcessor",
    "AutoDataProcessor",
    "InternetRetriever",
    "InternetKnowledgeRetriever",
    "HuggingFaceLoader",
]

__version__ = "1.0.0"
