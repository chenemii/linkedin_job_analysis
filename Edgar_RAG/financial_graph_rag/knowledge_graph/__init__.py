"""
Knowledge Graph Module

Contains entity definitions, extraction logic, and graph database management
for the Financial Graph RAG system.
"""

from .entities import (
    BaseEntity, Relationship, EntityType, RelationshipType,
    Company, Person, Transaction, Subsidiary, Division, Location,
    FinancialMetric, create_entity
)
from .extractor import EntityRelationshipExtractor
from .graph_db import GraphDatabaseManager

__all__ = [
    "BaseEntity", "Relationship", "EntityType", "RelationshipType",
    "Company", "Person", "Transaction", "Subsidiary", "Division", 
    "Location", "FinancialMetric", "create_entity",
    "EntityRelationshipExtractor",
    "GraphDatabaseManager"
] 