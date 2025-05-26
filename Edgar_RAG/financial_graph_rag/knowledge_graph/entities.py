"""
Entity definitions for Financial Knowledge Graph

Defines the core entities and their schemas for the knowledge graph
focused on M&A analysis and organizational structure changes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class EntityType(Enum):
    """Types of entities in the financial knowledge graph"""
    COMPANY = "Company"
    PERSON = "Person" 
    TRANSACTION = "Transaction"
    SUBSIDIARY = "Subsidiary"
    DIVISION = "Division"
    BUSINESS_UNIT = "BusinessUnit"
    PRODUCT = "Product"
    LOCATION = "Location"
    REGULATORY_BODY = "RegulatoryBody"
    FINANCIAL_METRIC = "FinancialMetric"
    LEGAL_ENTITY = "LegalEntity"

class RelationshipType(Enum):
    """Types of relationships in the financial knowledge graph"""
    # Ownership relationships
    OWNS = "owns"
    OWNED_BY = "owned_by"
    SUBSIDIARY_OF = "subsidiary_of"
    PARENT_OF = "parent_of"
    
    # M&A relationships  
    ACQUIRED = "acquired"
    ACQUIRED_BY = "acquired_by"
    MERGED_WITH = "merged_with"
    DIVESTED = "divested"
    SPUN_OFF = "spun_off"
    
    # Personnel relationships
    CEO_OF = "ceo_of"
    EXECUTIVE_OF = "executive_of"
    EMPLOYEE_OF = "employee_of"
    BOARD_MEMBER_OF = "board_member_of"
    
    # Business relationships
    COMPETITOR_OF = "competitor_of"
    PARTNER_OF = "partner_of"
    SUPPLIER_OF = "supplier_of"
    CUSTOMER_OF = "customer_of"
    
    # Organizational relationships
    PART_OF = "part_of"
    CONTAINS = "contains"
    REPORTS_TO = "reports_to"
    MANAGES = "manages"
    
    # Location relationships
    LOCATED_IN = "located_in"
    HEADQUARTERED_IN = "headquartered_in"
    
    # Regulatory relationships
    REGULATED_BY = "regulated_by"
    REGULATES = "regulates"
    
    # Financial relationships
    HAS_METRIC = "has_metric"
    IMPACTS = "impacts"

@dataclass
class BaseEntity:
    """Base entity class with common attributes"""
    id: str
    name: str
    entity_type: EntityType
    confidence_score: float = 0.0
    source_document: Optional[str] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass 
class Company(BaseEntity):
    """Company entity with financial and organizational attributes"""
    ticker: Optional[str] = None
    cik: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    revenue: Optional[float] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.COMPANY

@dataclass
class Person(BaseEntity):
    """Person entity representing executives and key personnel"""
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    biography: Optional[str] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.PERSON

@dataclass
class Transaction(BaseEntity):
    """Transaction entity representing M&A deals and major business transactions"""
    transaction_type: Optional[str] = None  # "acquisition", "merger", "divestiture", etc.
    acquirer: Optional[str] = None
    target: Optional[str] = None
    announcement_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    transaction_value: Optional[float] = None
    currency: Optional[str] = None
    status: Optional[str] = None  # "announced", "completed", "cancelled", etc.
    rationale: Optional[str] = None
    synergies: Optional[str] = None
    regulatory_approval: Optional[bool] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.TRANSACTION

@dataclass
class Subsidiary(BaseEntity):
    """Subsidiary entity representing subsidiary companies"""
    parent_company: Optional[str] = None
    ownership_percentage: Optional[float] = None
    acquisition_date: Optional[datetime] = None
    business_type: Optional[str] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.SUBSIDIARY

@dataclass
class Division(BaseEntity):
    """Division entity representing business divisions within companies"""
    parent_company: Optional[str] = None
    business_focus: Optional[str] = None
    revenue: Optional[float] = None
    employees: Optional[int] = None
    head_executive: Optional[str] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.DIVISION

@dataclass
class BusinessUnit(BaseEntity):
    """Business unit entity representing operational units"""
    parent_entity: Optional[str] = None
    unit_type: Optional[str] = None
    products_services: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.entity_type = EntityType.BUSINESS_UNIT

@dataclass
class Location(BaseEntity):
    """Location entity representing geographical locations"""
    country: Optional[str] = None
    state_province: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    location_type: Optional[str] = None  # "headquarters", "office", "facility", etc.
    
    def __post_init__(self):
        self.entity_type = EntityType.LOCATION

@dataclass
class FinancialMetric(BaseEntity):
    """Financial metric entity representing financial data points"""
    metric_type: Optional[str] = None  # "revenue", "profit", "market_cap", etc.
    value: Optional[float] = None
    currency: Optional[str] = None
    period: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None
    
    def __post_init__(self):
        self.entity_type = EntityType.FINANCIAL_METRIC

@dataclass
class Relationship:
    """Relationship between entities in the knowledge graph"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    confidence_score: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_document: Optional[str] = None
    extracted_text: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

# Entity type mapping for factory pattern
ENTITY_TYPE_MAP = {
    EntityType.COMPANY: Company,
    EntityType.PERSON: Person,
    EntityType.TRANSACTION: Transaction,
    EntityType.SUBSIDIARY: Subsidiary,
    EntityType.DIVISION: Division,
    EntityType.BUSINESS_UNIT: BusinessUnit,
    EntityType.LOCATION: Location,
    EntityType.FINANCIAL_METRIC: FinancialMetric,
}

def create_entity(entity_type: EntityType, **kwargs) -> BaseEntity:
    """
    Factory function to create entities of the specified type
    
    Args:
        entity_type: Type of entity to create
        **kwargs: Entity attributes
        
    Returns:
        Created entity instance
    """
    entity_class = ENTITY_TYPE_MAP.get(entity_type, BaseEntity)
    return entity_class(entity_type=entity_type, **kwargs) 