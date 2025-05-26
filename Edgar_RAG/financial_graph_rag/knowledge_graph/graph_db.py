"""
Graph Database Interface

Handles storage and querying of the financial knowledge graph using Neo4j.
Provides methods for storing entities, relationships, and complex graph queries.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json

from neo4j import GraphDatabase, Transaction
from neo4j.exceptions import ServiceUnavailable, DatabaseError

from ..config import settings
from .entities import BaseEntity, Relationship, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class GraphDatabaseManager:
    """
    Manages the Neo4j graph database for the financial knowledge graph
    """

    def __init__(self):
        """Initialize the graph database connection"""
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(settings.neo4j_uri,
                                               auth=(settings.neo4j_username,
                                                     settings.neo4j_password))

            # Test connection
            with self.driver.session(
                    database=settings.neo4j_database) as session:
                session.run("RETURN 1")

            logger.info("Successfully connected to Neo4j database")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j database connection")

    def setup_schema(self):
        """Create database schema with indexes and constraints"""
        with self.driver.session(database=settings.neo4j_database) as session:
            try:
                # Entity constraints and indexes
                constraints = [
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except DatabaseError as e:
                        # Constraint might already exist
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Error creating constraint: {e}")

                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                    "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
                    "CREATE INDEX company_cik IF NOT EXISTS FOR (c:Company) ON (c.cik)",
                    "CREATE INDEX transaction_date IF NOT EXISTS FOR (t:Transaction) ON (t.announcement_date)",
                    "CREATE INDEX related_type IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.type)",
                ]

                for index in indexes:
                    try:
                        session.run(index)
                    except DatabaseError as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Error creating index: {e}")

                logger.info("Database schema setup completed")

            except Exception as e:
                logger.error(f"Error setting up database schema: {e}")
                raise

    def store_entity(self, entity: BaseEntity) -> bool:
        """
        Store an entity in the graph database
        
        Args:
            entity: Entity to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session(
                    database=settings.neo4j_database) as session:
                # Convert entity to properties dict
                properties = self._entity_to_properties(entity)

                # Create appropriate node label based on entity type
                labels = ["Entity", entity.entity_type.value]
                label_str = ":".join(labels)

                # Merge entity (create or update)
                query = f"""
                MERGE (e:{label_str} {{id: $id}})
                SET e += $properties
                RETURN e.id
                """

                result = session.run(query,
                                     id=entity.id,
                                     properties=properties)

                if result.single():
                    logger.debug(f"Stored entity: {entity.id}")
                    return True

        except Exception as e:
            logger.error(f"Error storing entity {entity.id}: {e}")

        return False

    def store_relationship(self, relationship: Relationship) -> bool:
        """
        Store a relationship in the graph database
        
        Args:
            relationship: Relationship to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session(
                    database=settings.neo4j_database) as session:
                # Convert relationship to properties
                properties = self._relationship_to_properties(relationship)

                # Create relationship
                query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATED {id: $rel_id}]->(target)
                SET r += $properties
                SET r.type = $rel_type
                RETURN r.id
                """

                result = session.run(
                    query,
                    source_id=relationship.source_entity_id,
                    target_id=relationship.target_entity_id,
                    rel_id=relationship.id,
                    rel_type=relationship.relationship_type.value,
                    properties=properties)

                if result.single():
                    logger.debug(f"Stored relationship: {relationship.id}")
                    return True

        except Exception as e:
            logger.error(f"Error storing relationship {relationship.id}: {e}")

        return False

    def store_entities_batch(self, entities: List[BaseEntity]) -> int:
        """
        Store multiple entities in batch
        
        Args:
            entities: List of entities to store
            
        Returns:
            Number of successfully stored entities
        """
        stored_count = 0

        with self.driver.session(database=settings.neo4j_database) as session:
            # Group entities by type for batch processing
            entities_by_type = {}
            for entity in entities:
                entity_type = entity.entity_type.value
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)

            # Process each type in batch
            for entity_type, type_entities in entities_by_type.items():
                try:
                    # Prepare batch data
                    batch_data = []
                    for entity in type_entities:
                        properties = self._entity_to_properties(entity)
                        batch_data.append({
                            'id': entity.id,
                            'properties': properties
                        })

                    # Batch insert
                    labels = f"Entity:{entity_type}"
                    query = f"""
                    UNWIND $batch as row
                    MERGE (e:{labels} {{id: row.id}})
                    SET e += row.properties
                    """

                    session.run(query, batch=batch_data)
                    stored_count += len(type_entities)

                except Exception as e:
                    logger.error(
                        f"Error in batch storing {entity_type} entities: {e}")

        logger.info(f"Stored {stored_count}/{len(entities)} entities in batch")
        return stored_count

    def store_relationships_batch(self,
                                  relationships: List[Relationship]) -> int:
        """
        Store multiple relationships in batch
        
        Args:
            relationships: List of relationships to store
            
        Returns:
            Number of successfully stored relationships
        """
        try:
            with self.driver.session(
                    database=settings.neo4j_database) as session:
                # Prepare batch data
                batch_data = []
                for rel in relationships:
                    properties = self._relationship_to_properties(rel)
                    batch_data.append({
                        'source_id': rel.source_entity_id,
                        'target_id': rel.target_entity_id,
                        'rel_id': rel.id,
                        'rel_type': rel.relationship_type.value,
                        'properties': properties
                    })

                # Batch insert relationships
                query = """
                UNWIND $batch as row
                MATCH (source:Entity {id: row.source_id})
                MATCH (target:Entity {id: row.target_id})
                MERGE (source)-[r:RELATED {id: row.rel_id}]->(target)
                SET r += row.properties
                SET r.type = row.rel_type
                """

                session.run(query, batch=batch_data)

                logger.info(
                    f"Stored {len(relationships)} relationships in batch")
                return len(relationships)

        except Exception as e:
            logger.error(f"Error in batch storing relationships: {e}")
            return 0

    def find_ma_transactions(self,
                             company_id: Optional[str] = None) -> List[Dict]:
        """
        Find M&A transactions in the graph
        
        Args:
            company_id: Optional company ID to filter by
            
        Returns:
            List of transaction data
        """
        with self.driver.session(database=settings.neo4j_database) as session:
            if company_id:
                query = """
                MATCH (c:Company {id: $company_id})
                MATCH (c)-[r:RELATED]->(t:Transaction)
                WHERE r.type IN ['acquired', 'acquired_by', 'merged_with', 'divested', 'spun_off']
                RETURN t, r, c
                ORDER BY t.announcement_date DESC
                """
                result = session.run(query, company_id=company_id)
            else:
                query = """
                MATCH (t:Transaction)
                OPTIONAL MATCH (c1:Company)-[r1:RELATED]->(t)
                OPTIONAL MATCH (t)-[r2:RELATED]->(c2:Company)
                WHERE r1.type IN ['acquired', 'acquired_by', 'merged_with', 'divested', 'spun_off']
                   OR r2.type IN ['acquired', 'acquired_by', 'merged_with', 'divested', 'spun_off']
                RETURN t, c1, c2, r1, r2
                ORDER BY t.announcement_date DESC
                """
                result = session.run(query)

            transactions = []
            for record in result:
                transaction_data = dict(record['t'])
                transactions.append(transaction_data)

            return transactions

    def find_company_structure(self, company_id: str) -> Dict:
        """
        Find organizational structure for a company
        
        Args:
            company_id: Company ID
            
        Returns:
            Dictionary with company structure information
        """
        with self.driver.session(database=settings.neo4j_database) as session:
            # Get company and subsidiaries
            subsidiaries_query = """
            MATCH (c:Company {id: $company_id})
            OPTIONAL MATCH (c)-[r:RELATED]->(s:Subsidiary)
            WHERE r.type IN ['owns', 'parent_of']
            RETURN c, collect(s) as subsidiaries
            """

            # Get divisions and business units
            divisions_query = """
            MATCH (c:Company {id: $company_id})
            OPTIONAL MATCH (c)-[r:RELATED]->(d:Division)
            WHERE r.type IN ['contains', 'owns']
            RETURN collect(d) as divisions
            """

            # Get executives
            executives_query = """
            MATCH (c:Company {id: $company_id})
            OPTIONAL MATCH (p:Person)-[r:RELATED]->(c)
            WHERE r.type IN ['ceo_of', 'executive_of', 'board_member_of']
            RETURN collect({person: p, relationship: r}) as executives
            """

            company_result = session.run(subsidiaries_query,
                                         company_id=company_id)
            divisions_result = session.run(divisions_query,
                                           company_id=company_id)
            executives_result = session.run(executives_query,
                                            company_id=company_id)

            company_record = company_result.single()
            divisions_record = divisions_result.single()
            executives_record = executives_result.single()

            if not company_record:
                return {}

            # Safely handle divisions
            divisions = []
            if divisions_record and divisions_record['divisions']:
                divisions = [
                    dict(d) for d in divisions_record['divisions']
                    if d is not None
                ]

            # Safely handle executives
            executives = []
            if executives_record and executives_record['executives']:
                executives = [{
                    'person': dict(exec_data['person']),
                    'role': exec_data['relationship']['type']
                } for exec_data in executives_record['executives']
                              if exec_data is not None
                              and exec_data['person'] is not None]

            # Safely handle subsidiaries
            subsidiaries = []
            if company_record['subsidiaries']:
                subsidiaries = [
                    dict(s) for s in company_record['subsidiaries']
                    if s is not None
                ]

            return {
                'company': dict(company_record['c']),
                'subsidiaries': subsidiaries,
                'divisions': divisions,
                'executives': executives
            }

    def analyze_ma_impact_on_structure(self, company_id: str,
                                       years: List[int]) -> Dict:
        """
        Analyze M&A impact on organizational structure over time
        
        Args:
            company_id: Company ID
            years: Years to analyze
            
        Returns:
            Analysis results
        """
        with self.driver.session(database=settings.neo4j_database) as session:
            # Find M&A transactions by year
            query = """
            MATCH (c:Company {id: $company_id})
            MATCH (c)-[r:RELATED]->(t:Transaction)
            WHERE r.type IN ['acquired', 'acquired_by', 'merged_with', 'divested', 'spun_off']
            AND t.announcement_date IS NOT NULL
            AND t.announcement_date CONTAINS $year_filter
            RETURN t, r
            ORDER BY t.announcement_date
            """

            results = {}
            for year in years:
                year_transactions = []
                result = session.run(query,
                                     company_id=company_id,
                                     year_filter=str(year))

                for record in result:
                    transaction = dict(record['t'])
                    relationship = dict(record['r'])
                    year_transactions.append({
                        'transaction':
                        transaction,
                        'relationship_type':
                        relationship['type']
                    })

                results[year] = year_transactions

            return results

    def _entity_to_properties(self, entity: BaseEntity) -> Dict:
        """Convert entity to Neo4j properties dictionary"""
        properties = {
            'id':
            entity.id,
            'name':
            entity.name,
            'entity_type':
            entity.entity_type.value,
            'confidence_score':
            entity.confidence_score,
            'created_at':
            entity.created_at.isoformat() if entity.created_at else None
        }

        # Add optional fields if they exist
        if entity.source_document:
            properties['source_document'] = entity.source_document
        if entity.extracted_text:
            properties['extracted_text'] = entity.extracted_text
        if entity.metadata:
            properties['metadata'] = json.dumps(entity.metadata)

        # Add entity-specific properties
        for field_name in entity.__dataclass_fields__.keys():
            if field_name not in [
                    'id', 'name', 'entity_type', 'confidence_score',
                    'created_at', 'source_document', 'extracted_text',
                    'metadata'
            ]:
                value = getattr(entity, field_name)
                if value is not None:
                    if isinstance(value, datetime):
                        properties[field_name] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        properties[field_name] = json.dumps(value)
                    else:
                        properties[field_name] = value

        return properties

    def _relationship_to_properties(self, relationship: Relationship) -> Dict:
        """Convert relationship to Neo4j properties dictionary"""
        properties = {
            'id':
            relationship.id,
            'type':
            relationship.relationship_type.value,
            'confidence_score':
            relationship.confidence_score,
            'created_at':
            relationship.created_at.isoformat()
            if relationship.created_at else None
        }

        # Add optional fields
        if relationship.source_document:
            properties['source_document'] = relationship.source_document
        if relationship.extracted_text:
            properties['extracted_text'] = relationship.extracted_text
        if relationship.valid_from:
            properties['valid_from'] = relationship.valid_from.isoformat()
        if relationship.valid_to:
            properties['valid_to'] = relationship.valid_to.isoformat()
        if relationship.properties:
            properties.update(relationship.properties)

        return properties

    def execute_custom_query(self,
                             query: str,
                             parameters: Dict = None) -> List[Dict]:
        """
        Execute a custom Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            with self.driver.session(
                    database=settings.neo4j_database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []
