"""
Entity and Relationship Extractor

Uses LLMs to extract structured entities and relationships from financial documents,
specifically focused on M&A activities and organizational structure changes.
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Triplex model imports (optional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRIPLEX_AVAILABLE = True
except ImportError:
    TRIPLEX_AVAILABLE = False

from ..config import settings
from .entities import (BaseEntity, Relationship, EntityType, RelationshipType,
                       Company, Person, Transaction, Subsidiary, Division,
                       Location, create_entity)

logger = logging.getLogger(__name__)


class EntityRelationshipExtractor:
    """
    Extracts entities and relationships from financial documents using LLMs or Triplex
    """

    def __init__(self,
                 llm_provider: str = None,
                 model_name: str = None,
                 extraction_method: str = None):
        """
        Initialize the extractor with specified LLM or Triplex model
        
        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            model_name: Specific model name
            extraction_method: "llm" or "triplex"
        """
        self.extraction_method = extraction_method or settings.extraction_method
        self.llm_provider = llm_provider or settings.default_llm_provider
        self.model_name = model_name or settings.default_llm_model

        # Initialize extraction method
        if self.extraction_method == "triplex":
            self._initialize_triplex()
        else:
            self.llm = self._initialize_llm()

        # Text splitter for handling long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""])

        # Extraction prompts (for LLM method)
        if self.extraction_method == "llm":
            self.entity_extraction_prompt = self._create_entity_extraction_prompt(
            )
            self.relationship_extraction_prompt = self._create_relationship_extraction_prompt(
            )
            self.ma_context_prompt = self._create_ma_context_prompt()

    def _initialize_llm(self):
        """Initialize the language model based on provider"""
        if self.llm_provider == "openai":
            return ChatOpenAI(model=self.model_name,
                              temperature=0.1,
                              api_key=settings.openai_api_key,
                              base_url=getattr(settings, 'openai_base_url',
                                               None))
        elif self.llm_provider == "anthropic":
            return ChatAnthropic(model=self.model_name,
                                 temperature=0.1,
                                 api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _initialize_triplex(self):
        """Initialize the Triplex model for knowledge graph extraction"""
        if not TRIPLEX_AVAILABLE:
            raise ImportError(
                "Triplex requires transformers and torch. Install with: pip install transformers torch"
            )

        logger.info(f"Loading Triplex model: {settings.triplex_model_name}")

        # Determine device
        if settings.triplex_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends,
                         'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = settings.triplex_device

        logger.info(f"Using device: {device}")

        # Load model and tokenizer with specific configuration to avoid cache issues
        self.triplex_model = AutoModelForCausalLM.from_pretrained(
            settings.triplex_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            attn_implementation=
            "eager",  # Use eager attention to avoid flash attention issues
            device_map=device)

        self.triplex_tokenizer = AutoTokenizer.from_pretrained(
            settings.triplex_model_name, trust_remote_code=True)

        # Set pad token if not set
        if self.triplex_tokenizer.pad_token is None:
            self.triplex_tokenizer.pad_token = self.triplex_tokenizer.eos_token

        self.triplex_device = device
        logger.info("Triplex model loaded successfully")

    def _create_entity_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for entity extraction"""
        template = """
        You are a financial analyst expert at extracting structured information from SEC 10-K filings.
        Extract entities related to mergers, acquisitions, organizational changes, and corporate structure.

        From the following text, extract entities in the specified JSON format:

        TEXT:
        {text}

        Extract entities for these types:
        - Company: Companies, subsidiaries, joint ventures
        - Person: Executives, board members, key personnel  
        - Transaction: M&A deals, acquisitions, mergers, divestitures
        - Subsidiary: Subsidiary companies and ownership structures
        - Division: Business divisions, segments, units
        - Location: Headquarters, offices, geographical presence
        - FinancialMetric: Revenue, profit, valuation figures

        CRITICAL: Return ONLY a valid JSON object with this EXACT structure. Do not include any explanatory text, markdown formatting, or code blocks.

        {{
            "entities": [
                {{
                    "id": "unique_identifier",
                    "name": "entity_name", 
                    "type": "Company",
                    "confidence": 0.85,
                    "attributes": {{
                        "ticker": "MSFT",
                        "sector": "Technology"
                    }},
                    "extracted_text": "relevant_text_snippet"
                }}
            ]
        }}

        Valid type values: Company, Person, Transaction, Subsidiary, Division, Location, FinancialMetric

        Focus on entities related to:
        - Mergers and acquisitions
        - Corporate restructuring  
        - Subsidiary relationships
        - Executive changes
        - Organizational structure changes

        If no relevant entities are found, return: {{"entities": []}}
        
        Remember: ONLY return the JSON object, no other text.
        """

        return ChatPromptTemplate.from_template(template)

    def _create_relationship_extraction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for relationship extraction"""
        template = """
        You are a financial analyst expert at identifying relationships between entities in SEC 10-K filings.
        
        Given these entities and the text, extract relationships between them:

        ENTITIES:
        {entities}

        TEXT:
        {text}

        Extract relationships for these types:
        - acquired, acquired_by, merged_with, divested, spun_off
        - owns, owned_by, subsidiary_of, parent_of
        - ceo_of, executive_of, board_member_of
        - part_of, contains, reports_to, manages
        - located_in, headquartered_in
        - competitor_of, partner_of

        CRITICAL: Return ONLY a valid JSON object with this EXACT structure. Do not include any explanatory text, markdown formatting, or code blocks.

        {{
            "relationships": [
                {{
                    "id": "unique_identifier",
                    "source_entity": "entity_id_from_above",
                    "target_entity": "entity_id_from_above", 
                    "type": "acquired",
                    "confidence": 0.90,
                    "properties": {{
                        "date": "2023-03-15",
                        "value": "68.7 billion USD",
                        "status": "completed"
                    }},
                    "extracted_text": "relevant_text_snippet"
                }}
            ]
        }}

        Valid relationship types: acquired, acquired_by, merged_with, divested, spun_off, owns, owned_by, subsidiary_of, parent_of, ceo_of, executive_of, board_member_of, part_of, contains, reports_to, manages, located_in, headquartered_in, competitor_of, partner_of

        Focus on relationships that show:
        - M&A transactions and their status
        - Corporate structure and ownership
        - Executive appointments and changes
        - Organizational reporting relationships

        If no relationships are found, return: {{"relationships": []}}
        
        Remember: ONLY return the JSON object, no other text.
        """

        return ChatPromptTemplate.from_template(template)

    def _create_ma_context_prompt(self) -> ChatPromptTemplate:
        """Create prompt for identifying M&A context and organizational impact"""
        template = """
        You are analyzing a 10-K filing section for merger & acquisition activity and organizational impact.
        
        Analyze this text for M&A context and organizational structure changes:

        TEXT:
        {text}

        Provide analysis in JSON format:
        {{
            "ma_activity": {{
                "has_ma_content": true/false,
                "ma_score": 0.0-1.0,
                "summary": "brief_summary_of_ma_activity"
            }},
            "organizational_impact": {{
                "structure_changes": "description_of_changes",
                "integration_challenges": "integration_issues_mentioned", 
                "synergies": "synergy_expectations",
                "personnel_changes": "executive_or_organizational_changes"
            }},
            "key_insights": [
                "insight_1",
                "insight_2"
            ]
        }}

        Focus on:
        - Completed or announced M&A transactions
        - Impact on organizational structure
        - Integration challenges and synergies
        - Changes in business operations
        - Personnel and leadership changes
        """

        return ChatPromptTemplate.from_template(template)

    def extract_from_document(
            self, document_text: str,
            document_id: str) -> Tuple[List[BaseEntity], List[Relationship]]:
        """
        Extract entities and relationships from a document
        
        Args:
            document_text: Full document text
            document_id: Unique document identifier
            
        Returns:
            Tuple of (entities, relationships)
        """
        logger.info(
            f"Extracting entities and relationships from document {document_id}"
        )

        # Split document into chunks
        chunks = self.text_splitter.split_text(document_text)
        logger.info(f"Split document into {len(chunks)} chunks")

        all_entities = []
        all_relationships = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Extract entities from chunk
                chunk_entities = self._extract_entities_from_chunk(
                    chunk, f"{document_id}_chunk_{i}")
                all_entities.extend(chunk_entities)

                # Extract relationships from chunk if we have entities
                if chunk_entities:
                    chunk_relationships = self._extract_relationships_from_chunk(
                        chunk, chunk_entities, f"{document_id}_chunk_{i}")
                    all_relationships.extend(chunk_relationships)

            except Exception as e:
                logger.error(
                    f"Error processing chunk {i} of document {document_id}: {e}"
                )
                continue

        # Deduplicate entities and relationships
        entities = self._deduplicate_entities(all_entities)
        relationships = self._deduplicate_relationships(all_relationships)

        logger.info(
            f"Extracted {len(entities)} entities and {len(relationships)} relationships"
        )
        return entities, relationships

    def _extract_entities_from_chunk(self, chunk: str,
                                     chunk_id: str) -> List[BaseEntity]:
        """Extract entities from a text chunk"""
        if self.extraction_method == "triplex":
            return self._extract_entities_triplex(chunk, chunk_id)
        else:
            return self._extract_entities_llm(chunk, chunk_id)

    def _extract_entities_llm(self, chunk: str,
                              chunk_id: str) -> List[BaseEntity]:
        """Extract entities using LLM method"""
        try:
            # Create prompt
            prompt = self.entity_extraction_prompt.format(text=chunk)

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Clean and parse JSON response
            result = self._parse_json_response(response.content, chunk_id)
            if not result:
                return []

            entities = []
            for entity_data in result.get("entities", []):
                entity = self._create_entity_from_extraction(
                    entity_data, chunk_id)
                if entity:
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(
                f"Error extracting entities from chunk {chunk_id}: {e}")
            return []

    def _extract_entities_triplex(self, chunk: str,
                                  chunk_id: str) -> List[BaseEntity]:
        """Extract entities using Triplex model"""
        try:
            # Define entity types for financial/M&A extraction
            entity_types = [
                "COMPANY", "PERSON", "TRANSACTION", "SUBSIDIARY", "DIVISION",
                "LOCATION", "FINANCIAL_METRIC", "DATE", "MONEY", "PERCENTAGE",
                "ORGANIZATION"
            ]

            # Define predicates for financial relationships
            predicates = [
                "ACQUIRED", "ACQUIRED_BY", "MERGED_WITH", "DIVESTED",
                "SPUN_OFF", "OWNS", "OWNED_BY", "SUBSIDIARY_OF", "PARENT_OF",
                "CEO_OF", "EXECUTIVE_OF", "HEADQUARTERED_IN", "LOCATED_IN",
                "VALUE", "DATE", "STATUS"
            ]

            # Extract using Triplex
            result = self._triplex_extract(chunk, entity_types, predicates)

            # Check if Triplex output is meaningful
            if not result or len(result.strip(
            )) < 10 or "entities_and_triples" in result.lower():
                # Fall back to rule-based extraction
                logger.info(
                    f"Triplex output seems invalid, falling back to rule-based extraction for chunk {chunk_id}"
                )
                result = self._simple_rule_based_extraction(
                    chunk, entity_types, predicates)

            # Parse the result and convert to our entity format
            entities = self._parse_triplex_entities(result, chunk_id)

            return entities

        except Exception as e:
            logger.error(
                f"Error extracting entities with Triplex from chunk {chunk_id}: {e}"
            )
            # Fall back to rule-based extraction on error
            try:
                result = self._simple_rule_based_extraction(
                    chunk, entity_types, predicates)
                entities = self._parse_triplex_entities(result, chunk_id)
                return entities
            except Exception as e2:
                logger.error(
                    f"Rule-based fallback also failed for chunk {chunk_id}: {e2}"
                )
                return []

    def _triplex_extract(self, text: str, entity_types: List[str],
                         predicates: List[str]) -> str:
        """Run Triplex extraction on text"""
        prompt = f"Extract entities and relationships from this text:\n\n{text[:1500]}\n\nOutput format: (entity1, relationship, entity2)\n\nTriplets:"

        try:
            # Tokenize with attention mask
            inputs = self.triplex_tokenizer(prompt,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            add_special_tokens=True).to(
                                                self.triplex_device)

            # Add attention mask explicitly
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            # Use manual generation loop to avoid cache issues
            max_new_tokens = 128
            generated_ids = inputs['input_ids'].clone()

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Get model outputs
                    outputs = self.triplex_model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids))

                    # Get next token probabilities
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits,
                                                 dim=-1,
                                                 keepdim=True)

                    # Stop if EOS token
                    if next_token_id.item(
                    ) == self.triplex_tokenizer.eos_token_id:
                        break

                    # Append new token
                    generated_ids = torch.cat([generated_ids, next_token_id],
                                              dim=-1)

            # Decode the generated part only
            generated_text = self.triplex_tokenizer.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Triplex generation failed: {e}")
            # Return simple rule-based extraction as fallback
            return self._simple_rule_based_extraction(text, entity_types,
                                                      predicates)

    def _simple_rule_based_extraction(self, text: str, entity_types: List[str],
                                      predicates: List[str]) -> str:
        """Fallback rule-based extraction when Triplex fails"""
        import re

        results = []

        # Look for company names (more precise patterns)
        company_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(Corporation|Corp|Inc|LLC|Ltd|Company|Group)\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?=\s+(?:acquired|merged|divested))',
            r'\b([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)\b(?=\s+for\s+\$|\s+in\s+20)'
        ]

        companies = set()
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    company_name = f"{match[0]} {match[1]}".strip()
                else:
                    company_name = match.strip()
                if len(company_name) > 3:
                    companies.add(company_name)

        # Extract specific company names from common acquisition patterns
        acquisition_patterns = [
            r'([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))\s+acquired\s+([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group|\w+))',
            r'([A-Z][a-zA-Z\s]+?)\s+acquired\s+([A-Z][a-zA-Z\s]+?)\s+for',
            r'([A-Z][a-zA-Z\s]+?)\s+acquired\s+([A-Z][a-zA-Z\s]+?)\s+in\s+20'
        ]

        for pattern in acquisition_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                acquirer = match[0].strip()
                target = match[1].strip()

                # Clean up the target (remove trailing words like "for")
                target = re.sub(r'\s+for\s+.*$', '', target)
                target = re.sub(r'\s+in\s+20\d{2}.*$', '', target)

                if len(acquirer) > 2 and len(target) > 2:
                    companies.add(acquirer)
                    companies.add(target)
                    results.append(f"({acquirer}, acquired, {target})")

        # Look for merger patterns
        merger_patterns = [
            r'([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))\s+merged\s+with\s+([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))',
            r'merger\s+between\s+([A-Z][a-zA-Z\s]+?)\s+and\s+([A-Z][a-zA-Z\s]+?)'
        ]

        for pattern in merger_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                company1 = match[0].strip()
                company2 = match[1].strip()
                if len(company1) > 2 and len(company2) > 2:
                    companies.add(company1)
                    companies.add(company2)
                    results.append(f"({company1}, merged_with, {company2})")

        # Look for divestiture patterns
        divestiture_patterns = [
            r'([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))\s+divested\s+([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group|\w+))',
            r'([A-Z][a-zA-Z\s]+?)\s+divested\s+([A-Z][a-zA-Z\s]+?)\s+(?:for|to)',
            r'([A-Z][a-zA-Z\s]+?)\s+sold\s+([A-Z][a-zA-Z\s]+?)\s+(?:division|business|unit)'
        ]

        for pattern in divestiture_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                divester = match[0].strip()
                divested_asset = match[1].strip()

                # Clean up the divested asset
                divested_asset = re.sub(r'\s+(?:for|to)\s+.*$', '',
                                        divested_asset)

                if len(divester) > 2 and len(divested_asset) > 2:
                    companies.add(divester)
                    companies.add(divested_asset)
                    results.append(f"({divester}, divested, {divested_asset})")

        # Look for spin-off patterns
        spinoff_patterns = [
            r'([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))\s+spun\s+off\s+([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group|\w+))',
            r'([A-Z][a-zA-Z\s]+?)\s+spun-off\s+([A-Z][a-zA-Z\s]+?)',
            r'spin-off\s+of\s+([A-Z][a-zA-Z\s]+?)\s+from\s+([A-Z][a-zA-Z\s]+?(?:Corporation|Corp|Inc|LLC|Ltd|Company|Group))'
        ]

        for pattern in spinoff_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    if 'from' in pattern:  # spin-off of X from Y
                        spun_off_entity = match[0].strip()
                        parent_company = match[1].strip()
                    else:  # Y spun off X
                        parent_company = match[0].strip()
                        spun_off_entity = match[1].strip()

                    if len(parent_company) > 2 and len(spun_off_entity) > 2:
                        companies.add(parent_company)
                        companies.add(spun_off_entity)
                        results.append(
                            f"({parent_company}, spun_off, {spun_off_entity})")

        # Add company entities
        for company in sorted(companies)[:5]:  # Limit to top 5 companies
            results.append(f"({company}, type, COMPANY)")

        # Look for monetary amounts and associate with transactions
        money_pattern = r'\$?([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?\s*(USD|dollars?)?'
        amounts = re.findall(money_pattern, text, re.IGNORECASE)

        if amounts and results:
            # Associate the first amount with the first transaction
            amount_str = f"${amounts[0][0]} {amounts[0][1] or ''} {amounts[0][2] or ''}".strip(
            )
            if ',' in results[0]:  # If it's a relationship
                parts = results[0].strip('()').split(', ')
                if len(parts) == 3:
                    results.append(f"({parts[0]}, paid, {amount_str})")

        return '\n'.join(results)

    def _parse_triplex_entities(self, triplex_output: str,
                                chunk_id: str) -> List[BaseEntity]:
        """Parse Triplex output to extract entities"""
        entities = []
        seen_entities = set()

        if not triplex_output or not triplex_output.strip():
            logger.warning(f"Empty Triplex output for chunk {chunk_id}")
            return entities

        try:
            # Clean the output
            cleaned_output = triplex_output.strip()
            logger.debug(
                f"Triplex output for {chunk_id}: {cleaned_output[:200]}...")

            # Split into lines for processing
            lines = cleaned_output.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for triplet format: (subject, predicate, object)
                triplet_matches = re.findall(
                    r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
                for match in triplet_matches:
                    subject, predicate, obj = match

                    # Handle entity type declarations (e.g., "(Microsoft Corporation, type, COMPANY)")
                    if predicate.strip().lower() in ['type', 'is_a', 'is']:
                        entity_name = subject.strip().strip('"').strip("'")
                        entity_type_str = obj.strip().strip('"').strip(
                            "'").upper()

                        if entity_name and entity_name not in seen_entities and len(
                                entity_name) > 1:
                            # Map type strings to our EntityType enum
                            entity_type = self._map_type_string_to_entity_type(
                                entity_type_str)
                            entity = self._create_triplex_entity_with_type(
                                entity_name, entity_type, chunk_id)
                            if entity:
                                entities.append(entity)
                                seen_entities.add(entity_name)
                    else:
                        # Create entities for subject and object in relationships
                        for entity_text in [subject.strip(), obj.strip()]:
                            entity_name = entity_text.strip().strip('"').strip(
                                "'")
                            if entity_name and entity_name not in seen_entities and len(
                                    entity_name) > 1:
                                # Skip monetary amounts and other non-entity objects
                                if not re.match(
                                        r'^\$[\d,]+', entity_name
                                ) and not entity_name.lower() in [
                                        'acquired', 'merged_with', 'paid'
                                ]:
                                    entity = self._create_triplex_entity(
                                        entity_name, chunk_id)
                                    if entity:
                                        entities.append(entity)
                                        seen_entities.add(entity_name)

                # Also look for quoted entities or capitalized words (fallback)
                if not entities:  # Only if we haven't found entities through triplet parsing
                    quoted_entities = re.findall(r'"([^"]+)"', line)
                    for quoted in quoted_entities:
                        if quoted and quoted not in seen_entities and len(
                                quoted) > 1:
                            entity = self._create_triplex_entity(
                                quoted, chunk_id)
                            if entity:
                                entities.append(entity)
                                seen_entities.add(quoted)

                    # Extract capitalized phrases as potential entities
                    capitalized_phrases = re.findall(
                        r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', line)
                    for phrase in capitalized_phrases:
                        if (phrase and phrase not in seen_entities
                                and len(phrase) > 2 and phrase not in [
                                    'The', 'This', 'That', 'And', 'Or',
                                    'COMPANY', 'PERSON'
                                ]):
                            entity = self._create_triplex_entity(
                                phrase, chunk_id)
                            if entity:
                                entities.append(entity)
                                seen_entities.add(phrase)

        except Exception as e:
            logger.error(
                f"Error parsing Triplex output for chunk {chunk_id}: {e}")

        logger.info(
            f"Parsed {len(entities)} entities from Triplex output for chunk {chunk_id}"
        )
        return entities

    def _map_type_string_to_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum"""
        type_mapping = {
            'COMPANY': EntityType.COMPANY,
            'CORPORATION': EntityType.COMPANY,
            'ORGANIZATION': EntityType.COMPANY,
            'PERSON': EntityType.PERSON,
            'INDIVIDUAL': EntityType.PERSON,
            'LOCATION': EntityType.LOCATION,
            'PLACE': EntityType.LOCATION,
            'TRANSACTION': EntityType.TRANSACTION,
            'DEAL': EntityType.TRANSACTION,
            'SUBSIDIARY': EntityType.SUBSIDIARY,
            'DIVISION': EntityType.DIVISION
        }
        return type_mapping.get(type_str.upper(), EntityType.COMPANY)

    def _create_triplex_entity_with_type(
            self, entity_name: str, entity_type: EntityType,
            chunk_id: str) -> Optional[BaseEntity]:
        """Create entity with specific type from Triplex extracted text"""
        try:
            # Clean entity text
            entity_name = entity_name.strip().strip('"').strip("'")
            if not entity_name or len(entity_name) < 2:
                return None

            # Generate ID
            entity_id = self._generate_entity_id(entity_name,
                                                 entity_type.value)

            # Create entity with specified type
            return create_entity(
                entity_type,
                id=entity_id,
                name=entity_name,
                confidence_score=0.8,  # Default confidence for Triplex
                source_document=chunk_id,
                extracted_text=entity_name)

        except Exception as e:
            logger.error(
                f"Error creating entity with type '{entity_type}' from Triplex text '{entity_name}': {e}"
            )
            return None

    def _create_triplex_entity(self, entity_text: str,
                               chunk_id: str) -> Optional[BaseEntity]:
        """Create entity from Triplex extracted text with inferred type"""
        try:
            # Clean entity text
            entity_name = entity_text.strip().strip('"').strip("'")
            if not entity_name or len(entity_name) < 2:
                return None

            # Determine entity type based on patterns
            entity_type = self._infer_entity_type(entity_name)

            # Generate ID
            entity_id = self._generate_entity_id(entity_name,
                                                 entity_type.value)

            # Create entity
            return create_entity(
                entity_type,
                id=entity_id,
                name=entity_name,
                confidence_score=0.8,  # Default confidence for Triplex
                source_document=chunk_id,
                extracted_text=entity_text)

        except Exception as e:
            logger.error(
                f"Error creating entity from Triplex text '{entity_text}': {e}"
            )
            return None

    def _infer_entity_type(self, entity_name: str) -> EntityType:
        """Infer entity type from name patterns"""
        name_lower = entity_name.lower()

        # Company indicators
        if any(
                indicator in name_lower for indicator in
            ['corp', 'inc', 'ltd', 'llc', 'corporation', 'company', 'group']):
            return EntityType.COMPANY

        # Person indicators (contains common titles or name patterns)
        if any(title in name_lower for title in
               ['ceo', 'cfo', 'president', 'director', 'mr.', 'ms.', 'dr.']):
            return EntityType.PERSON

        # Location indicators
        if any(loc in name_lower for loc in
               ['city', 'state', 'country', 'headquarters', 'office']):
            return EntityType.LOCATION

        # Transaction indicators
        if any(trans in name_lower
               for trans in ['acquisition', 'merger', 'deal', 'transaction']):
            return EntityType.TRANSACTION

        # Default to Company for financial context
        return EntityType.COMPANY

    def _extract_entity_names_from_line(self, line: str) -> List[str]:
        """Extract entity names from a line of text"""
        # Simple extraction - look for capitalized words/phrases
        import re

        # Find capitalized phrases (potential entity names)
        capitalized_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        matches = re.findall(capitalized_pattern, line)

        # Filter out common words and keep meaningful entities
        stopwords = {
            'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At', 'To',
            'For', 'Of', 'With'
        }
        entities = [
            match for match in matches
            if match not in stopwords and len(match) > 2
        ]

        return entities

    def _extract_relationships_from_chunk(self, chunk: str,
                                          entities: List[BaseEntity],
                                          chunk_id: str) -> List[Relationship]:
        """Extract relationships from a text chunk given entities"""
        if self.extraction_method == "triplex":
            return self._extract_relationships_triplex(chunk, entities,
                                                       chunk_id)
        else:
            return self._extract_relationships_llm(chunk, entities, chunk_id)

    def _extract_relationships_llm(self, chunk: str,
                                   entities: List[BaseEntity],
                                   chunk_id: str) -> List[Relationship]:
        """Extract relationships using LLM method"""
        try:
            # Serialize entities for prompt
            entities_json = [{
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type.value
            } for entity in entities]

            # Create prompt
            prompt = self.relationship_extraction_prompt.format(
                entities=json.dumps(entities_json, indent=2), text=chunk)

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Clean and parse JSON response
            result = self._parse_json_response(response.content, chunk_id)
            if not result:
                return []

            relationships = []
            for rel_data in result.get("relationships", []):
                relationship = self._create_relationship_from_extraction(
                    rel_data, chunk_id)
                if relationship:
                    relationships.append(relationship)

            return relationships

        except Exception as e:
            logger.error(
                f"Error extracting relationships from chunk {chunk_id}: {e}")
            return []

    def _extract_relationships_triplex(self, chunk: str,
                                       entities: List[BaseEntity],
                                       chunk_id: str) -> List[Relationship]:
        """Extract relationships using Triplex model"""
        try:
            if len(entities) < 2:
                return []  # Need at least 2 entities for relationships

            # Extract triplets that include relationships
            entity_types = ["COMPANY", "PERSON", "ORGANIZATION", "LOCATION"]
            predicates = [
                "ACQUIRED", "ACQUIRED_BY", "MERGED_WITH", "DIVESTED",
                "SPUN_OFF", "OWNS", "OWNED_BY", "SUBSIDIARY_OF", "PARENT_OF",
                "CEO_OF", "EXECUTIVE_OF", "HEADQUARTERED_IN", "LOCATED_IN",
                "VALUE", "DATE", "STATUS"
            ]

            # Get Triplex output
            triplex_output = self._triplex_extract(chunk, entity_types,
                                                   predicates)

            # Check if Triplex output is meaningful for relationships
            if not triplex_output or len(triplex_output.strip(
            )) < 10 or "entities_and_triples" in triplex_output.lower():
                # Fall back to rule-based extraction
                logger.info(
                    f"Triplex relationship output seems invalid, falling back to rule-based extraction for chunk {chunk_id}"
                )
                triplex_output = self._simple_rule_based_extraction(
                    chunk, entity_types, predicates)

            # Parse relationships from triplets
            relationships = self._parse_triplex_relationships(
                triplex_output, entities, chunk_id)

            return relationships

        except Exception as e:
            logger.error(
                f"Error extracting relationships with Triplex from chunk {chunk_id}: {e}"
            )
            # Fall back to rule-based extraction on error
            try:
                fallback_output = self._simple_rule_based_extraction(
                    chunk, entity_types, predicates)
                relationships = self._parse_triplex_relationships(
                    fallback_output, entities, chunk_id)
                return relationships
            except Exception as e2:
                logger.error(
                    f"Rule-based relationship fallback also failed for chunk {chunk_id}: {e2}"
                )
                return []

    def _parse_triplex_relationships(self, triplex_output: str,
                                     entities: List[BaseEntity],
                                     chunk_id: str) -> List[Relationship]:
        """Parse relationships from Triplex triplet output"""
        relationships = []
        entity_name_to_id = {
            entity.name.lower(): entity.id
            for entity in entities
        }

        logger.debug(
            f"Parsing relationships for chunk {chunk_id} with {len(entities)} entities"
        )
        logger.debug(f"Entity names: {list(entity_name_to_id.keys())}")

        try:
            lines = triplex_output.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Parse triplet format: (subject, predicate, object)
                triplet_match = re.search(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
                                          line)
                if triplet_match:
                    subject, predicate, obj = triplet_match.groups()
                    subject = subject.strip().strip('"').strip("'")
                    predicate = predicate.strip().strip('"').strip("'")
                    obj = obj.strip().strip('"').strip("'")

                    logger.debug(
                        f"Found triplet: ({subject}, {predicate}, {obj})")

                    # Skip entity type declarations
                    if predicate.lower() in ['type', 'is_a', 'is']:
                        logger.debug(f"Skipping type declaration: {predicate}")
                        continue

                    # Skip monetary relationships for now (they need special handling)
                    if predicate.lower() in ['paid'] or obj.startswith('$'):
                        logger.debug(
                            f"Skipping monetary relationship: {predicate} -> {obj}"
                        )
                        continue

                    # Find corresponding entity IDs
                    source_id = entity_name_to_id.get(subject.lower())
                    target_id = entity_name_to_id.get(obj.lower())

                    logger.debug(
                        f"Entity mapping: '{subject}' -> {source_id}, '{obj}' -> {target_id}"
                    )

                    if source_id and target_id and source_id != target_id:
                        # Map predicate to our relationship types
                        rel_type = self._map_predicate_to_relationship_type(
                            predicate)
                        logger.debug(
                            f"Mapped predicate '{predicate}' to relationship type: {rel_type}"
                        )

                        if rel_type:
                            rel_id = self._generate_relationship_id(
                                source_id, target_id, rel_type.value)

                            relationship = Relationship(
                                id=rel_id,
                                source_entity_id=source_id,
                                target_entity_id=target_id,
                                relationship_type=rel_type,
                                confidence_score=
                                0.8,  # Default confidence for Triplex/rule-based
                                properties={"predicate": predicate},
                                source_document=chunk_id,
                                extracted_text=line)
                            relationships.append(relationship)
                            logger.debug(
                                f"Created relationship: {source_id} -{rel_type.value}-> {target_id}"
                            )
                        else:
                            logger.debug(
                                f"No relationship type mapping found for predicate: {predicate}"
                            )
                    else:
                        logger.debug(
                            f"Could not find entities or same entity: source_id={source_id}, target_id={target_id}"
                        )

        except Exception as e:
            logger.error(
                f"Error parsing Triplex relationships for chunk {chunk_id}: {e}"
            )

        logger.info(
            f"Parsed {len(relationships)} relationships from Triplex output for chunk {chunk_id}"
        )
        return relationships

    def _map_predicate_to_relationship_type(
            self, predicate: str) -> Optional[RelationshipType]:
        """Map Triplex predicate to our RelationshipType"""
        predicate_lower = predicate.lower()

        # M&A relationships
        if "acquired" in predicate_lower:
            return RelationshipType.ACQUIRED if "by" not in predicate_lower else RelationshipType.ACQUIRED_BY
        elif "merged" in predicate_lower:
            return RelationshipType.MERGED_WITH
        elif "divested" in predicate_lower or "divest" in predicate_lower:
            return RelationshipType.DIVESTED
        elif "spun_off" in predicate_lower or "spinoff" in predicate_lower or "spin" in predicate_lower:
            return RelationshipType.SPUN_OFF
        elif "owned_by" in predicate_lower or "owned" in predicate_lower:
            return RelationshipType.OWNED_BY
        elif "owns" in predicate_lower or "own" in predicate_lower:
            return RelationshipType.OWNS
        elif "subsidiary" in predicate_lower:
            return RelationshipType.SUBSIDIARY_OF
        elif "parent" in predicate_lower:
            return RelationshipType.PARENT_OF

        # Executive relationships
        elif "ceo" in predicate_lower:
            return RelationshipType.CEO_OF
        elif "executive" in predicate_lower:
            return RelationshipType.EXECUTIVE_OF

        # Location relationships
        elif "headquartered" in predicate_lower:
            return RelationshipType.HEADQUARTERED_IN
        elif "located" in predicate_lower:
            return RelationshipType.LOCATED_IN

        # Default - return None for unmapped predicates
        else:
            logger.debug(f"No mapping found for predicate: {predicate}")
            return None

    def _parse_json_response(self, response_content: str,
                             chunk_id: str) -> Optional[Dict]:
        """Parse and clean JSON response from LLM"""
        if not response_content or not response_content.strip():
            logger.warning(f"Empty response from LLM for chunk {chunk_id}")
            return None

        try:
            # First try direct parsing
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass

        try:
            # Try to extract JSON from markdown code blocks
            cleaned_content = self._clean_json_response(response_content)
            if cleaned_content:
                return json.loads(cleaned_content)
        except json.JSONDecodeError:
            pass

        try:
            # Try to fix common JSON issues
            fixed_content = self._fix_json_issues(response_content)
            if fixed_content:
                return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        logger.error(
            f"Failed to parse JSON response for chunk {chunk_id}. Response: {response_content[:200]}..."
        )
        return None

    def _clean_json_response(self, content: str) -> Optional[str]:
        """Clean JSON response by removing markdown formatting"""
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```.*$', '', content, flags=re.MULTILINE)

        # Find JSON object boundaries
        start_idx = content.find('{')
        if start_idx == -1:
            return None

        # Find matching closing brace
        brace_count = 0
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[start_idx:i + 1]

        return None

    def _fix_json_issues(self, content: str) -> Optional[str]:
        """Try to fix common JSON formatting issues"""
        try:
            # Remove any text before first {
            start_idx = content.find('{')
            if start_idx > 0:
                content = content[start_idx:]

            # Remove any text after last }
            end_idx = content.rfind('}')
            if end_idx != -1:
                content = content[:end_idx + 1]

            # Fix common issues
            content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
            content = re.sub(r',\s*]', ']',
                             content)  # Remove trailing commas in arrays

            return content
        except Exception:
            return None

    def _create_entity_from_extraction(self, entity_data: Dict,
                                       source_id: str) -> Optional[BaseEntity]:
        """Create entity object from extraction data"""
        try:
            entity_type_str = entity_data.get("type", "")
            entity_type = EntityType(entity_type_str)

            # Generate unique ID if not provided
            entity_id = entity_data.get("id") or self._generate_entity_id(
                entity_data.get("name", ""), entity_type_str)

            # Basic entity attributes (excluding entity_type since it's passed separately)
            attributes = {
                "id": entity_id,
                "name": entity_data.get("name", ""),
                "confidence_score": entity_data.get("confidence", 0.0),
                "source_document": source_id,
                "extracted_text": entity_data.get("extracted_text", "")
            }

            # Add type-specific attributes
            if "attributes" in entity_data:
                attributes.update(entity_data["attributes"])

            return create_entity(entity_type, **attributes)

        except Exception as e:
            logger.error(f"Error creating entity from extraction data: {e}")
            return None

    def _create_relationship_from_extraction(
            self, rel_data: Dict, source_id: str) -> Optional[Relationship]:
        """Create relationship object from extraction data"""
        try:
            rel_type_str = rel_data.get("type", "")
            rel_type = RelationshipType(rel_type_str)

            # Generate unique ID if not provided
            rel_id = rel_data.get("id") or self._generate_relationship_id(
                rel_data.get("source_entity", ""),
                rel_data.get("target_entity", ""), rel_type_str)

            return Relationship(
                id=rel_id,
                source_entity_id=rel_data.get("source_entity", ""),
                target_entity_id=rel_data.get("target_entity", ""),
                relationship_type=rel_type,
                confidence_score=rel_data.get("confidence", 0.0),
                properties=rel_data.get("properties", {}),
                source_document=source_id,
                extracted_text=rel_data.get("extracted_text", ""))

        except Exception as e:
            logger.error(
                f"Error creating relationship from extraction data: {e}")
            return None

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        content = f"{name}_{entity_type}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_relationship_id(self, source: str, target: str,
                                  rel_type: str) -> str:
        """Generate unique relationship ID"""
        content = f"{source}_{target}_{rel_type}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _deduplicate_entities(self,
                              entities: List[BaseEntity]) -> List[BaseEntity]:
        """Remove duplicate entities based on name and type"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _deduplicate_relationships(
            self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships"""
        seen = set()
        unique_relationships = []

        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id,
                   rel.relationship_type)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        return unique_relationships

    def analyze_ma_context(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for M&A context and organizational impact
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with M&A analysis results
        """
        try:
            prompt = self.ma_context_prompt.format(text=text)
            response = self.llm.invoke(prompt)

            # Use improved JSON parsing
            result = self._parse_json_response(response.content, "ma_context")
            if result:
                return result

        except Exception as e:
            logger.error(f"Error analyzing M&A context: {e}")

        # Return default structure on error
        return {
            "ma_activity": {
                "has_ma_content": False,
                "ma_score": 0.0
            },
            "organizational_impact": {},
            "key_insights": []
        }
