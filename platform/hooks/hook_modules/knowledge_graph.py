#!/usr/bin/env python3
"""
Knowledge Graph Module - Entity and Relation Types

This module contains knowledge graph patterns for semantic relationships.
Extracted from hook_utils.py for modular architecture.

Exports:
- Entity: Knowledge graph entity
- Relation: Entity relationships
- KnowledgeGraph: Graph container with operations

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Entity:
    """
    Knowledge graph entity.

    Represents a node in the knowledge graph with:
    - name: Unique identifier
    - entity_type: Classification (e.g., "person", "concept", "file")
    - observations: List of observations about this entity
    """
    name: str
    entity_type: str
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "entityType": self.entity_type,
            "observations": self.observations
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            entity_type=data.get("entityType", "unknown"),
            observations=data.get("observations", [])
        )

    def add_observation(self, observation: str) -> None:
        """Add an observation to this entity."""
        if observation not in self.observations:
            self.observations.append(observation)


@dataclass
class Relation:
    """
    Knowledge graph relation.

    Represents an edge between two entities:
    - from_entity: Source entity name
    - to_entity: Target entity name
    - relation_type: Type of relationship (e.g., "depends_on", "created_by")
    """
    from_entity: str
    to_entity: str
    relation_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "relationType": self.relation_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            from_entity=data["from"],
            to_entity=data["to"],
            relation_type=data.get("relationType", "related_to")
        )


@dataclass
class KnowledgeGraph:
    """
    Knowledge graph container.

    Manages entities and relations with graph operations:
    - Entity CRUD
    - Relation management
    - Graph traversal
    - Persistence
    """
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity."""
        if entity.name in self.entities:
            # Merge observations
            existing = self.entities[entity.name]
            for obs in entity.observations:
                existing.add_observation(obs)
        else:
            self.entities[entity.name] = entity

    def add_relation(self, relation: Relation) -> None:
        """Add a relation if not duplicate."""
        # Check for duplicates
        for r in self.relations:
            if (r.from_entity == relation.from_entity and
                r.to_entity == relation.to_entity and
                r.relation_type == relation.relation_type):
                return
        self.relations.append(relation)

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)

    def get_related_entities(self, name: str, relation_type: Optional[str] = None) -> List[Entity]:
        """Get entities related to the given entity."""
        related: Set[str] = set()
        for r in self.relations:
            if r.from_entity == name:
                if relation_type is None or r.relation_type == relation_type:
                    related.add(r.to_entity)
            elif r.to_entity == name:
                if relation_type is None or r.relation_type == relation_type:
                    related.add(r.from_entity)
        return [self.entities[n] for n in related if n in self.entities]

    def search_entities(self, query: str) -> List[Entity]:
        """Search entities by name or observation content."""
        query_lower = query.lower()
        results = []
        for entity in self.entities.values():
            if query_lower in entity.name.lower():
                results.append(entity)
            elif any(query_lower in obs.lower() for obs in entity.observations):
                results.append(entity)
        return results

    def get_entity_types(self) -> Set[str]:
        """Get all unique entity types."""
        return {e.entity_type for e in self.entities.values()}

    def get_relation_types(self) -> Set[str]:
        """Get all unique relation types."""
        return {r.relation_type for r in self.relations}

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary format."""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls()
        for entity_data in data.get("entities", []):
            graph.add_entity(Entity.from_dict(entity_data))
        for relation_data in data.get("relations", []):
            graph.add_relation(Relation.from_dict(relation_data))
        return graph

    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge another graph into this one."""
        for entity in other.entities.values():
            self.add_entity(entity)
        for relation in other.relations:
            self.add_relation(relation)


# Export all symbols
__all__ = [
    "Entity",
    "Relation",
    "KnowledgeGraph",
]
