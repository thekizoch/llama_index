import asyncio
from typing import Any, Dict, List, Literal, Tuple, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, Field, validator
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate


DEFAULT_ENTITY_PROPERTIES = {
    "PERSON": [
        {
            "property": "name",
            "type": "STRING",
        },
        {
            "property": "profession",
            "type": "STRING",
        },
        {
            "property": "role",
            "type": "STRING",
        },
    ],
    "PRODUCT": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
    "LOCATION": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
    "ORGANIZATION": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
}

DEFAULT_RELATION_PROPERTIES = {
    "HAS": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
    "WORKED_ON": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
    "USED_BY": [
        {
            "property": "name",
            "type": "STRING",
        }
    ],
}

DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT = PromptTemplate(
    "Give the following text, extract the knowledge graph according to the provided schema. "
    "Try to limit to the output {max_triplets_per_chunk} extracted paths.\n"
    "-------\n"
    "{text}\n"
    "-------\n"
)


# Convert the above dict schema into a list of triples
Triple = Tuple[str, str, str]
DEFAULT_VALIDATION_SCHEMA: List[Triple] = [
    ("PERSON", "HAS", "PRODUCT"),
    ("PERSON", "WORKED_ON", "PRODUCT"),
    ("PERSON", "USED_BY", "PRODUCT"),
]


class FullSchemaLLMPathExtractor(TransformComponent):
    """
    Extract nodes from a graph using a schema.

    Args:
        llm (LLM):
            The language model to use.
        entity_properties (Dict[str, Any]):
            The properties of entities to extract, if not provided, will use DEFAULT_ENTITY_PROPERTIES
        extract_prompt (Union[PromptTemplate, str], optional):
            The prompt to use for extracting entities. Defaults to None.
    """

    llm: LLM
    extract_prompt: Union[PromptTemplate, str]
    kg_schema_cls: Any
    entity_properties: Dict[str, Any]

    def __init__(
        self,
        llm: LLM,
        extract_prompt: Union[PromptTemplate, str] = None,
        entity_properties: Dict[str, Any] = DEFAULT_ENTITY_PROPERTIES,
        relation_properties: Dict[str, Any] = DEFAULT_RELATION_PROPERTIES,
        kg_schema_cls: Any = None,
    ) -> None:
        """Init params."""
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        possible_entities = Literal[tuple(entity_properties.keys())]
        entity_cls = create_model(
            "Entity",
            type=(
                possible_entities,
                Field(
                    ...,
                    description=(
                        "Entity in a knowledge graph. Only entities with types that are listed are valid:"
                        + str(possible_entities)
                    ),
                ),
            ),
            # name should already be a property of each entity type
        )

        possible_relations = Literal[tuple(DEFAULT_RELATION_PROPERTIES.keys())]
        relation_cls = create_model(
            "Relation",
            type=(
                possible_relations,
                Field(
                    ...,
                    description="Relation in a knowledge graph. Only relations with types that are listed are valid:"
                    + str(possible_relations),
                ),
            ),
        )

        triplet_cls = create_model(
            "Triplet", subject=entity_cls, relation=relation_cls, object=entity_cls
        )

        def validate(v: Any, values: Any) -> Any:
            """Validate all triplets."""
            passing_triplets = []
            for triplet in v:
                if (
                    triplet.subject.type in possible_entities
                    and triplet.relation.type in possible_relations
                    and triplet.object.type in possible_entities
                ):
                    passing_triplets.append(triplet)
                else:
                    raise ValueError(f"Invalid triplet: {triplet}")
            return passing_triplets

        root = validator("triplets", pre=True)(validate)
        print(f"root: {root}")
        kg_schema_cls = create_model(
            "KGSchema",
            __validators__={"validator1": root},
            triplets=(List[triplet_cls], ...),
        )

    @classmethod
    def class_name(cls) -> str:
        return "FullSchemaLLMPathExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract nodes from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract nodes from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            extracted_nodes = await self.llm.astructured_predict(
                output_cls=self.kg_schema_cls,
                prompt=self.extract_prompt,
                text=text,
            )
            nodes = self._extract_nodes(extracted_nodes)
        except ValueError:
            nodes = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])

        for entity_node in nodes:
            existing_nodes.append(entity_node)

        node.metadata[KG_NODES_KEY] = existing_nodes

        return node

    def _extract_nodes(self, extracted_nodes: List[Dict[str, Any]]) -> List[EntityNode]:
        """Extract nodes from the extracted data."""
        nodes = []
        for entity in extracted_nodes:
            entity_type = entity.get("type")
            entity_name = entity.get("name")
            if entity_type and entity_name:
                nodes.append(EntityNode(label=entity_type, name=entity_name))
        return nodes

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract nodes from nodes async."""
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=4,
            show_progress=show_progress,
            desc="Extracting nodes from text with schema",
        )
