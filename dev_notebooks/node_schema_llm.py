import asyncio
from typing import Any, Dict, List, Union, Literal, Optional
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, Field

DEFAULT_ENTITY_PROPERTIES = {
    "PERSON": [
        {"property": "profession", "type": "STRING"},
    ],
    "PRODUCT": [
        {"property": "price", "type": "STRING"},
    ],
    "LOCATION": [{"property": "country", "type": "STRING"}],
    "ORGANIZATION": [
        {"property": "ticker", "type": "STRING"},
        {"property": "industry", "type": "STRING"},
        {"property": "sector", "type": "STRING"},
    ],
}

DEFAULT_RELATION_PROPERTIES = {
    "WORKED_ON": [
        {"property": "role", "type": "STRING"},
    ],
    "HAS": [
        {"property": "duration", "type": "STRING"},
    ],
    "LOCATED_IN": [
        {"property": "since", "type": "STRING"},
    ],
}

DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT = PromptTemplate(
    "Given the following text, extract the knowledge graph according to the provided schema. "
    "For each extracted entity, include the following properties if available in the text, otherwise exclude them:\n"
    "-------\n"
    "{entity_properties}\n"
    "-------\n"
    "Try to limit the output to {max_nodes_per_chunk} extracted nodes.\n"
    "Text:\n"
    "-------\n"
    "{text}\n"
    "-------\n"
)


class NodeSchemaLLMPathExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    max_nodes_per_chunk: int
    num_workers: int
    entity_properties: Dict[str, List[Dict[str, str]]]
    entities_schema_cls: Any

    def __init__(
        self,
        llm: LLM,
        extract_prompt: Union[PromptTemplate, str] = None,
        entity_properties: Dict[str, List[Dict[str, str]]] = DEFAULT_ENTITY_PROPERTIES,
        entities_schema_cls: Any = None,
        max_nodes_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        possible_entities = Literal[tuple(entity_properties.keys())]

        def validate_properties(v: Any) -> Dict[str, Any]:
            entity_type = v["type"]
            properties = entity_properties.get(entity_type, [])
            valid_props = {}
            for prop in properties:
                prop_name = prop["property"]
                if prop_name in v:
                    valid_props[prop_name] = v[prop_name]
            return valid_props

        properties = {}
        for props in DEFAULT_ENTITY_PROPERTIES.values():
            for prop in props:
                properties[prop["property"]] = (Optional[str], None)

        entity_cls = create_model(
            "Entity",
            type=(
                possible_entities,
                Field(
                    ...,
                ),
            ),
            name=(str, ...),
            **properties,
        )

        entities_schema_cls = create_model(
            "EntitiesSchema",
            entities=(List[entity_cls], ...),
        )

        entities_schema_cls.__doc__ = "Entities Schema."

        super().__init__(
            llm=llm,
            extract_prompt=extract_prompt or DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT,
            entity_properties=entity_properties,
            entities_schema_cls=entities_schema_cls,
            max_nodes_per_chunk=max_nodes_per_chunk,
            num_workers=num_workers,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NodeSchemaLLMPathExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")

        try:
            entities_schema = await self.llm.astructured_predict(
                self.entities_schema_cls,
                self.extract_prompt,
                text=text,
                max_nodes_per_chunk=self.max_nodes_per_chunk,
                entity_properties=self.entity_properties,
            )

            nodes = self._prune_invalid_entities(entities_schema)
        except ValueError as e:
            print(f"error in node schema llm: {e}")
            nodes = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_nodes.extend(nodes)
        node.metadata[KG_NODES_KEY] = existing_nodes

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting nodes from text",
        )

    def _prune_invalid_entities(self, entities_schema: Any) -> List[EntityNode]:
        """Prune invalid entities and entity properties from the knowledge graph."""
        assert isinstance(entities_schema, self.entities_schema_cls)

        valid_nodes = []
        for entity in entities_schema.entities:
            entity_type = entity.type
            # Validate entity type
            if entity_type not in self.entity_properties:
                continue

            props = {}
            for prop_dict in self.entity_properties[entity_type]:
                prop_name = prop_dict["property"]
                if hasattr(entity, prop_name):
                    props[prop_name] = getattr(entity, prop_name)

            entity_node = EntityNode(
                label=entity_type, name=entity.name, properties=props
            )
            valid_nodes.append(entity_node)

        return valid_nodes
