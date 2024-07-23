import asyncio
from typing import Any, Dict, List, Union, Literal
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.prompts import PromptTemplate

# from llama_index.core.llms.llm import LLM
from llama_index_dev.core.llms.llm import LLM

print(f"Imported LLM from: {LLM.__module__}")
from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import create_model, Field, validator

DEFAULT_ENTITY_PROPERTIES = {
    "PERSON": [
        {"property": "profession", "type": "STRING"},
        {"property": "role", "type": "STRING"},
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

# original from schema_llm.py
DEFAULT_SCHEMA_PATH_EXTRACT_PROMPT = PromptTemplate(
    "Give the following text, extract the knowledge graph according to the provided schema. "
    "Try to limit to the output {max_nodes_per_chunk} extracted paths.s\n"
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

        print(f"entity_properties: {entity_properties}")

        def validate_properties(v: Any) -> Dict[str, Any]:
            entity_type = v["type"]
            properties = entity_properties.get(entity_type, [])
            valid_props = {}
            for prop in properties:
                prop_name = prop["property"]
                if prop_name in v:
                    valid_props[prop_name] = v[prop_name]
            return valid_props

        root = validator("properties", allow_reuse=True, pre=True)(validate_properties)
        entity_cls = create_model(
            "Entity",
            type=(
                possible_entities,
                Field(
                    ...,
                    description=(
                        "Entity in a knowledge graph. Only extract entities with types that are listed as valid: "
                        + str(possible_entities)
                    ),
                ),
            ),
            name=(str, ...),
            properties=(  # currently nothing is being extracted. its always empty
                Dict[str, Any],
                Field(
                    default_factory=dict,
                    description=(
                        "Properties of the entity based on its type. Only include properties that are valid: "
                        + str(entity_properties)
                    ),
                ),
            ),
            __validators__={"validator1": root},
        )

        # below works as
        #     ticker=(
        #         Optional[str],
        #         Field(
        #             default=None,
        #             description="Ticker symbol of the organization, if available."
        #         )
        #     ),
        #     industry=(
        #         Optional[str],
        #         Field(
        #             default=None,
        #             description="Industry of the organization, if available."
        #         )
        #     ),
        #     sector=(
        #         Optional[str],
        #         Field(
        #             default=None,
        #             description="Sector of the organization, if available."
        #         )
        #     )
        # )

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
            )

            # print(f"Raw LLM output: {entities_schema}")
            # print(f"Type of entities_schema: {type(entities_schema)}")
            # print(f"Entities: {entities_schema.entities}")
            for entity in entities_schema.entities:
                print(f"Entity: {entity}")
                # print(f"Entity Type: {entity.type}")
                # print(f"Entity Name: {entity.name}")
                # print(f"Entity Properties: {entity.properties}")
                # print(f"Type of Entity Properties: {type(entity.properties)}")

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
        """ "Prune invalid entities from the knowledge graph."""
        assert isinstance(entities_schema, self.entities_schema_cls)

        valid_nodes = []
        for entity in entities_schema.entities:
            entity_type = entity.type
            entity_name = entity.name
            # entity_properties = entity.properties

            # Validate entity type
            if entity_type not in self.entity_properties:
                continue

            entity_node = EntityNode(
                label=entity_type,
                name=entity_name,
                # properties=entity_properties
            )
            valid_nodes.append(entity_node)

        return valid_nodes
