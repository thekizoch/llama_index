import nest_asyncio

nest_asyncio.apply()

from llama_index.graph_stores.neo4j import Neo4jPGStore

username = "neo4j"
password = "password"
url = "bolt://localhost:7687"

graph_store = Neo4jPGStore(username=username, password=password, url=url)

from utils import load_env
from llama_index.core import Document

import requests
from xml.etree import ElementTree


# Step 1: Search for articles
search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
search_params = {"db": "pubmed", "term": "prostatitis plant medicine", "retmax": 10}

response = requests.get(search_url, params=search_params)
root = ElementTree.fromstring(response.content)
pmids = [id_elem.text for id_elem in root.findall(".//Id")]

# Step 2: Fetch article details
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

response = requests.get(fetch_url, params=fetch_params)
root = ElementTree.fromstring(response.content)
abstracts = [abstract_elem.text for abstract_elem in root.findall(".//AbstractText")]

# # Step 3: Create Document objects, skipping None and where len(abstract) < 100
documents = [
    Document(text=abstract)
    for abstract in abstracts
    if abstract is not None and len(abstract) > 100
]

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_env()
llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import PropertyGraphIndex


# # Define entities and relations
from typing import Literal

possible_entities = Literal[
    "DRUG",
    "DISEASE",
    "BIOLOGICAL_PROCESS",
    "MOLECULAR_FUNCTION",
    "CELL_LINE",
    "SIGNALING_PATHWAY",
    "COMPOUND",
    "PLANT",
]

possible_relations = Literal[
    "TREATS", "HAS_EFFECT_ON", "INVOLVES", "EXPRESSED_IN", "PART_OF", "CONTAINS"
]

## strict=False

possible_entity_props = ["SYNONYMS", "SOURCE", "TOXICITY"]
possible_relation_props = ["EFFECT_STRENGTH", "EVIDENCE", "DOSAGE"]

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=20,
    strict=False,
    possible_entities=possible_entities,
    possible_entity_props=possible_entity_props,
    possible_relations=possible_relations,
    possible_relation_props=possible_relation_props,
    num_workers=4,
)

schema_index = PropertyGraphIndex.from_documents(
    documents[1:2],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True,
)