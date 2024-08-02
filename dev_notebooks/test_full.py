import nest_asyncio

nest_asyncio.apply()

from llama_index.graph_stores.neo4j import Neo4jPGStore

username = "neo4j"
password = "password"
url = "bolt://localhost:7687"

graph_store = Neo4jPGStore(username=username, password=password, url=url)

load_env()


from llama_index.core import Document
import pandas as pd

news = pd.read_csv("news.csv")
documents = [
    Document(text=f"{row['title']}: {row['text']}") for i, row in news.iterrows()
]

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

from full_schema_llm import FullSchemaLLMPathExtractor
from llama_index.core import PropertyGraphIndex

kg_extractor = FullSchemaLLMPathExtractor(
    llm=llm,
)

index = PropertyGraphIndex.from_documents(
    documents[:3],
    kg_extractors=[kg_extractor],
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)