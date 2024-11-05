import os
import openai
import os
import magic
from openai import OpenAI
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

# file_path = os.path.join("data", "UBER", "UBER", "UBER_2022.html")
#
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"no such file: {file_path}")
# #
# 继续处理文件

os.environ["OPENAI_API_KEY"] = "xxx"
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(
    base_url="https://api.gptsapi.net/v1",
    api_key="xxx"
)

import nest_asyncio

nest_asyncio.apply()

from llama_index.readers.file import UnstructuredReader
from pathlib import Path


# ingest data
years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"data/UBER/UBER/UBER_{year}.html"), split_documents=False
    )
    #insert your meta data into each year
    for d in year_docs:
        d.metadata = {"years": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

# initialize simple vector indices
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings

Settings.chunk_size = 512
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")
    # but i think there will be some errors


# load indices from disk

from llama_index.core import load_index_from_storage

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{year}"
    )
    cur_index = load_index_from_storage(
        storage_context
    )
    index_set[year] = cur_index

# setting up a sub query engine to synthesis answers across 10k fillings
from llama_index.core.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"useful for when you want to answer queries about the {year} SEC 10-k for Uber",
        ),
    )
    for year in years
]

from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-4o-mini"),
)

query_engine_tools = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber"
    ),
)

# combine the Tools we define above into a single list of tools we defined above
tools = individual_query_engine_tools + [query_engine_tools]

from llama_index.agent.openai import OpenAIAgent
agent = OpenAIAgent.from_tools(tools,verbose=True)

response = agent.chat("hi, i am bob")
print(str(response))
