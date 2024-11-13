from typing import List
import llama_index
import openai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import faiss
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))
# Add the parent directory to the path since we work with notebooks

EMBED_DIMENSION = 512

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# load environment from a .env file
load_dotenv()

# the openai_api
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["BASE_URL"] = os.getenv('BASE_URL')

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["BASE_URL"]

# set embedding model on llama_index global settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)

# READ DOCS
path = "data/"
node_parser = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = node_parser.load_data()
print(documents[0])

# VECTOR STORE
faiss_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# TEXT CLEANER TRANSFORMATION
class TextCleaner(TransformComponent):
    """
    transformation to be used within the ingestion pipeline
    cleans clutters from texts
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        for node in nodes:
            node.text = node.text.replace('\t', ' ')
            node.text = node.text.replace('\n', ' ')

        return nodes


# INGESTION PIPELINE
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
    ],
    vector_store=vector_store,
)
# run pipeline and get generated nodes
nodes = pipeline.run(documents=documents)

# CREATE RETRIEVER
vector_store_index = VectorStoreIndex(nodes)
retriever = vector_store_index.as_retriever(similarity_top_k=2)

# TEST RETRIEVER jian_suo
def show_context(context):
    for i, c in enumerate(context):
        print(f"Context{i+1}:")
        print(c.text)
        print("\n")


test_query = "What is the main cause of climate change??"
context = retriever.retrieve(test_query)
show_context(context)

import json
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from evaluation.evalute_rag import create_deep_eval_test_cases
from enum import Enum

# Set llm model for evaluation of the question and answers
LLM_MODEL = "gpt-4o"

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=LLM_MODEL,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=LLM_MODEL,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=LLM_MODEL,
    include_reason=True
)


def evaluate_rag(query_engine, num_questions: int = 5) -> None:
    """
    Evaluate the RAG system using predefined metrics.

    Args:
        query_engine: Query engine to ask questions and get answers along with retrieved context.
        num_questions (int): Number of questions to evaluate (default: 5).
    """

    # Load questions and answers from JSON file
    q_a_file_name = "data/q_a.json"
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a = json.load(json_file)

    questions = [qa["question"] for qa in q_a][:num_questions]
    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]
    generated_answers = []
    retrieved_documents = []

    # Generate answers and retrieve documents for each question
    for question in questions:
        response = query_engine.query(question)
        context = [doc.text for doc in response.source_nodes]
        retrieved_documents.append(context)
        generated_answers.append(response.response)

    # Create test cases and evaluate
    test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevance_metric]
    )

query_engine  = vector_store_index.as_query_engine(similarity_top_k=2)
evaluate_rag(query_engine, num_questions=1)



