import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

# NOTE: for local test only,do not deploy with your key hardcoded

os.environ["OPENAI_API_KEY"] = "xxx"

index = None

# initialized our index. if we call this just before starting the flask server in the main function
# then our index will be ready for users queries
def initialize_index():
    global index
    storage_context = StorageContext.from_defaults()
    index_dir = "./.index"
    if os.path.exists(index_dir):
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        storage_context.persist(index_dir)

