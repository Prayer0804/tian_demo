import os
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document

# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ["OPENAI_API_KEY"] = "your key here"

index = None
lock = Lock()


def initialize_index():
    global index

    with lock:
        # same as before ...
        pass


def query_index(query_text):
    global index
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response)


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("query_index", query_index)
    server = manager.get_server()

    print("starting server...")
    server.serve_forever()