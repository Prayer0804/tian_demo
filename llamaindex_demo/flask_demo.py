from flask import Flask, request
from multiprocessing.managers import BaseManager
app = Flask(__name__)

manager = BaseManager(("", 5602), b"password")
manager.register("query_index")
manager.connect()

@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    response = manager.query_index(query_text)._getvalue()
    return str(response), 200


@app.route("/")
def home():
    return "hello world"

#
#
# # create interface
#
# from flask import request
#
# @app.route("/query", methods={"GET"})
# def query_index():
#     global index
#     query_text = request.args.get("text", None)
#     if query_text is None:
#         return (
#             "Not text found, Please include a ?text=blah parameter in the URL",
#             400,
#         )
#     query_engine = index.as_query_engine()
#     response = query_engine.query(query_text)
#     return str(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)