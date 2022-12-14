# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import glob
import json
import os
import time
from typing import Optional, Union
import os 
import numpy as np
import pandas as pd

import uvicorn
from fastapi import APIRouter, FastAPI, Response
from schema import ModelSchema
from serving.models.unixcoder import UniXCoderEmbedder

app = FastAPI()
controller = APIRouter()

preloaded_models = {}


base_model = os.environ['base_model']

@app.on_event("startup")
def startup_event():
    print("downloading wrapped class for finetuned embedding models-")
    preloaded_models["code_search_handler"] = UniXCoderEmbedder(base_model)
    pass


@controller.get("/ping")
def ping():
    """SageMaker required method, ping heartbeat"""
    return Response(status_code=200)


@controller.post("/invocations", status_code=200)
async def transformation(payload: ModelSchema):
    """
    Make an inference on a set of code or query snippits to return a set of embeddings

    Parameters
    ----------
    payload - Pydantic.BaseClass:
        a validated json object containing source code and queries that embeddings need to be returned for

    Returns
    -------
    predictions : dict:
        a dictionary object with embeddings for all queries and code snippits that are parsed in the request
    """
    model = preloaded_models["code_search_handler"]
    if payload.language not in model.allowed_languages:
        response_msg = f"Language currently unsupported. Supported language types are {model.allowed_languages}, got {payload.language}"
        return Response(
            response_msg,
            status_code=400,
            media_type="plain/text",
        )

    if payload.task == "embedding":
        predictions = model.make_embeddings(
            code_batch=payload.code_snippit,
            query_batch=payload.query,
            language=payload.language,
        )
        return Response(content=json.dumps(predictions), media_type="application/json")

    else:
        return Response(
            "Task currently unsupported. Supported task types are embedding, got task {}".format(
                payload.task
            ),
            status_code=400,
            media_type="plain/text",
        )


@controller.post("/stream_invocations", status_code = 200)
async def transformation(payload : ModelSchema):
    """
    Make an inference on a set of code or query snippits to return a set of embeddings

    Parameters 
    ----------
    payload - Pydantic.BaseClass:
        a validated json object containing source code and queries that embeddings need to be returned for 
    
    Returns 
    -------
    predictions : dict:
        a dictionary object with embeddings for all queries and code snippits that are parsed in the request
    """
    model = preloaded_models['code_search_handler']
    if payload.language not in ["python", "javascript", "go"]: 
        return Response("Language currently unsupported. Supported task types are python, go, javascript- got task {}".format(payload.language), status_code = 400, media_type = "plain/text")


    if payload.task == "embedding": 
        predictions = model.make_embeddings(
            code_batch = payload.code_snippit, query_batch = payload.query, language = payload.language
        )
        return Response(content= json.dumps(predictions),
                    media_type="application/json")

    else:
        return Response("Task currently unsupported. Supported task types are embedding, got task {}".format(payload.task), status_code = 400, media_type = "plain/text")



app.include_router(controller)


if __name__ == "__main__":
    uvicorn.run(app=app, port=8080)
