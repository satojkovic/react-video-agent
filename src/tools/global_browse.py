import json
from typing import Annotated as A

from nano_vectordb import NanoVectorDB

import src.config.settings as config
from src.llm.openai import AzureOpenAIEmbeddingService, call_openai_model_with_tools
from src.utils.schema import doc as D


def global_browse_tool(
    database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
    query: A[str, D("A textual description which will be used to search for relevant video clips in the database.")],
) -> str:
    """
    Analyzes a video database to answer a detailed question by first searching for relevant video clips and then generating a comprehensive answer based on their captions.

    Args:
        database (NanoVectorDB): The database object that supports querying with embeddings.
        query (str): A textual description or question to search for in the video.

    Returns:
        str: A JSON string containing the subject registry and the model's answer to the query based on the most relevant video clips.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=[query],
        api_key=config.OPENAI_API_KEY,
    )[0]['embedding']
    results = database.query(query_emb, top_k=config.GLOBAL_BROWSE_TOPK)
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    clip_captions = "Here is the searched video clip scripts:\n\n" + captions

    input_msgs = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant specializing in analyzing video content and providing detailed, insightful answers."
        },
        {
            "role": "user",
            "content": (
                "Below are descriptions of video clips, each with its corresponding timestamp. "
                "Carefully review the sequence of events, the details and movements of objects, and the actions and poses of people. "
                "Based on these observations, provide a thorough and specific answer to the following question, referencing key events and timestamps as appropriate.\n"
                "Question: {question}\n\n{clip_captions}"
            ),
        },
    ]
    input_msgs[1]['content'] = input_msgs[1]['content'].format(
        question=query, clip_captions=clip_captions
    )

    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")

    subject_registry = database.get_additional_data()['subject_registry']
    return json.dumps({'subject_registry': subject_registry, 'query_related_event': msgs["content"]})
