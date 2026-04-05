from typing import Annotated as A

from nano_vectordb import NanoVectorDB

import src.config.settings as config
from src.llm.openai import AzureOpenAIEmbeddingService
from src.utils.schema import doc as D


def clip_search_tool(
    database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
    event_description: A[str, D("A textual description of the event to search for.")],
    top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16,
) -> str:
    """
    Searches for events in a video clip database based on a given event description and retrieves the top-k most relevant video clip captions.

    Returns:
        str: A formatted string containing the concatenated captions of the searched video clip scripts.

    Notes:
        - This function utilizes the OpenAI Embedding Service to generate embeddings for the input text.
        - Use default values for `top_k` to limit the number of results returned.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=[event_description],
        api_key=config.OPENAI_API_KEY,
    )[0]['embedding']
    results = database.query(query_emb, top_k=top_k)
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    return "Here is the searched video clip scripts:\n\n" + captions
