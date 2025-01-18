from langchain_openai import OpenAIEmbeddings
#from langchain_openai import embed_documents
import numpy as np


def embedding(texts: list[str]) -> np.ndarray:
    return np.array([OpenAIEmbeddings(
        api_key=""
    ).embed_query(text) for text in texts])

def embed_docs(docs):
    return np.array(
        [
            OpenAIEmbeddings(
                    api_key=""
            ).embed_documents(docs)
        ]
    )