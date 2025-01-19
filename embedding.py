from langchain_openai import OpenAIEmbeddings
#from langchain_openai import embed_documents
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


def embedding(texts: list[str]) -> np.ndarray:
    return np.array([OpenAIEmbeddings(
        api_key = os.getenv("OPENAI_API_KEY")
    ).embed_query(text) for text in texts])

def embed_docs(docs):
    return np.array(
        [
            OpenAIEmbeddings(
                    api_key = os.getenv("OPENAI_API_KEY")
            ).embed_documents(docs)
        ]
    )