import os
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import TransformChain
import vector_database
import embedding
import wiki


class WikiRAG:
    def __init__(self, 
                 db_name: str = "wiki_db",
                 model_name: str = "llama3.1:8b",
                 documents_retrieved: int = 4):
        self.db = vector_database.VectorDatabaseWraper(db_name)
        self.model_name = model_name
        self.documents_retrieved = documents_retrieved

    def db_search(self, input_: str):
        emb = embedding.embedding([input_])[0]
        search_results, likeness = self.db.search(emb, self.documents_retrieved)
        return search_results, likeness

    def wikipedia_retriever(self, input_: str) -> str:
            search_results, _ = self.db_search(input_)
            documents = []
            for result in search_results:
                section_text = wiki.get_section_text(result['page_title'], 
                                                    result['section_title'],
                                                    result['subsection_title'],
                                                    result['subsubsection_title'],
                                                    result['part'])
                documents.append(section_text)
            retrievals = "\n".join(documents)
            return retrievals
    
    def query(self, question):
    # Template for the prompt
        template_RAG = "Your training data is up to march 2023. You are given a context describing events in 2023 and 2024. Answer the question based on the context. If the context doesn't involve reliable information say so. Context: {context} \n Question: {question}"
        
        # Preparing the prompt using the template
        prompt = template_RAG.format(context=self.wikipedia_retriever(question), question=question)
        
        # Initialize the model (ChatOllama) with the model name
        model = ChatOllama(model=self.model_name)
        
        # Run the model with the generated prompt
        response = model.invoke(prompt)
        
        return response
