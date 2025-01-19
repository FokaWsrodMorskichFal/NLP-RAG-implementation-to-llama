from wikirag import WikiRAG
from langchain_ollama import ChatOllama

num_doc_retrieved = 3
basic_wikirag = WikiRAG(db_name = "./databases/Russia_2023_2024_full_4_table_rows_db", model_name="llama3.1:8b",documents_retrieved=num_doc_retrieved)
base_model = ChatOllama(model = basic_wikirag.model_name)
#ollama31_8b = ChatOllama(model = "llama3.1:8b")
question = "When, where and how died Yevgeny Prigozhin?"
#"Who won Men's 100 metre butterfly? Elaborate on the winner and his competitors"
answer = basic_wikirag.query(question)
print(answer)