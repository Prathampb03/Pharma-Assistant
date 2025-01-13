from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def get_embedding_function():
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    return embedding_function

import os
import json
from langchain.schema import Document

# def json_to_string(json_obj):
#     """Convert a JSON object to the specified string format."""
#     return ",\n".join(f"{key}:{value}" for key, value in json_obj.items())

# def process_json_files_to_list(folder_path):
#     """Process all JSON files in a folder and return a list of formatted strings."""
#     result_list = []

#     # Iterate through all files in the folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json"):  # Process only JSON files
#             file_path = os.path.join(folder_path, filename)
            
#             # Read and process the JSON file
#             with open(file_path, 'r') as f:
#                 json_data = json.load(f)
#                 result_list.append(json_to_string(json_data))

#     return result_list

def process_json_files_to_list_keyval_level(folder_path):
    """Process all JSON files in a folder and return a list of formatted strings."""

    def json_to_string(json_obj):
        """Convert a JSON object to the specified string format."""
        return [f"{key}:{value}" for key, value in json_obj.items()]
    
    result_list = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(folder_path, filename)
            
            # Read and process the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                result_list.extend(json_to_string(json_data))
    return result_list



folder_path = 'altcleaned'
doc_splits = process_json_files_to_list_keyval_level(folder_path)
documents = [Document(page_content=string) for string in doc_splits]

persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=get_embedding_function(),
    persist_directory=persist_directory,  # Directory to persist the vectorstore
)

vectorstore.persist()
print("Chroma db saved")
