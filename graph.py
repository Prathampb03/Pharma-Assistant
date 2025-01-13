from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
import pprint

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field


import os

from langgraph.prebuilt import tools_condition

from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv('OLLAMA_NGROK_URL')


def get_embedding_function():
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    return embedding_function

"""RETRIEVER"""
# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=50
# )

# import os
# import json
# from langchain.schema import Document

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

# doc_splits = process_json_files_to_list('scrapped')
# documents = [Document(page_content=string) for string in doc_splits]
# doc_splits = text_splitter.split_documents(result_list)

# Add to vectorDB
persist_directory = "./chroma_db"

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=get_embedding_function(),
    persist_directory=persist_directory,  # Same directory used for saving
)
# vectorstore.persist()

retriever = vectorstore.as_retriever()



retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]


""" INIT """ 
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


""" Nodes """

    


""" Edges """

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    return "generate"
    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0,streaming=True,base_url=f"{base_url}/v1",api_key="ollama",max_tokens=2048,model="llama3.1")

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0,streaming=True,base_url=f"{base_url}/v1",api_key="ollama",max_tokens=2048,model="llama3.1")

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0,streaming=True,base_url=f"{base_url}/v1",api_key="ollama",max_tokens=2048,model="llama3.1")

    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    with open('statelog.txt', 'w', encoding='utf-8') as file:
        file.write(str(state))
    messages = state["messages"]
    question = messages[-2].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(temperature=0,streaming=True,base_url=f"{base_url}/v1",api_key="ollama",max_tokens=2048,model="llama3.1")


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
# prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

""" Graph Builder """

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()


## Run Graph

if __name__=="__main__":

    inputs = {
        "messages": [
            # ("user", "What is the primary use of Clindamycin?"),
            # ("user", "Tell me about Ketorolac"),

        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

