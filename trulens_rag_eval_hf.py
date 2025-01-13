import os
from trulens_eval import Tru, TruChain, Feedback
from trulens_eval.feedback.provider.hugs import Huggingface, HuggingfaceLocal
from trulens_eval.feedback.provider.langchain import Langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from graph import graph, retriever, retriever_tool
from typing import List, Dict, Any, Tuple
from huggingface_hub import login, hf_hub_download

from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv('OLLAMA_NGROK_URL')

os.environ["HUGGINGFACE_TOKEN"] = "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"
login(token="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")

# Initialize Hugging Face provider
# hf_provider = Huggingface(
#     # model_name="facebook/bart-medium-mnli",  # Example model for relevance
#     huggingfacehub_api_token="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"
# )

# hf_provider = HuggingfaceLocal()
# Define custom feedback functions

llm = ChatOpenAI(temperature=0, streaming=True,
                 base_url=f"{base_url}/v1", api_key="ollama", max_tokens=2048, model="llama3.1")


hf_provider = Langchain(llm)


def context_relevance(query: str, context: str) -> float:
    """
    Evaluate context relevance using Hugging Face model

    Args:
        query (str): Original query
        context (str): Retrieved context

    Returns:
        float: Relevance score (0-1)
    """
    return hf_provider.context_relevance(query, context)


def answer_relevance(query: str, response: str) -> float:
    """
    Evaluate answer relevance using Hugging Face model

    Args:
        query (str): Original query
        response (str): Generated response

    Returns:
        float: Relevance score (0-1)
    """
    return hf_provider.context_relevance(query, response)


# def hallucination_detection(context: str, response: str) -> float:
#     """
#     Detect potential hallucinations

#     Args:
#         context (str): Retrieved context
#         response (str): Generated response

#     Returns:
#         float: Hallucination score (0-1)
#     """
#     return 1.0 - hf_provider.hallucination_evaluator(response, context)


def build_evaluation_feedbacks():
    """
    Create TruLens feedback objects

    Returns:
        List of Feedback objects
    """
    return [
        Feedback(context_relevance)
        .on_input_output(),

        Feedback(answer_relevance)
        .on_input_output(),

        # Feedback(hallucination_detection)
        # .on_output()
    ]


def evaluate_rag_with_trulens(rag_chain, query: str, docs):
    """
    Evaluate RAG pipeline using TruLens and Hugging Face providers

    Args:
        rag_chain: LangChain or custom RAG pipeline
        query (str): Input query

    Returns:
        Dict: Evaluation results
    """
    tru = Tru()

    # Create feedback objects
    feedbacks = build_evaluation_feedbacks()

    # Wrap RAG chain for evaluation
    tru_recorder = TruChain(
        rag_chain,
        app_id="HuggingFace_RAG_Evaluation",
        # feedbacks=feedbacks
    )

    # Run evaluation
    with tru_recorder as recorder:
        result = rag_chain.invoke({"context": docs, "question": query})

    # Retrieve evaluation records
    records = tru.get_records_and_feedback(
        app_ids=["HuggingFace_RAG_Evaluation"]
    )

    return {
        "response": result,
        "context_relevance": context_relevance(query, docs),
        "answer_relevance": answer_relevance(query, result),
        # "hallucination_detection": hallucination_detection(docs, result)
    }

# Example usage


def main():
    # Assuming you have a RAG chain
    from langchain_openai import ChatOpenAI

    query = "Can I take Ibuprofen if I have a history of stomach ulcers?"
    query = "What is the primary use of Clindamycin?"

    docs = retriever_tool.invoke(query)

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(temperature=0, streaming=True,
                     base_url=f"{base_url}/v1", api_key="ollama", max_tokens=2048, model="llama3.1")

    # Post-processing

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    evaluation_results = evaluate_rag_with_trulens(rag_chain, query, docs)
    print(evaluation_results)


if __name__ == "__main__":
    main()
