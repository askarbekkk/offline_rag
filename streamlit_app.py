# streamlit_rag.py
import streamlit as st
import logging
from typing import Dict, List, Optional
import random
from initialize_rag import RAGInitializer


def needs_retrieval(query: str, chat_history: List[Dict]) -> bool:
    """
    Router: determine if this query needs document retrieval.

    Uses a simple heuristic-based approach to avoid extra LLM calls:
    - If query is a greeting, follow-up, clarification, or general knowledge → no retrieval
    - If query asks about specific facts, documents, or technical details → retrieval

    For 8GB RAM efficiency, this uses keyword/rules instead of an extra LLM call.
    """
    if not query or not query.strip():
        return False

    query_lower = query.strip().lower()

    # Patterns that DON'T need retrieval (general conversation)
    no_retrieval_patterns = [
        "hello",
        "hi ",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "what's up",
        "nice to meet",
        "thank",
        "thanks",
        "appreciate",
        "explain what you just said",
        "what did you mean",
        "clarify",
        "can you repeat",
        "say that again",
        "what is 2+2",
        "what's 2+2",
        "calculate",
        "who are you",
        "what can you do",
        "tell me about yourself",
        "i don't understand",
        "can you help",
        "yes",
        "no",
        "maybe",
        "okay",
        "ok",
        "what does that mean",
        "can you elaborate",
        "give me an example",
        "for example",
        "in other words",
        "simplify",
        "why",
        "how come",
        "what for",
        "tell me more",
        "continue",
        "go on",
        "summarize what you said",
        "summarize your answer",
    ]

    for pattern in no_retrieval_patterns:
        if query_lower.startswith(pattern) or query_lower == pattern:
            return False

    # If there's chat history and query is short/refers to previous context
    if chat_history and len(query.split()) < 5:
        # Short follow-up questions likely refer to previous answer
        reference_words = [
            "it",
            "that",
            "this",
            "they",
            "he",
            "she",
            "them",
            "его",
            "её",
            "это",
            "тот",
            "эти",
            "они",
            "그",
            "이",
            "저",
            "그것",
            "이것",
        ]
        words = query_lower.split()
        if any(word in reference_words for word in words):
            return False

    # Default: retrieve for anything that looks like a factual question
    return True


def process_query(
    query: str,
    retriever,
    reranker,
    response_generator,
    process_config: Dict,
    send_nb_chunks_to_llm=1,
    status=None,
    chat_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Process a single query through the complete RAG pipeline.

    This function orchestrates the query processing workflow:
    1. Router: decide if retrieval is needed
    2. If yes: retrieve → rerank → generate with context
    3. If no: generate directly from conversation history

    Args:
        query (str): The user's query
        retriever: Document retrieval component
        reranker: Result reranking component
        response_generator: Response generation component
        process_config (Dict): Processing configuration
        send_nb_chunks_to_llm (int): Number of chunks to send to LLM
        chat_history (Optional[List[Dict]]): Previous conversation messages

    Returns:
        Dict: Processing results containing:
            - Query: Original query
            - Response: Generated response
            - Score: Best retrieval/reranking score
            - Sources: Retrieved documents
            - Retrieval_used: Whether retrieval was performed
    """
    try:
        # Step 1: Router - decide if we need retrieval
        retrieval_needed = needs_retrieval(query, chat_history or [])

        if retrieval_needed:
            if status:
                status.write("🔍 Searching for relevant documents...")

            # Retrieve relevant documents
            if process_config["retrieval"]["use_bm25"]:
                retrieved_results = retriever.retrieve_with_method(
                    query,
                    method="hybrid",
                    top_k=process_config["retrieval"]["top_k"],
                )
            else:
                retrieved_results = retriever.retrieve_with_method(
                    query,
                    method="vector",
                    top_k=process_config["retrieval"]["top_k"],
                )
            logging.info(f"Retrieved {len(retrieved_results)} documents")
            if status:
                status.write(f"Found {len(retrieved_results)} documents")

            # Apply reranking if configured
            if process_config["retrieval"]["use_reranking"] and retrieved_results:
                if status:
                    status.write("📊 Reranking results...")
                reranked_results = reranker.rerank(
                    query,
                    [r.document for r in retrieved_results],
                    top_k=send_nb_chunks_to_llm,
                )
                relevant_docs = [r.document for r in reranked_results]
                best_score = reranked_results[0].score if reranked_results else 0.0
                logging.info(f"Reranked results. Best score: {best_score}")
            else:
                relevant_docs = [r.document for r in retrieved_results]
                best_score = retrieved_results[0].score if retrieved_results else 0.0
                logging.info(f"Using retrieval scores. Best score: {best_score}")
        else:
            # No retrieval needed - answer from conversation history
            if status:
                status.write("💬 Answering from conversation...")
            relevant_docs = []
            best_score = 0.0
            logging.info("No retrieval needed, answering from history")

        # Generate final response
        if status:
            status.write("✍️ Generating answer...")

        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={
                "retrieval_score": best_score,
                "retrieval_used": retrieval_needed,
            },
            chat_history=chat_history,
        )

        return {
            "Query": query,
            "Response": response_data["response"],
            "Score": best_score,
            "Sources": relevant_docs,
            "Retrieval_used": retrieval_needed,
        }

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {
            "Query": query,
            "Response": "An error occurred processing your query.",
            "Score": 0.0,
            "Sources": [],
            "Retrieval_used": False,
        }


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_components" not in st.session_state:
        st.session_state.rag_components = None


def display_chat_message(
    role: str,
    content: str,
    sources: list = None,
    score: float = None,
    retrieval_used: bool = None,
):
    """Display a chat message with optional sources and confidence score."""
    with st.chat_message(role):
        # Convert the text to proper Markdown format
        formatted_content = (
            content.replace("\n-", "\n\n-")  # Add extra newline before list items
            .replace("\n ", "\n")  # Remove extra spaces at line starts
            .strip()  # Remove extra whitespace
        )

        # Use markdown to render the formatted text
        st.markdown(formatted_content)

        if sources:
            with st.expander("View Sources Used"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"**Source {idx}:**")
                    # Create a scrollable text area with fixed height
                    st.text_area(
                        label=f"Source {idx} content",
                        value="From : "
                        + source.metadata.get("source", "unknown")
                        + "\n\nContent : \n"
                        + source.page_content,
                        height=200,
                        label_visibility="collapsed",
                        key=f"source_{role}_{idx}_{hash(source.page_content + str(random.random() * 1000000))}",
                        disabled=False,
                    )
        if score is not None and score > 0:
            # Normalize score to be between 0 and 1
            normalized_score = max(0.0, min(abs(score), 1.0))
            st.progress(normalized_score, text=f"Confidence: {normalized_score:.2%}")

        if retrieval_used is not None:
            if retrieval_used:
                st.caption("📄 Answered from documents")
            else:
                st.caption("💬 Answered from conversation")


def main():
    st.title("RAG Chat System")

    # Initialize session state
    initialize_session_state()

    # Initialize RAG components if not already done
    if st.session_state.rag_components is None:
        with st.spinner("Initializing RAG system..."):
            try:
                initializer = RAGInitializer(
                    "config/init_config.yaml", "config/process_config.yaml"
                )
                components = initializer.initialize()

                components.retriever.initialize(components.original_chunks)

                st.session_state.rag_components = components
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
                return

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources"),
            score=message.get("score"),
            retrieval_used=message.get("retrieval_used"),
        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message
        display_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process query and display response
        with st.status("Processing your question...", expanded=True) as status:
            result = process_query(
                prompt,
                st.session_state.rag_components.retriever,
                st.session_state.rag_components.reranker,
                st.session_state.rag_components.response_generator,
                st.session_state.rag_components.process_config,
                st.session_state.rag_components.process_config["retrieval"][
                    "send_nb_chunks_to_llm"
                ],
                status=status,
                chat_history=st.session_state.messages,
            )
            status.update(label="Done", state="complete", expanded=False)

        display_chat_message(
            role="assistant",
            content=result["Response"],
            sources=result["Sources"],
            score=result["Score"],
            retrieval_used=result.get("Retrieval_used"),
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["Response"],
                "sources": result["Sources"],
                "score": result["Score"],
                "retrieval_used": result.get("Retrieval_used"),
            }
        )

    # Sidebar with system information
    with st.sidebar:
        st.header("System Information")
        st.write("Retrieval Settings:")
        st.write(
            "- Top N:",
            st.session_state.rag_components.process_config["retrieval"][
                "send_nb_chunks_to_llm"
            ],
        )
        st.write(
            "- Reranking:",
            st.session_state.rag_components.process_config["retrieval"][
                "use_reranking"
            ],
        )
        st.write(
            "- Model:",
            st.session_state.rag_components.init_config["ollama"]["model_name"],
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
