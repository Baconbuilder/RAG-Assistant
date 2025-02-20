# chatbot.py
from langchain_ollama import ChatOllama
from config import MODEL_NAME, TEMPERATURE
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.prompts import PromptTemplate
import re

def setup_chatbot(vectorstore, model=MODEL_NAME, temperature=TEMPERATURE):
    llm = ChatOllama(temperature=temperature, model=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are a helpful assistant that helps explain content from the documents. "
            "Provide accurate responses based on the available documents. "
            "If you don't know the answer, just say that you don't know. "
            "If a question is general (e.g., hi, thanks), respond politely without retrieving documents.\n\n"
            "Context: {context}\n\nQuestion: {question}"
        )
    )
    
    general_template = (
        "The following is a conversation between a human and an AI assistant. "
        "The AI provides detailed, accurate, and contextually grounded responses, "
        "similar to how it would if it had access to documents. "
        "If the AI does not know the answer, it should truthfully say so. "
        "Answer in a factual and helpful manner.\n\n"
        "Current conversation:\n"
        "{chat_history}\n"
        "Human: {input}\n"
        "AI Assistant:"
    )
    
    prompt_2 = PromptTemplate(input_variables=["chat_history", "input"], template=general_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        rephrase_question=False
        
    )
    
    non_retrieval_chain = ConversationChain(
        prompt=prompt_2,
        llm=llm, 
        memory=memory
        )
    
    return conversation_chain, vectorstore, non_retrieval_chain

def preprocess_query(query):
    # Remove leading courtesy phrases from the query.
    query = query.lower().strip()
    
    courtesy_pattern = r'^(thanks[!,.]*(\s+|$)|thank you[!,.]*(\s+|$)|hi[!,.]*(\s+|$)|hello[!,.]*(\s+|$)|hey[!,.]*(\s+|$))'
    # Remove courtesy phrases from the start.
    return re.sub(courtesy_pattern, '', query, flags=re.IGNORECASE)

def is_rag_related(question, vectorstore, similarity_threshold=0.3):
    # Checks if the question is related to stored documents by performing a similarity search that returns scores
    # If the top document has a similarity score above the threshold, it is considered rag related
    
    processed_question = preprocess_query(question)
    
    if not processed_question:
        # print("Query identified as courtesy-only. Skipping RAG.")
        return False
    
    # docs_with_scores = vectorstore.similarity_search_with_score(question)
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question)
    if docs_with_scores:
        doc, score = docs_with_scores[0]
        doc.metadata["similarity_score"] = score
        # print("Similarity score:", score)
        if score >= similarity_threshold:
            return True
    return False
