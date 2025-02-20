# main.py
import gradio as gr
from loaders import load_documents, split_documents
from vectorstore import create_vectorstore
from chatbot import setup_chatbot, is_rag_related


def main():
    print("Loading documents...")
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    print("Vector store created.")
    
    print("Setting up chatbot...")
    # chatbot, vectorstore_for_search, chatbot2 = setup_chatbot(vectorstore)
    chatbot, vectorstore_for_search = setup_chatbot(vectorstore)
    print("Chatbot is ready.")
    
    # def chat(question, history):
    #     if is_rag_related(question, vectorstore_for_search):
    #         result = chatbot.invoke({"question": question})
    #         print("Using RAG-based retrieval...")
    #         return result["answer"] if isinstance(result, dict) else str(result)
    #     else:
    #         # Use the base LLaMA model without retrieval
    #         result = chatbot2.invoke({"input": question})
    #         print("Response from non-retrieval chain:")
    #         return result["response"] if isinstance(result, dict) else str(result)
    
    def chat(question, history):
            result = chatbot.invoke({"question": question})
            return result["answer"] if isinstance(result, dict) else str(result)
        
    print("Launching chat interface...")
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)

if __name__ == "__main__":
    main()

