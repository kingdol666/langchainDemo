from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.retrieval_qa.base import RetrievalQA # Replaced by agent, but now also used for a separate QA chain
from langchain.prompts import PromptTemplate # New import for custom prompt
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent # New import for agent
from langchain_core.tools import Tool # New import for tool
from langchain import hub # New import for prompt hub
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader # For TXT
# from langchain_community.document_loaders import UnstructuredFileLoader # For Word, if available, requires 'unstructured' package
# from langchain_community.document_loaders import Docx2txtLoader # Alternative for Word, requires 'docx2txt' package

load_dotenv() # Corrected call


modelscope_api_key = os.getenv("MODELSCOPE_API_KEY")
modelscope_base_url = os.getenv("MODELSCOPE_BASE_URL")
multimodal_model_name = "Qwen/Qwen2.5-VL-72B-Instruct" # 确保这是 ModelScope 上正确的模型ID

# NEW: Define the folder for documents
DOCUMENTS_DIR = "data_for_rag" # Or make this configurable
# Create the directory if it doesn't exist and add sample files for initial run
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    print(f"Created documents directory: {os.path.abspath(DOCUMENTS_DIR)}")
    # Add sample files to guide the user
    with open(os.path.join(DOCUMENTS_DIR, "sample_rag_data.txt"), "w", encoding="utf-8") as f:
        f.write("This is sample text for LangChain. LangChain is a framework for developing applications powered by language models. "
                "It enables applications that are data-aware, agentic, and can reason.")
    with open(os.path.join(DOCUMENTS_DIR, "another_rag_document.txt"), "w", encoding="utf-8") as f:
        f.write("This is another sample text about RAG. Retrieval Augmented Generation (RAG) is a technique to improve LLM responses "
                "by grounding them in external knowledge. This helps reduce hallucinations and provides up-to-date information.")
    print(f"Added sample files to {DOCUMENTS_DIR}. Please add your PDF and TXT files here for the RAG system.")
    print("To load Word (.docx) files, you might need to install 'unstructured' or 'python-docx' and uncomment the relevant loader code.")


# NEW/MODIFIED: Function to load documents from a folder
def load_all_documents_from_folder(folder_path: str) -> list[Document]:
    """
    Loads documents from PDF and TXT files in the specified folder.
    Handles potential errors during loading and ensures a non-empty list is returned for FAISS.
    """
    print(f"Attempting to load documents from: {os.path.abspath(folder_path)}")
    loaded_documents = []

    # For TXT files
    try:
        txt_loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}, # Specify encoding for text files
            show_progress=True,
            use_multithreading=True,
            silent_errors=True # Log errors instead of raising for individual files
        )
        print("Loading TXT files...")
        txt_docs = txt_loader.load()
        if txt_docs:
            loaded_documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} TXT documents.")
        else:
            print("No TXT documents found or loaded.")
    except Exception as e:
        print(f"Error initializing or loading TXT files from {folder_path}: {e}")

    # For PDF files
    try:
        pdf_loader = DirectoryLoader(
            folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        print("Loading PDF files...")
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            loaded_documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} PDF documents.")
        else:
            print("No PDF documents found or loaded.")
    except Exception as e:
        print(f"Error initializing or loading PDF files from {folder_path}: {e}")

    # Placeholder for Word document loading (requires 'unstructured' or 'python-docx')
    # try:
    #     # Option 1: UnstructuredFileLoader (more general, handles .doc, .docx, etc.)
        # from langchain_community.document_loaders import UnstructuredFileLoader
        # print("Attempting to load DOCX/DOC files using UnstructuredFileLoader...")
        # word_loader_cls = UnstructuredFileLoader
    #
    #     # Option 2: Docx2txtLoader (specifically for .docx)
        from langchain_community.document_loaders import Docx2txtLoader
        print("Attempting to load DOCX files using Docx2txtLoader...")
        word_loader_cls = Docx2txtLoader
    
        word_loader = DirectoryLoader(
            folder_path,
            glob="**/*.docx", # Adjust glob if using UnstructuredFileLoader for .doc as well
            loader_cls=word_loader_cls,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        word_docs = word_loader.load()
        if word_docs:
            loaded_documents.extend(word_docs)
            print(f"Loaded {len(word_docs)} Word documents.")
        else:
            print("No Word documents found or loaded.")
    except ImportError:
        print("Skipping Word document loading: 'unstructured' or 'docx2txt' (and their dependencies) not installed. "
              "To enable, install the required library (e.g., `pip install unstructured` or `pip install docx2txt`) and uncomment the code.")
    except Exception as e:
        print(f"Error initializing or loading Word files from {folder_path}: {e}")

    print(f"Total documents loaded before filtering empty ones: {len(loaded_documents)}")
    # Filter out documents that might be empty or just whitespace
    loaded_documents = [doc for doc in loaded_documents if doc.page_content and doc.page_content.strip()]
    print(f"Total non-empty documents loaded: {len(loaded_documents)}")

    if not loaded_documents:
        print(f"Warning: No non-empty documents found or loaded from {folder_path}. "
              "The RAG system will use placeholder content. Please check the directory and file types.")
        # Add a default document to prevent FAISS from failing with empty docs
        # and to provide some content for the RAG tool.
        loaded_documents.append(Document(page_content="No actual documents were loaded from the specified folder. "
                                                     "This is placeholder content. Please add PDF or TXT files to the 'data_for_rag' folder."))
    return loaded_documents

# 1. Load documents from the folder
all_raw_docs = load_all_documents_from_folder(DOCUMENTS_DIR)

# 2. Split documents
docs = []
# Ensure all_raw_docs is not empty (it should contain at least a placeholder if loading failed)
if all_raw_docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_raw_docs)
    print(f"Total chunks after splitting: {len(docs)}")

# This check might be redundant if load_all_documents_from_folder guarantees a non-empty list,
# but it's a good safeguard.
if not docs:
    print("Critical Warning: No document chunks to process for FAISS, even after placeholder logic. "
          "Using a final emergency placeholder.")
    docs = [Document(page_content="Emergency placeholder: Document loading and splitting failed to produce any processable chunks.")]

# 3. 使用 Hugging Face 嵌入模型
#    默认从 Hugging Face Hub 加载 sentence-transformers/all-MiniLM-L6-v2
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 1. 初始化 LLM
llm = ChatOpenAI(
    openai_api_key=modelscope_api_key,
    openai_api_base=modelscope_base_url,
    model_name=multimodal_model_name,
    temperature=0.8 # 对多模态任务通常希望结果更客观准确
)

# 1. 初始化 LLM
llm = ChatOpenAI(
    openai_api_key=modelscope_api_key,
    openai_api_base=modelscope_base_url,
    model_name=multimodal_model_name,
    temperature=0.8 # 对多模态任务通常希望结果更客观准确
)

print("Creating FAISS vector store...")
vector_store = FAISS.from_documents(docs, hf_embeddings)
print("FAISS vector store created successfully.")

# 2. 将 RAG 链定义为一个工具
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def retrieval_qa_tool_function(query: str) -> str:
    """当需要回答关于 LangChain 或 RAG 的问题时，使用此工具。"""
    # Re-implementing a simplified RAG logic for the tool
    # In a more complex scenario, you might use a pre-built chain here
    docs_retrieved = retriever.invoke(query)
    # For simplicity, just concatenating page_content. 
    # A real RAG tool would pass this to an LLM for synthesis.
    # However, the agent itself will use an LLM, so this might be sufficient for some cases.
    # Or, you could use a RetrievalQA chain internally here if needed.
    context = "\n".join([doc.page_content for doc in docs_retrieved])
    # The agent will use its LLM to answer based on this context
    # For now, let's return the context directly, or a simple answer
    # To make it more like a RAG chain, we'd need another LLM call here.
    # Let's keep it simple and let the agent decide based on retrieved docs.
    # A more complete RAG tool would look like:
    # from langchain.chains.retrieval_qa.base import RetrievalQA
    # rag_chain_internal = RetrievalQA.from_chain_type(
    #     llm=llm, # or a dedicated LLM for the tool
    #     chain_type="stuff",
    #     retriever=retriever,
    # )
    # return rag_chain_internal.run(query)
    # For now, returning the retrieved context for the agent to process
    if docs_retrieved:
        return f"根据检索到的信息： {context}"
    return "无法找到相关信息。"

rag_tool = Tool(
    name="LangChain_RAG_Search",
    func=retrieval_qa_tool_function,
    description="当需要回答关于 LangChain 或 RAG 的问题时，使用此工具。它会检索相关信息来帮助回答问题。"
)

tools = [rag_tool]

# 3. 创建 Agent
# Pull the ReAct prompt
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# 4. 创建 Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# 5. 修改提问函数以使用 Agent Executor
def ask_agent(question: str) -> str: # Renamed to ask_agent to avoid conflict
    response = agent_executor.invoke({"input": question})
    return response.get("output", "无法获取到智能体的回答。")

# 6. 创建传统的 RetrievalQA 链
print("Creating RetrievalQA chain...")

# 定义自定义提示模板
custom_prompt_template = """使用以下上下文来回答问题。如果您不知道答案，请直接说“不知道”，不要尝试编造答案。

上下文: {context}

问题: {question}
回答:"""

CUSTOM_QA_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Common chain type, can be adjusted
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}, # Apply custom prompt
    return_source_documents=True # Optional: to see which documents were retrieved
)
print("RetrievalQA chain created successfully.")

# 7. 定义使用 RetrievalQA 链的提问函数
def ask_qa_chain(question: str) -> dict:
    response = qa_chain.invoke({"query": question})
    return response

# Old load_document function (def load_document(): ...) is now removed 
# as its functionality is replaced by load_all_documents_from_folder introduced earlier.

if __name__ == "__main__":
    print("--- Testing Agent Executor ---")
    q1_agent = "什么是 RAG？"
    print("Q (Agent):", q1_agent)
    print("A (Agent):", ask_agent(q1_agent))

    q2_agent = "LangChain 库的主要用途是什么？"
    print("\nQ (Agent):", q2_agent)
    print("A (Agent):", ask_agent(q2_agent))

    q3_agent = "你好吗？" # Test a non-RAG question for agent
    print("\nQ (Agent):", q3_agent)
    print("A (Agent):", ask_agent(q3_agent))

    print("\n--- Testing RetrievalQA Chain ---")
    q1_qa = "什么是 RAG？请用检索到的信息回答。"
    print("Q (QA Chain):", q1_qa)
    qa_response1 = ask_qa_chain(q1_qa)
    print("A (QA Chain):", qa_response1.get("result"))
    # print("Source Documents:", qa_response1.get("source_documents"))

    q2_qa = "LangChain 库的主要用途是什么？请基于文档回答。"
    print("\nQ (QA Chain):", q2_qa)
    qa_response2 = ask_qa_chain(q2_qa)
    print("A (QA Chain):", qa_response2.get("result"))
    # print("Source Documents:", qa_response2.get("source_documents"))

    # q3_qa = "你好吗？" # This question is not suitable for QA chain as it's not about the documents
    # print("\nQ (QA Chain):", q3_qa)
    # qa_response3 = ask_qa_chain(q3_qa)
    # print("A (QA Chain):", qa_response3.get("result"))