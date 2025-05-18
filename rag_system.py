import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# 设置API密钥（建议通过环境变量设置）
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-huggingface-token"

class RAGSystem:
    def __init__(self, 
                 embedding_model="openai", 
                 llm_model="openai", 
                 persist_directory="db",
                 document_dir="documents"):
        """初始化RAG系统"""
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory
        self.document_dir = document_dir
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vectordb = None
        self.qa_chain = None

    def _init_embeddings(self):
        """初始化嵌入模型"""
        if self.embedding_model.lower() == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_model.lower() == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise ValueError(f"不支持的嵌入模型: {self.embedding_model}")

    def _init_llm(self):
        """初始化大语言模型"""
        if self.llm_model.lower() == "openai":
            return OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        elif self.llm_model.lower() == "huggingface":
            return HuggingFaceHub(
                repo_id="google/flan-t5-xl",
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
        else:
            raise ValueError(f"不支持的LLM模型: {self.llm_model}")

    def load_documents(self):
        """加载并处理文档"""
        # 支持多种文档格式
        loader = DirectoryLoader(
            self.document_dir,
            glob="**/*.pdf",  # 可以扩展支持其他格式如 .txt, .docx 等
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_documents(documents)

    def create_vector_database(self):
        """创建向量数据库"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # 从持久化存储加载
            self.vectordb = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # 创建新的向量数据库
            documents = self.load_documents()
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectordb.persist()
        return self.vectordb

    def setup_qa_chain(self, with_memory=False):
        """设置问答链"""
        if self.vectordb is None:
            self.create_vector_database()
            
        # 自定义提示模板
        prompt_template = """使用以下上下文来回答问题。如果您不知道答案，请直接说您不知道，不要尝试编造答案。

        {context}

        问题: {question}
        回答:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # 创建检索器
        retriever = self.vectordb.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}  # 返回最相关的3个文档
        )
        
        if with_memory:
            # 带记忆功能的问答链
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )
        else:
            # 普通问答链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        return self.qa_chain

    def ask_question(self, question):
        """向RAG系统提问"""
        if self.qa_chain is None:
            self.setup_qa_chain()
            
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }

# 使用示例
if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = RAGSystem(
        embedding_model="openai",
        llm_model="openai",
        document_dir="documents",  # 文档目录
        persist_directory="db"     # 向量数据库持久化目录
    )
    
    # 设置问答链
    rag_system.setup_qa_chain(with_memory=True)
    
    # 提问示例
    question = "什么是检索增强生成?"
    response = rag_system.ask_question(question)
    
    print(f"问题: {question}")
    print(f"回答: {response['answer']}")
    
    # 打印来源
    print("\n来源文档:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"{i+1}. {doc.metadata.get('source', '未知来源')}")
        print(f"   位置: {doc.metadata.get('page', '?')}")    