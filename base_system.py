"""
Base class for RAG systems to eliminate code duplication.
"""
import logging
import tempfile
import uuid
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRAGSystem:
    """
    Base class for RAG systems with shared functionality.
    """
    
    def __init__(self):
        """Initialize base system components."""
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.vectorstore = None
        self.retriever = None
    
    def _validate_urls(self, urls: List[str]) -> List[str]:
        """Validate and limit URLs."""
        if not urls:
            urls = config.SAMPLE_URLS
        
        # Validate URL format
        valid_urls = []
        for url in urls:
            if url.strip() and (url.startswith('http://') or url.startswith('https://')):
                valid_urls.append(url.strip())
            else:
                logger.warning(f"Invalid URL format: {url}")
        
        # Limit number of URLs
        if len(valid_urls) > config.MAX_URLS:
            valid_urls = valid_urls[:config.MAX_URLS]
            logger.warning(f"Limited to {config.MAX_URLS} URLs for demo purposes")
        
        return valid_urls
    
    def _load_documents_from_urls(self, urls: List[str]) -> List:
        """Load documents from URLs with proper error handling."""
        docs = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} documents from {url}")
                
            except Exception as e:
                logger.error(f"Failed to load {url}: {e}")
                continue
        
        if not docs:
            raise ValueError(config.Constants.ERROR_MESSAGES['LOAD_FAILED'])
        
        return docs
    
    def _split_and_limit_documents(self, docs: List) -> List:
        """Split documents into chunks and apply limits."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # Limit number of chunks for demo performance
        if len(doc_splits) > config.MAX_CHUNKS:
            doc_splits = doc_splits[:config.MAX_CHUNKS]
            logger.warning(f"Limited to {config.MAX_CHUNKS} chunks for demo performance")
        
        return doc_splits
    
    def _create_vector_store(self, doc_splits: List, collection_prefix: str) -> None:
        """Create vector store with proper settings."""
        persist_directory = tempfile.mkdtemp()
        collection_name = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K_RETRIEVAL}
        )
    
    def load_documents(self, urls: Optional[List[str]] = None) -> int:
        """
        Load documents from URLs and create vector store.
        
        Args:
            urls: List of URLs to load documents from
            
        Returns:
            Number of document chunks created
            
        Raises:
            ValueError: If no valid URLs provided or documents can't be loaded
        """
        try:
            # Validate and limit URLs
            valid_urls = self._validate_urls(urls or [])
            
            if not valid_urls:
                raise ValueError("No valid URLs provided")
            
            # Load documents from URLs
            docs = self._load_documents_from_urls(valid_urls)
            
            # Split documents and apply limits
            doc_splits = self._split_and_limit_documents(docs)
            
            # Create vector store
            collection_prefix = self.__class__.__name__.lower().replace('system', '')
            self._create_vector_store(doc_splits, collection_prefix)
            
            logger.info(f"Successfully created vector store with {len(doc_splits)} chunks")
            return len(doc_splits)
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to load documents: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_question(self, question: str) -> str:
        """Validate and clean question input."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        
        if len(question) > config.Constants.MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long (max {config.Constants.MAX_QUESTION_LENGTH} characters)")
        
        return question
    
    def _create_error_response(self, error_msg: str, system_type: str) -> dict:
        """Create standardized error response."""
        return {
            "answer": f"Error: {error_msg}",
            "retrieved_documents": [],
            "num_documents": 0,
            "system_type": system_type,
            "error": error_msg
        }
