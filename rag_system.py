"""
Basic RAG (Retrieval-Augmented Generation) implementation.
"""
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base_system import BaseRAGSystem
import config

logger = logging.getLogger(__name__)


class BasicRAGSystem(BaseRAGSystem):
    """
    A basic RAG system that retrieves documents and generates answers.
    """
    
    def __init__(self):
        super().__init__()
        # Use OpenRouter for chat completions
        self.llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL
        )
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Set up the RAG prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and retrieved documents
        """
        try:
            # Validate input
            question = self._validate_question(question)
            
            if not self.retriever:
                return self._create_error_response(
                    config.Constants.ERROR_MESSAGES['NO_DOCS_LOADED'],
                    config.Constants.SYSTEM_TYPES['RAG']
                )
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Generate answer
            answer = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            logger.info(f"RAG query completed: {len(retrieved_docs)} docs retrieved")
            
            return {
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "num_documents": len(retrieved_docs),
                "system_type": config.Constants.SYSTEM_TYPES['RAG']
            }
            
        except ValueError as e:
            logger.error(f"RAG query validation error: {e}")
            return self._create_error_response(str(e), config.Constants.SYSTEM_TYPES['RAG'])
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return self._create_error_response(str(e), config.Constants.SYSTEM_TYPES['RAG'])
