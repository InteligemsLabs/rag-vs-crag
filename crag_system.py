"""
CRAG (Corrective Retrieval-Augmented Generation) implementation.
"""
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_core.documents import Document
from base_system import BaseRAGSystem
import config

logger = logging.getLogger(__name__)


class RetrievalEvaluator(BaseModel):
    """Data model for document relevance evaluation."""
    binary_score: str = Field(
        description="Document relevance to the question, 'yes' or 'no'"
    )


class CRAGSystem(BaseRAGSystem):
    """
    CRAG system with document evaluation and corrective retrieval.
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
        self.web_search_tool = TavilyClient(api_key=config.TAVILY_API_KEY)
        self._setup_components()
    
    def _setup_components(self):
        """Set up CRAG components: evaluator, rewriter, and generator."""
        
        # Document relevance evaluator (use OpenRouter for chat completions)
        evaluator_llm = ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=0,
            openai_api_key=config.OPENROUTER_API_KEY,
            openai_api_base=config.OPENROUTER_BASE_URL
        )
        self.structured_evaluator = evaluator_llm.with_structured_output(RetrievalEvaluator)
        
        evaluator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a retrieval quality assessor tasked with determining 
            the relevance of a retrieved document to the user's query.
            
            When the document contains relevant keywords or semantic content that relates 
            to the question, mark it as relevant. Provide a binary score of 'yes' or 'no' 
            to show whether the document is pertinent to the question."""),
            ("human", "Document content: {document}\n\nUser query: {question}")
        ])
        
        self.evaluator = evaluator_prompt | self.structured_evaluator
        
        # Question rewriter
        rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query optimizer that transforms user questions 
            into improved versions suitable for web search. Analyze the input and 
            understand the core semantic intent and meaning behind it."""),
            ("human", "Original question: {question}\n\nCreate an enhanced version of this question.")
        ])
        
        self.question_rewriter = rewriter_prompt | self.llm | StrOutputParser()
        
        # Answer generator
        generator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant specialized in answering questions. 
            Based on the provided retrieved context below, respond to the user's question. 
            If the information is insufficient, simply state that you don't know. 
            Keep your response concise and limit it to three sentences maximum.
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
        self.generator = generator_prompt | self.llm | StrOutputParser()
    
    def _evaluate_documents(self, documents: List[Document], question: str) -> Dict[str, Any]:
        """
        Evaluate document relevance and determine corrective action.
        
        Args:
            documents: Retrieved documents
            question: User question
            
        Returns:
            Dictionary with filtered documents and correction strategy
        """
        filtered_docs = []
        evaluation_details = []
        
        for doc in documents:
            try:
                score = self.evaluator.invoke({
                    "document": doc.page_content,
                    "question": question
                })
                
                is_relevant = score.binary_score.lower() == "yes"
                evaluation_details.append({
                    "document_preview": doc.page_content[:config.Constants.DOC_PREVIEW_LENGTH] + "...",
                    "relevant": is_relevant
                })
                
                if is_relevant:
                    filtered_docs.append(doc)
                    
            except Exception as e:
                logger.error(f"Error evaluating document: {e}")
                evaluation_details.append({
                    "document_preview": doc.page_content[:config.Constants.DOC_PREVIEW_LENGTH] + "...",
                    "relevant": False,
                    "error": str(e)
                })
        
        # Determine correction strategy
        relevance_ratio = len(filtered_docs) / len(documents) if documents else 0
        needs_web_search = relevance_ratio < config.RELEVANCE_THRESHOLD
        
        correction_strategy = "web_search" if needs_web_search else "use_filtered"
        
        return {
            "filtered_documents": filtered_docs,
            "relevance_ratio": relevance_ratio,
            "needs_web_search": needs_web_search,
            "correction_strategy": correction_strategy,
            "evaluation_details": evaluation_details
        }
    
    def _perform_web_search(self, question: str) -> List[Document]:
        """
        Perform web search to get additional documents.
        
        Args:
            question: Search query
            
        Returns:
            List of documents from web search
        """
        try:
            # Rewrite question for better web search
            improved_question = self.question_rewriter.invoke({"question": question})
            
            # Perform web search using Tavily client
            search_response = self.web_search_tool.search(
                query=improved_question,
                max_results=config.WEB_SEARCH_RESULTS
            )
            
            # Convert to documents
            web_docs = []
            if search_response and "results" in search_response:
                for result in search_response["results"]:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("url", "web_search"),
                            "title": result.get("title", "Web Search Result")
                        }
                    )
                    web_docs.append(doc)
            
            return web_docs, improved_question
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [], question
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the CRAG system with a question.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and system information
        """
        try:
            # Validate input
            question = self._validate_question(question)
            
            if not self.retriever:
                return self._create_error_response(
                    config.Constants.ERROR_MESSAGES['NO_DOCS_LOADED'],
                    config.Constants.SYSTEM_TYPES['CRAG']
                )
            
            # Step 1: Retrieve initial documents
            initial_docs = self.retriever.get_relevant_documents(question)
            logger.info(f"CRAG retrieved {len(initial_docs)} initial documents")
            
            # Step 2: Evaluate document relevance
            evaluation_result = self._evaluate_documents(initial_docs, question)
            logger.info(f"CRAG relevance evaluation: {evaluation_result['relevance_ratio']:.2f}")
            
            # Step 3: Apply corrective strategy
            final_docs = evaluation_result["filtered_documents"]
            improved_question = question
            web_search_performed = False
            
            if evaluation_result["needs_web_search"]:
                logger.info("CRAG triggering web search due to low relevance")
                web_docs, improved_question = self._perform_web_search(question)
                final_docs.extend(web_docs)
                web_search_performed = True
            
            # Step 4: Generate answer
            if final_docs:
                context = "\n\n".join([doc.page_content for doc in final_docs])
                answer = self.generator.invoke({
                    "context": context,
                    "question": improved_question
                })
                logger.info(f"CRAG generated answer using {len(final_docs)} documents")
            else:
                answer = config.Constants.ERROR_MESSAGES['INSUFFICIENT_INFO']
                logger.warning("CRAG found no relevant documents for answer generation")
            
            return {
                "answer": answer,
                "retrieved_documents": final_docs,
                "num_documents": len(final_docs),
                "system_type": config.Constants.SYSTEM_TYPES['CRAG'],
                "initial_documents_count": len(initial_docs),
                "relevance_ratio": evaluation_result["relevance_ratio"],
                "correction_strategy": evaluation_result["correction_strategy"],
                "web_search_performed": web_search_performed,
                "improved_question": improved_question if improved_question != question else None,
                "evaluation_details": evaluation_result["evaluation_details"]
            }
            
        except ValueError as e:
            logger.error(f"CRAG query validation error: {e}")
            return self._create_error_response(str(e), config.Constants.SYSTEM_TYPES['CRAG'])
        except Exception as e:
            logger.error(f"CRAG query error: {e}")
            return self._create_error_response(str(e), config.Constants.SYSTEM_TYPES['CRAG'])
