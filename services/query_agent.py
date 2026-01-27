"""
Query agent for orchestrating RAG pipeline.
Combines LLM, semantic search, and database operations to answer questions.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class QueryAgent:
    """Orchestrate RAG pipeline for answering questions about CVs."""
    
    def __init__(self, db_repo, faiss_index, embedding_service, 
                 api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the query agent.
        
        Args:
            db_repo: SQLiteRepository instance
            faiss_index: FAISSIndex instance
            embedding_service: EmbeddingService instance
            api_key (Optional[str]): OpenAI API key
            model (str): OpenAI model to use
        """
        self.db_repo = db_repo
        self.faiss_index = faiss_index
        self.embedding_service = embedding_service
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def answer_question(self, question: str, candidate_id: Optional[str] = None,
                       top_k: int = 10) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question (str): User's question
            candidate_id (Optional[str]): Scope search to specific candidate
            top_k (int): Number of context chunks to retrieve (default: 10)
            
        Returns:
            Dict: Answer, context, sources, and confidence
        """
        try:
            # Step 1: Generate embedding for question
            logger.info(f"Processing question: {question[:50]}...")
            question_embedding = self.embedding_service.embed(question)
            
            # Step 2: Retrieve relevant context from FAISS (fetch more than needed)
            # Request extra results in case filtering reduces the count
            search_k = max(top_k, 15)
            search_results = self.faiss_index.search(
                question_embedding,
                k=search_k,
                candidate_id=candidate_id
            )
            
            # If no results and we have a candidate_id, try again without filtering
            if not search_results and candidate_id:
                logger.info(f"No results for candidate {candidate_id}, trying across all candidates")
                search_results = self.faiss_index.search(
                    question_embedding,
                    k=search_k
                )
            
            if not search_results:
                logger.warning("No relevant context found in FAISS index")
                return {
                    'answer': "No relevant information found in the CV database.",
                    'context': [],
                    'sources': [],
                    'confidence': 'low'
                }
            
            # Step 3: Format context for LLM
            context_text = self._format_context(search_results[:top_k])
            sources = self._extract_sources(search_results[:top_k])
            
            # Step 4: Generate answer using LLM
            answer = self._generate_answer(question, context_text)
            
            # Step 5: Determine confidence
            confidence = self._evaluate_confidence(search_results[:top_k])
            
            logger.info(f"Generated answer with confidence: {confidence}")
            
            return {
                'answer': answer,
                'context': context_text,
                'sources': sources,
                'confidence': confidence
            }
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'context': [],
                'sources': [],
                'confidence': 'error'
            }
    
    def extract_filters(self, question: str) -> Dict[str, Any]:
        """
        Extract query filters from natural language question.
        
        Args:
            question (str): User's question
            
        Returns:
            Dict: Extracted filters (candidate_name, skills, experience, etc.)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract structured filters from user questions about CVs.
                        Return JSON with: candidate_name, required_skills, min_experience_years, company.
                        Only include fields that are explicitly mentioned. Use null for missing fields."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract filters from this question: {question}"
                    }
                ],
                temperature=0.3
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                filters = json.loads(json_match.group())
                # Remove null values
                return {k: v for k, v in filters.items() if v is not None}
            
            return {}
        
        except Exception as e:
            logger.warning(f"Error extracting filters: {str(e)}")
            return {}
    
    def _format_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context for LLM.
        
        Args:
            search_results (List[Dict]): Search results from FAISS
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            candidate_id = metadata.get('candidate_id')
            
            # Get candidate info
            candidate = self.db_repo.get_candidate(candidate_id)
            if not candidate:
                continue
            
            chunk_type = metadata.get('chunk_type', 'unknown')
            section = metadata.get('section', 'unknown')
            distance = result.get('distance', 0)
            
            # Use stored text from metadata, or reconstruct if not available
            text = metadata.get('text')
            if not text:
                text = self._reconstruct_chunk_text(candidate, chunk_type, section)
            
            if text:
                # Calculate relevance score (inverse of distance)
                relevance_score = 1 / (1 + distance)
                context_parts.append(
                    f"[{candidate.get('name')} - {chunk_type.upper()} ({section})]\n{text}\n"
                    f"(Relevance Score: {relevance_score:.2f})"
                )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _reconstruct_chunk_text(self, candidate: Dict, chunk_type: str, section: str) -> str:
        """
        Reconstruct text chunk from candidate data and metadata.
        
        Args:
            candidate (Dict): Candidate data
            chunk_type (str): Type of chunk (experience, project, skills, etc.)
            section (str): Section identifier
            
        Returns:
            str: Reconstructed text
        """
        if chunk_type == 'summary':
            return candidate.get('summary', '')
        
        elif chunk_type == 'skills':
            skills = candidate.get('skills', [])
            return f"Skills: {', '.join(skills)}"
        
        elif chunk_type == 'experience':
            section_num = int(section.split('_')[-1]) if '_' in section else 0
            experiences = candidate.get('experience', [])
            if section_num < len(experiences):
                exp = experiences[section_num]
                return f"{exp.get('title')} at {exp.get('company')} ({exp.get('duration', '')}). {exp.get('description', '')}"
        
        elif chunk_type == 'project':
            section_num = int(section.split('_')[-1]) if '_' in section else 0
            projects = candidate.get('projects', [])
            if section_num < len(projects):
                proj = projects[section_num]
                techs = ', '.join(proj.get('technologies', []))
                return f"{proj.get('name')}. {proj.get('description', '')}. Technologies: {techs}"
        
        elif chunk_type == 'education':
            section_num = int(section.split('_')[-1]) if '_' in section else 0
            education = candidate.get('education', [])
            if section_num < len(education):
                edu = education[section_num]
                return f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('year', '')}). {edu.get('details', '')}"
        
        elif chunk_type == 'certification':
            section_num = int(section.split('_')[-1]) if '_' in section else 0
            certifications = candidate.get('certifications', [])
            if section_num < len(certifications):
                cert = certifications[section_num]
                return f"{cert.get('name', '')} from {cert.get('issuer', '')} ({cert.get('year', '')})"
        
        elif chunk_type == 'interests':
            interests = candidate.get('interests', [])
            return f"Interests and Hobbies: {', '.join(interests)}"
        
        return ""
    
    def _extract_sources(self, search_results: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract source information from search results.
        
        Args:
            search_results (List[Dict]): Search results
            
        Returns:
            List[Dict]: Source information
        """
        sources = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            candidate_id = metadata.get('candidate_id')
            
            candidate = self.db_repo.get_candidate(candidate_id)
            if candidate:
                sources.append({
                    'candidate_name': candidate.get('name'),
                    'candidate_id': candidate_id,
                    'section': metadata.get('section'),
                    'chunk_type': metadata.get('chunk_type'),
                    'relevance': 1 / (1 + result.get('distance', 0))
                })
        
        return sources
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with provided context.
        
        Args:
            question (str): User's question
            context (str): Retrieved context from FAISS
            
        Returns:
            str: Generated answer
        """
        try:
            prompt = f"""Based on the following CV context, answer the user's question comprehensively.
You have been provided with the most relevant sections from the CV.
- Answer as completely as possible using all the provided context
- If the question asks for a list, provide the complete list from the context
- Reference specific details like dates, companies, technologies mentioned
- If certain information isn't in the context, clearly state that
- Be thorough and don't omit relevant details

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a thorough CV analysis assistant. Answer questions based on the provided CV data. Be comprehensive and include all relevant details from the context provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Unable to generate answer due to an error."
    
    def _evaluate_confidence(self, search_results: List[Dict]) -> str:
        """
        Evaluate confidence based on search result relevance.
        
        Args:
            search_results (List[Dict]): Search results with distances
            
        Returns:
            str: Confidence level (high, medium, low)
        """
        if not search_results:
            return 'low'
        
        # Calculate average distance (lower is better)
        avg_distance = np.mean([r.get('distance', float('inf')) for r in search_results])
        
        if avg_distance < 1.0:
            return 'high'
        elif avg_distance < 2.0:
            return 'medium'
        else:
            return 'low'
