"""
Query routes for RAG-based question answering on CV data.
"""

from flask import Blueprint, request, jsonify, current_app
from services.sqlite_repo import SQLiteRepository
from services.faiss_index import FAISSIndex
from services.embedding import EmbeddingService
from services.query_agent import QueryAgent

query_bp = Blueprint('query', __name__)


@query_bp.route('/query', methods=['POST'])
def query_candidates():
    """
    Answer natural language questions about candidates using RAG.
    
    Expected request JSON:
    {
        "question": "Who has experience with Python?",
        "candidate_id": "optional-uuid-to-scope-search",
        "top_k": 5
    }
    
    Returns:
        JSON with answer and retrieved context
    """
    try:
        # Validate request
        data = request.get_json()
        if not data or 'question' not in data:
            return {'error': 'No question provided'}, 400
        
        question = data['question']
        candidate_id = data.get('candidate_id')
        top_k = data.get('top_k', 10)  # Increased default from 5 to 10 for more comprehensive results
        
        # Initialize services
        db_path = current_app.config['DATABASE']
        faiss_index_path = current_app.config['FAISS_INDEX_PATH']
        
        db_repo = SQLiteRepository(db_path)
        faiss_index = FAISSIndex(faiss_index_path)
        embedding_service = EmbeddingService()
        query_agent = QueryAgent(db_repo, faiss_index, embedding_service)
        # Get answer from RAG pipeline
        answer_data = query_agent.answer_question(
            question=question,
            candidate_id=candidate_id,
            top_k=top_k
        )
        
        return {
            'question': question,
            'answer': answer_data['answer'],
            'context': answer_data['context'],
            'sources': answer_data['sources'],
            'confidence': answer_data.get('confidence', 'unknown')
        }, 200
        
    except Exception as e:
        return {
            'error': 'Query failed',
            'message': str(e)
        }, 500


@query_bp.route('/search', methods=['POST'])
def semantic_search():
    """
    Perform semantic search on CV embeddings.
    
    Expected request JSON:
    {
        "query": "machine learning experience",
        "candidate_id": "optional-uuid",
        "top_k": 5
    }
    
    Returns:
        JSON with search results and scores
    """
    try:
        # Validate request
        data = request.get_json()
        if not data or 'query' not in data:
            return {'error': 'No query provided'}, 400
        
        search_query = data['query']
        candidate_id = data.get('candidate_id')
        top_k = data.get('top_k', 10)  # Increased default from 5 to 10 for more comprehensive results
        
        # Initialize services
        faiss_index_path = current_app.config['FAISS_INDEX_PATH']
        db_path = current_app.config['DATABASE']
        
        faiss_index = FAISSIndex(faiss_index_path)
        embedding_service = EmbeddingService()
        db_repo = SQLiteRepository(db_path)
        
        # Generate query embedding
        query_embedding = embedding_service.embed(search_query)
        
        # Search FAISS index
        results = faiss_index.search(query_embedding, top_k=top_k, candidate_id=candidate_id)
        
        # Enrich results with candidate data
        enriched_results = []
        for result in results:
            candidate = db_repo.get_candidate(result['metadata']['candidate_id'])
            enriched_results.append({
                'candidate_name': candidate.get('name') if candidate else 'Unknown',
                'candidate_id': result['metadata']['candidate_id'],
                'chunk_type': result['metadata']['chunk_type'],
                'section': result['metadata'].get('section'),
                'distance': result['distance']
            })
        
        return {
            'query': search_query,
            'results': enriched_results,
            'total_results': len(enriched_results)
        }, 200
        
    except Exception as e:
        return {
            'error': 'Search failed',
            'message': str(e)
        }, 500


@query_bp.route('/filter-candidates', methods=['POST'])
def filter_candidates():
    """
    Filter candidates based on structured criteria.
    
    Expected request JSON:
    {
        "skills": ["Python", "Machine Learning"],
        "min_experience_years": 3,
        "company": "Google",
        "limit": 10
    }
    
    Returns:
        JSON with matching candidates
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return {'error': 'No filter criteria provided'}, 400
        
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        
        # Filter candidates using various criteria
        candidates = db_repo.filter_candidates(
            skills=data.get('skills'),
            min_experience_years=data.get('min_experience_years'),
            company=data.get('company'),
            limit=data.get('limit', 50)
        )
        
        return {
            'filters': data,
            'results': candidates,
            'total_results': len(candidates)
        }, 200
        
    except Exception as e:
        return {
            'error': 'Filter failed',
            'message': str(e)
        }, 500


@query_bp.route('/candidate/<candidate_id>/context', methods=['GET'])
def get_candidate_context(candidate_id):
    """
    Get full context and metadata for a candidate for analysis.
    
    Args:
        candidate_id (str): The candidate's unique identifier
        
    Returns:
        JSON with candidate's full profile and indexed sections
    """
    try:
        db_path = current_app.config['DATABASE']
        faiss_index_path = current_app.config['FAISS_INDEX_PATH']
        
        db_repo = SQLiteRepository(db_path)
        faiss_index = FAISSIndex(faiss_index_path)
        
        # Get candidate data
        candidate = db_repo.get_candidate(candidate_id)
        if not candidate:
            return {'error': 'Candidate not found'}, 404
        
        # Get indexed chunks for this candidate
        indexed_sections = faiss_index.get_candidate_chunks(candidate_id)
        
        return {
            'candidate': candidate,
            'indexed_sections': indexed_sections,
            'total_sections': len(indexed_sections)
        }, 200
        
    except Exception as e:
        return {
            'error': 'Context retrieval failed',
            'message': str(e)
        }, 500

@query_bp.route('/candidate/<candidate_id>/full-summary', methods=['GET'])
def get_candidate_full_summary(candidate_id):
    """
    Get a comprehensive summary of all information about a candidate.
    
    This endpoint retrieves ALL sections of a candidate's CV including:
    - Personal info (name, email, phone)
    - Professional summary
    - All work experience entries
    - All projects
    - All skills
    - All education
    - All certifications
    
    Args:
        candidate_id (str): The candidate's unique identifier
    
    Returns:
        JSON with complete candidate information
    """
    try:
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        
        # Get complete candidate data
        candidate = db_repo.get_candidate(candidate_id)
        if not candidate:
            return {'error': 'Candidate not found'}, 404
        
        # Format comprehensive summary
        summary = {
            'candidate_id': candidate_id,
            'name': candidate.get('name'),
            'email': candidate.get('email'),
            'phone': candidate.get('phone'),
            'summary': candidate.get('summary'),
            'skills': candidate.get('skills', []),
            'experience': candidate.get('experience', []),
            'projects': candidate.get('projects', []),
            'education': candidate.get('education', []),
            'certifications': candidate.get('certifications', []),
            'interests': candidate.get('interests', []),
            'totals': {
                'skills_count': len(candidate.get('skills', [])),
                'experience_count': len(candidate.get('experience', [])),
                'project_count': len(candidate.get('projects', [])),
                'education_count': len(candidate.get('education', [])),
                'certification_count': len(candidate.get('certifications', [])),
                'interests_count': len(candidate.get('interests', []))
            }
        }
        
        return summary, 200
        
    except Exception as e:
        return {
            'error': 'Failed to retrieve candidate summary',
            'message': str(e)
        }, 500