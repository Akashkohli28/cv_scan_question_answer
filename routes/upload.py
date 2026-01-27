"""
Upload routes for handling CV file uploads, parsing, and storage.
"""

import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.text_extraction import TextExtractor
from services.llm_parser import LLMParser
from services.sqlite_repo import SQLiteRepository
from services.embedding import EmbeddingService
from services.faiss_index import FAISSIndex

upload_bp = Blueprint('upload', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The filename to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route('/upload', methods=['POST'])
def upload_cv():
    """
    Handle CV file upload, parsing, and storage.
    
    Expected request:
    - file: PDF or DOCX file
    - candidate_name (optional): Name of the candidate
    
    Returns:
        JSON response with candidate_id and parsing results
    """
    try:
        # Validate file presence
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if not allowed_file(file.filename):
            return {'error': 'File type not allowed. Use PDF or DOCX'}, 400
        
        # Generate candidate ID
        candidate_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, f"{candidate_id}_{filename}")
        file.save(file_path)
        
        # Extract text from file
        text_extractor = TextExtractor()
        raw_text = text_extractor.extract(file_path)
        
        if not raw_text:
            return {'error': 'Could not extract text from file'}, 400
        
        # Parse structured data using LLM
        llm_parser = LLMParser()
        parsed_data = llm_parser.parse(raw_text)
        
        # Ensure all expected fields exist with defaults
        if not isinstance(parsed_data.get('interests'), list):
            parsed_data['interests'] = []
        
        # Add candidate_id and file metadata
        parsed_data['candidate_id'] = candidate_id
        parsed_data['file_path'] = file_path
        parsed_data['file_name'] = filename
        
        # Use candidate name from request or parsed data
        candidate_name = request.form.get('candidate_name') or parsed_data.get('name')
        if candidate_name:
            parsed_data['name'] = candidate_name
        
        # Store in SQLite
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        candidate_data = db_repo.insert_candidate(parsed_data)
        
        # Generate embeddings for experience and projects
        embedding_service = EmbeddingService()
        faiss_index = FAISSIndex(current_app.config['FAISS_INDEX_PATH'])
        
        chunks = _create_searchable_chunks(parsed_data, candidate_id)
        
        for chunk in chunks:
            # Generate embedding
            embedding = embedding_service.embed(chunk['text'])
            
            # Add to FAISS index with text content included for retrieval
            faiss_index.add_vector(
                embedding,
                metadata={
                    'candidate_id': candidate_id,
                    'chunk_type': chunk['type'],
                    'section': chunk.get('section', 'unknown'),
                    'text': chunk['text']  # Store actual chunk text for retrieval
                }
            )
        
        # Save FAISS index
        faiss_index.save()
        
        return {
            'message': 'CV uploaded and processed successfully',
            'candidate_id': candidate_id,
            'candidate_name': parsed_data.get('name', 'Unknown'),
            'parsed_data': {
                'name': parsed_data.get('name'),
                'email': parsed_data.get('email'),
                'phone': parsed_data.get('phone'),
                'skills': parsed_data.get('skills', []),
                'experience_count': len(parsed_data.get('experience', [])),
                'project_count': len(parsed_data.get('projects', [])),
                'interests': parsed_data.get('interests', [])
            }
        }, 201
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR in upload_cv(): {str(e)}")
        print(f"{'='*60}")
        print(error_trace)
        print(f"{'='*60}\n")
        return {
            'error': 'Processing failed',
            'message': str(e),
            'trace': error_trace
        }, 500


def _create_searchable_chunks(parsed_data, candidate_id):
    """
    Create searchable chunks from parsed CV data.
    
    Args:
        parsed_data (dict): Parsed CV data
        candidate_id (str): Unique candidate identifier
        
    Returns:
        list: List of chunks with text and metadata
    """
    chunks = []
    
    # Add summary/overview
    if parsed_data.get('summary'):
        chunks.append({
            'type': 'summary',
            'text': parsed_data['summary'],
            'section': 'professional_summary'
        })
    
    # Add each experience entry
    for idx, exp in enumerate(parsed_data.get('experience', [])):
        exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}. {exp.get('description', '')}"
        chunks.append({
            'type': 'experience',
            'text': exp_text,
            'section': f"experience_{idx}"
        })
    
    # Add each project entry
    for idx, proj in enumerate(parsed_data.get('projects', [])):
        proj_text = f"{proj.get('name', '')}. {proj.get('description', '')}. Technologies: {', '.join(proj.get('technologies', []))}"
        chunks.append({
            'type': 'project',
            'text': proj_text,
            'section': f"project_{idx}"
        })
    
    # Add skills as a single chunk
    if parsed_data.get('skills'):
        skills_text = f"Skills: {', '.join(parsed_data['skills'])}"
        chunks.append({
            'type': 'skills',
            'text': skills_text,
            'section': 'skills'
        })
    
    # Add each education entry
    for idx, edu in enumerate(parsed_data.get('education', [])):
        edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('year', '')}). {edu.get('details', '')}"
        chunks.append({
            'type': 'education',
            'text': edu_text,
            'section': f"education_{idx}"
        })
    
    # Add each certification entry
    for idx, cert in enumerate(parsed_data.get('certifications', [])):
        cert_text = f"{cert.get('name', '')} from {cert.get('issuer', '')} ({cert.get('year', '')})"
        chunks.append({
            'type': 'certification',
            'text': cert_text,
            'section': f"certification_{idx}"
        })
    
    # Add interests/hobbies as a single chunk
    if parsed_data.get('interests'):
        interests_text = f"Interests and Hobbies: {', '.join(parsed_data['interests'])}"
        chunks.append({
            'type': 'interests',
            'text': interests_text,
            'section': 'interests_hobbies'
        })
    
    return chunks


@upload_bp.route('/candidates', methods=['GET'])
def list_candidates():
    """
    List all uploaded candidates.
    
    Returns:
        JSON list of candidates with basic info
    """
    try:
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        candidates = db_repo.get_all_candidates()
        
        return {
            'candidates': candidates,
            'total': len(candidates)
        }, 200
        
    except Exception as e:
        return {
            'error': 'Failed to retrieve candidates',
            'message': str(e)
        }, 500


@upload_bp.route('/candidate/<candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    """
    Get detailed information about a specific candidate.
    
    Args:
        candidate_id (str): The candidate's unique identifier
        
    Returns:
        JSON with candidate's full parsed data
    """
    try:
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        candidate = db_repo.get_candidate(candidate_id)
        
        if not candidate:
            return {'error': 'Candidate not found'}, 404
        
        return candidate, 200
        
    except Exception as e:
        return {
            'error': 'Failed to retrieve candidate',
            'message': str(e)
        }, 500


@upload_bp.route('/candidate/<candidate_id>', methods=['DELETE'])
def delete_candidate(candidate_id):
    """
    Delete a candidate and their CV data.
    
    Args:
        candidate_id (str): The candidate's unique identifier
        
    Returns:
        JSON confirmation of deletion
    """
    try:
        db_path = current_app.config['DATABASE']
        db_repo = SQLiteRepository(db_path)
        
        # Get candidate before deletion to get file path
        candidate = db_repo.get_candidate(candidate_id)
        if not candidate:
            return {'error': 'Candidate not found'}, 404
        
        # Delete from database
        success = db_repo.delete_candidate(candidate_id)
        
        if success:
            # Try to delete uploaded file
            file_path = candidate.get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete file {file_path}: {str(e)}")
            
            return {
                'message': 'Candidate deleted successfully',
                'candidate_id': candidate_id,
                'candidate_name': candidate.get('name')
            }, 200
        else:
            return {'error': 'Failed to delete candidate'}, 500
        
    except Exception as e:
        return {
            'error': 'Delete failed',
            'message': str(e)
        }, 500
