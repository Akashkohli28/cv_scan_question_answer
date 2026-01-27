"""
SQLite repository for storing and retrieving CV data.
Serves as the single source of truth for all candidate information.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SQLiteRepository:
    """Repository for managing CV data in SQLite database."""
    
    def __init__(self, db_path: str = 'data/cv.db'):
        """
        Initialize the SQLite repository.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if not already present."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create candidates table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS candidates (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT,
                        phone TEXT,
                        summary TEXT,
                        skills TEXT,
                        experience TEXT,
                        education TEXT,
                        projects TEXT,
                        certifications TEXT,
                        interests TEXT,
                        file_path TEXT,
                        file_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check if interests column exists, if not add it
                cursor.execute("PRAGMA table_info(candidates)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'interests' not in columns:
                    try:
                        cursor.execute("ALTER TABLE candidates ADD COLUMN interests TEXT")
                        logger.info("Added interests column to candidates table")
                    except sqlite3.OperationalError:
                        logger.warning("Could not add interests column (may already exist)")
                
                # Create embeddings metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id TEXT NOT NULL,
                        chunk_type TEXT,
                        section TEXT,
                        text TEXT,
                        embedding_index INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (candidate_id) REFERENCES candidates(id)
                    )
                """)
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def insert_candidate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert or update a candidate record.
        
        Args:
            data (Dict): Candidate data including name, email, skills, experience, etc.
            
        Returns:
            Dict: The inserted/updated candidate data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                candidate_id = data.get('candidate_id')
                
                # Convert lists/dicts to JSON for storage
                skills_json = json.dumps(data.get('skills', []))
                experience_json = json.dumps(data.get('experience', []))
                education_json = json.dumps(data.get('education', []))
                projects_json = json.dumps(data.get('projects', []))
                certifications_json = json.dumps(data.get('certifications', []))
                interests_json = json.dumps(data.get('interests', []))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO candidates
                    (id, name, email, phone, summary, skills, experience, education, projects, certifications, interests, file_path, file_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate_id,
                    data.get('name'),
                    data.get('email'),
                    data.get('phone'),
                    data.get('summary'),
                    skills_json,
                    experience_json,
                    education_json,
                    projects_json,
                    certifications_json,
                    interests_json,
                    data.get('file_path'),
                    data.get('file_name')
                ))
                
                conn.commit()
                logger.info(f"Candidate {candidate_id} inserted/updated successfully")
                
                return data
        
        except Exception as e:
            logger.error(f"Error inserting candidate: {str(e)}")
            raise
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a candidate by ID.
        
        Args:
            candidate_id (str): The candidate's unique identifier
            
        Returns:
            Optional[Dict]: Candidate data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving candidate: {str(e)}")
            return None
    
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """
        Retrieve all candidates with basic info.
        
        Returns:
            List[Dict]: List of all candidates (basic fields only)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, email, phone, created_at FROM candidates
                    ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                
                candidates = []
                for row in rows:
                    candidates.append({
                        'candidate_id': row[0],
                        'name': row[1],
                        'email': row[2],
                        'phone': row[3],
                        'created_at': row[4]
                    })
                
                return candidates
        
        except Exception as e:
            logger.error(f"Error retrieving candidates: {str(e)}")
            return []
    
    def filter_candidates(self, skills: Optional[List[str]] = None,
                         min_experience_years: Optional[int] = None,
                         company: Optional[str] = None,
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Filter candidates based on criteria.
        
        Args:
            skills (Optional[List]): List of required skills
            min_experience_years (Optional[int]): Minimum years of experience
            company (Optional[str]): Previous employer name
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Filtered list of candidates
        """
        try:
            candidates = self.get_all_candidates()
            
            # Filter by skills
            if skills:
                skills_lower = [s.lower() for s in skills]
                filtered = []
                for cand_id in [c['candidate_id'] for c in candidates]:
                    cand_data = self.get_candidate(cand_id)
                    if cand_data:
                        cand_skills = [s.lower() for s in cand_data.get('skills', [])]
                        if all(skill in cand_skills for skill in skills_lower):
                            filtered.append(cand_id)
                candidates = [c for c in candidates if c['candidate_id'] in filtered]
            
            # Filter by experience
            if min_experience_years:
                filtered = []
                for cand_id in [c['candidate_id'] for c in candidates]:
                    cand_data = self.get_candidate(cand_id)
                    if cand_data and len(cand_data.get('experience', [])) >= min_experience_years:
                        filtered.append(cand_id)
                candidates = [c for c in candidates if c['candidate_id'] in filtered]
            
            # Filter by company
            if company:
                filtered = []
                company_lower = company.lower()
                for cand_id in [c['candidate_id'] for c in candidates]:
                    cand_data = self.get_candidate(cand_id)
                    if cand_data:
                        for exp in cand_data.get('experience', []):
                            if company_lower in exp.get('company', '').lower():
                                filtered.append(cand_id)
                                break
                candidates = [c for c in candidates if c['candidate_id'] in filtered]
            
            return candidates[:limit]
        
        except Exception as e:
            logger.error(f"Error filtering candidates: {str(e)}")
            return []
    
    def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete a candidate and their associated data.
        
        Args:
            candidate_id (str): The candidate's unique identifier
            
        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete candidate
                cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
                
                # Delete associated chunks
                cursor.execute("DELETE FROM chunks WHERE candidate_id = ?", (candidate_id,))
                
                conn.commit()
                logger.info(f"Candidate {candidate_id} deleted successfully")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting candidate: {str(e)}")
            return False
    
    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """
        Convert database row to dictionary with parsed JSON fields.
        
        Args:
            row (tuple): Database row tuple
            
        Returns:
            Dict: Candidate data dictionary
        """
        return {
            'candidate_id': row[0],
            'name': row[1],
            'email': row[2],
            'phone': row[3],
            'summary': row[4],
            'skills': json.loads(row[5]) if row[5] else [],
            'experience': json.loads(row[6]) if row[6] else [],
            'education': json.loads(row[7]) if row[7] else [],
            'projects': json.loads(row[8]) if row[8] else [],
            'certifications': json.loads(row[9]) if row[9] else [],
            'interests': json.loads(row[10]) if row[10] else [],
            'file_path': row[11],
            'file_name': row[12],
            'created_at': row[13]
        }
