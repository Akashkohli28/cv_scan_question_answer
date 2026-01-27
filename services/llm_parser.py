"""
LLM-based CV parser for extracting structured information.
Uses OpenAI function calling to parse CV text into structured JSON.
"""

import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI


class LLMParser:
    """Parse CV text into structured data using LLM function calling."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the LLM parser.
        
        Args:
            api_key (Optional[str]): OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model (str): OpenAI model to use (default: gpt-4-turbo)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def parse(self, cv_text: str) -> Dict[str, Any]:
        """
        Parse CV text into structured format using LLM function calling.
        
        Args:
            cv_text (str): Raw extracted CV text
            
        Returns:
            Dict[str, Any]: Structured CV data with fixed JSON schema
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a CV parsing expert. Extract ALL information from CVs and return structured JSON.
                        Be thorough and extract everything mentioned in the CV:
                        - Extract ALL work experience entries
                        - Extract ALL projects mentioned
                        - Extract ALL education entries
                        - Extract ALL certifications and credentials
                        - Extract ALL skills listed
                        - Do not omit or summarize any information
                        - If information is not present, use empty arrays or null values appropriately"""
                    },
                    {
                        "role": "user",
                        "content": f"Parse this CV completely and extract ALL information:\n\n{cv_text}"
                    }
                ],
                tools=[{"type": "function", "function": self._get_cv_schema()}],
                tool_choice={"type": "function", "function": {"name": "parse_cv"}}
            )
            
            # Extract tool call result
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                parsed_data = json.loads(tool_call.function.arguments)
                return parsed_data
            else:
                # Fallback if no tool call
                return self._default_parse(cv_text)
        
        except Exception as e:
            raise ValueError(f"Error parsing CV with LLM: {str(e)}")
    
    def _get_cv_schema(self) -> Dict[str, Any]:
        """
        Define the CV schema for function calling.
        
        Returns:
            Dict: OpenAI function schema for CV parsing
        """
        return {
            "name": "parse_cv",
            "description": "Parse CV and extract structured information",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name of the candidate"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Phone number"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Professional summary or objective"
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of technical and soft skills"
                    },
                    "experience": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "company": {"type": "string"},
                                "duration": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        },
                        "description": "Professional experience entries"
                    },
                    "education": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "degree": {"type": "string"},
                                "institution": {"type": "string"},
                                "year": {"type": "string"},
                                "details": {"type": "string"}
                            }
                        },
                        "description": "Educational background"
                    },
                    "projects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "technologies": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "url": {"type": "string"}
                            }
                        },
                        "description": "Notable projects"
                    },
                    "certifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "issuer": {"type": "string"},
                                "year": {"type": "string"}
                            }
                        },
                        "description": "Professional certifications"
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Personal interests and hobbies listed in the CV"
                    }
                },
                "required": ["name", "email", "skills"]
            }
        }
    
    def _default_parse(self, cv_text: str) -> Dict[str, Any]:
        """
        Fallback parsing logic if LLM function calling fails.
        
        Args:
            cv_text (str): Raw CV text
            
        Returns:
            Dict[str, Any]: Partially parsed CV data
        """
        # Simple fallback parsing
        lines = cv_text.split("\n")
        
        return {
            "name": self._extract_name(lines),
            "email": self._extract_email(cv_text),
            "phone": self._extract_phone(cv_text),
            "summary": self._extract_summary(lines),
            "skills": self._extract_skills(lines),
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "interests": self._extract_interests(lines)
        }
    
    def _extract_name(self, lines: list) -> str:
        """Extract candidate name from first non-empty line."""
        for line in lines:
            if line.strip() and len(line.strip()) < 100:
                return line.strip()
        return "Unknown"
    
    def _extract_email(self, text: str) -> str:
        """Extract email address from text."""
        import re
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        return match.group(0) if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text."""
        import re
        match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        return match.group(0) if match else ""
    
    def _extract_summary(self, lines: list) -> str:
        """Extract professional summary."""
        for i, line in enumerate(lines):
            if 'summary' in line.lower() or 'objective' in line.lower():
                # Return next few non-empty lines
                summary_lines = []
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip():
                        summary_lines.append(lines[j].strip())
                return " ".join(summary_lines)
        return ""
    
    def _extract_skills(self, lines: list) -> list:
        """Extract skills section."""
        skills = []
        in_skills_section = False
        
        for line in lines:
            if 'skills' in line.lower():
                in_skills_section = True
                continue
            
            if in_skills_section:
                if any(keyword in line.lower() for keyword in ['experience', 'education', 'projects']):
                    break
                
                if line.strip():
                    # Split by common delimiters
                    skill_items = [s.strip() for s in line.replace(',', '|').replace(';', '|').split('|')]
                    skills.extend([s for s in skill_items if s])
        
        return skills    
    def _extract_interests(self, lines: list) -> list:
        """Extract interests and hobbies section."""
        interests = []
        in_interests_section = False
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['interests', 'hobbies', 'personal interests', 'activities']):
                in_interests_section = True
                continue
            
            if in_interests_section:
                if any(keyword in line_lower for keyword in ['experience', 'education', 'projects', 'skills', 'certifications']):
                    break
                
                if line.strip():
                    # Split by common delimiters
                    interest_items = [s.strip() for s in line.replace(',', '|').replace(';', '|').split('|')]
                    interests.extend([s for s in interest_items if s and len(s) > 2])
        
        return interests