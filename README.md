# CV Scan - RAG-Based CV Parser & Query System

A Flask-based web application for parsing CVs, extracting structured information using LLM function calling, and answering natural language questions through a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **CV Upload & Parsing**: Upload PDF/DOCX files, extract text, and parse structured data (name, email, skills, experience, projects, education, certifications)
- **LLM-Powered Extraction**: Uses OpenAI function calling for accurate structured data extraction
- **Semantic Search**: Generate embeddings for CV chunks and store in FAISS for efficient semantic search
- **RAG Pipeline**: Answer natural language questions by combining database queries with semantic search and LLM generation
- **Candidate Management**: List, view, and filter candidates by skills, experience, and company
- **Web Interface**: Simple, responsive HTML/JavaScript frontend for uploading CVs and querying

## Architecture

### Backend Stack
- **Framework**: Flask with CORS support
- **Database**: SQLite (structured data storage)
- **Vector Index**: FAISS (semantic search)
- **LLM**: OpenAI API (GPT-4 for parsing and QA)
- **Embeddings**: OpenAI text-embedding-3-small

### Frontend Stack
- **HTML5/CSS3**: Responsive modern UI
- **JavaScript**: Fetch API for backend communication
- **Real-time Status**: Upload progress and query processing feedback

## Project Structure

```
project/
├── app.py                      # Flask app factory and configuration
├── routes/
│   ├── upload.py              # CV upload and parsing endpoints
│   └── query.py               # RAG query and search endpoints
├── services/
│   ├── text_extraction.py      # PDF/DOCX text extraction
│   ├── llm_parser.py           # LLM function calling for CV parsing
│   ├── sqlite_repo.py          # Database operations
│   ├── embedding.py            # OpenAI embedding service
│   ├── faiss_index.py          # FAISS vector index management
│   └── query_agent.py          # RAG pipeline orchestration
├── frontend/
│   ├── index.html              # Web UI
│   └── app.js                  # Frontend JavaScript logic
├── data/
│   ├── cv.db                   # SQLite database (auto-created)
│   └── faiss.index             # FAISS index (auto-created)
├── uploads/                     # Temporary CV file storage
├── swagger/
│   └── swagger.yaml            # OpenAPI documentation
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- pip package manager

### 1. Clone & Setup Environment

```bash
cd cv_scan
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create or update `.env` file with your OpenAI API key:

```env
FLASK_ENV=development
FLASK_PORT=5000
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_PATH=data/cv.db
FAISS_INDEX_PATH=data/faiss.index
UPLOAD_FOLDER=uploads
```

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 5. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000/frontend/index.html
```

## API Endpoints

### Upload & Management
- `POST /api/upload` - Upload and parse a CV
- `GET /api/candidates` - List all candidates
- `GET /api/candidate/<candidate_id>` - Get candidate details
- `GET /api/candidate/<candidate_id>/context` - Get candidate's indexed context

### Query & Search
- `POST /api/query` - Answer questions using RAG pipeline
- `POST /api/search` - Perform semantic search on CVs
- `POST /api/filter-candidates` - Filter candidates by criteria

### System
- `GET /health` - Health check

## Usage Examples

### Upload a CV

```bash
curl -X POST \
  -F "file=@resume.pdf" \
  -F "candidate_name=John Doe" \
  http://localhost:5000/api/upload
```

### Query a CV

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What programming languages does the candidate know?",
    "candidate_id": "uuid-here",
    "top_k": 5
  }' \
  http://localhost:5000/api/query
```

### Semantic Search

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning experience",
    "top_k": 10
  }' \
  http://localhost:5000/api/search
```

### Filter Candidates

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["Python", "Machine Learning"],
    "min_experience_years": 3,
    "company": "Google"
  }' \
  http://localhost:5000/api/filter-candidates
```

## How It Works

### 1. CV Upload Pipeline
1. User uploads PDF/DOCX file
2. Text extraction using PyPDF2/python-docx
3. LLM function calling extracts structured fields
4. Data stored in SQLite (source of truth)
5. Text chunks generated and embedded using OpenAI
6. Embeddings stored in FAISS with candidate_id metadata

### 2. RAG Query Pipeline
1. User asks natural language question
2. Question embedding generated
3. FAISS semantic search retrieves relevant CV chunks
4. Retrieved context formatted with candidate metadata
5. LLM generates answer based on context + question
6. Confidence score determined by embedding distances
7. Sources and metadata returned to user

## Configuration

### Model Selection

Edit `services/llm_parser.py` and `services/query_agent.py` to change models:

```python
# Available models
model = "gpt-4"              # More accurate, slower
model = "gpt-3.5-turbo"      # Faster, good quality

# For embeddings (in embedding.py)
model = "text-embedding-3-small"   # 1536 dimensions, cost-effective
model = "text-embedding-3-large"   # 3072 dimensions, more powerful
```

### FAISS Index

For production deployments, consider:

```python
# In faiss_index.py - Change index type
self.index = faiss.IndexIVFFlat(...)  # For large datasets
self.index = faiss.IndexHNSW(...)     # For fast approximate search
```

## Limitations & Improvements

### Current Limitations
- FAISS deletion is simulated (marks as removed, doesn't reclaim space)
- SQLite suitable for <10k candidates; use PostgreSQL for larger scale
- PDF text extraction quality depends on PDF structure
- OpenAI API rate limits apply

### Future Improvements
- Add authentication and multi-user support
- Implement incremental FAISS index management
- Support for video/image CVs
- Real-time streaming answers
- Candidate comparison features
- Advanced filtering with date ranges
- Export candidate reports

## Performance Considerations

### Optimization Tips
1. **Batch Embeddings**: Use `embed_batch()` for multiple texts
2. **Database Indexing**: Add indexes on `candidate_id` in chunks table
3. **FAISS Tuning**: Use IndexIVF for >100k vectors
4. **Caching**: Cache embeddings for frequently queried sections
5. **Vector Quantization**: Use PQ or OPQ for compression

### Scaling
- Split FAISS index by date/skill for faster searches
- Use Redis for caching LLM responses
- Implement async task queue for background parsing
- Use connection pooling for SQLite

## Troubleshooting

### API Key Error
```
ValueError: OPENAI_API_KEY environment variable not set
```
Solution: Set OPENAI_API_KEY in `.env` file

### PDF Extraction Fails
```
Error extracting PDF: ...
```
Solution: Some PDFs are image-based. Use OCR preprocessing

### FAISS Index Error
```
File not found: data/faiss.index
```
Solution: Index is auto-created on first upload. Upload a CV first.

### Slow Queries
Solution: Reduce `top_k` parameter, or filter by `candidate_id` first

## API Documentation

Full OpenAPI/Swagger documentation available at:
- Swagger YAML: `swagger/swagger.yaml`
- Import into Postman, Insomnia, or Swagger UI for interactive testing

## Contributing

Contributions welcome! Areas for improvement:
- Add more LLM models (Claude, Cohere, etc.)
- Implement ReRanking for better relevance
- Add voice input for queries
- Mobile app frontend

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation in `swagger/swagger.yaml`
3. Check Flask logs for detailed error messages
4. Verify OpenAI API quota and rate limits

## Version History

- **v1.0.0** (Jan 2026) - Initial release
  - CV upload and parsing
  - RAG query pipeline
  - Web interface
  - Full API documentation

---

**Built with Flask, OpenAI, FAISS, and SQLite**
