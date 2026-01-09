# Resume Builder API

AI-powered Resume Enhancement API that helps you create professional, impactful, and leadership-toned resumes with optional job description alignment.

## Features

- **Professional Summary Generation** - Generate enhanced resume summaries in multiple tones
- **Experience Description Enhancement** - Improve work experience descriptions
- **Job Description Alignment** - Tailor your resume content to match specific job requirements
- **Missing Skills Analysis** - Identify gaps between your skills and job requirements

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume_tool
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
Security_ID=your_security_id_here
```

## Usage

### Start the Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Professional Summary (without JD)
Generate an enhanced professional summary.

**Endpoint**: `POST /agent/professional/summary`

**Request Body**:
```json
{
  "security_id": "your_security_id",
  "tone": "Professional",
  "original_text": "Experienced software developer..."
}
```

**Response**:
```json
{
  "summary": "Enhanced summary text...",
  "timestamp": 1234567890.123
}
```

### 2. Professional Summary (with JD)
Generate a professional summary tailored to a job description.

**Endpoint**: `POST /agent/JD/professional/summary`

**Request Body**:
```json
{
  "security_id": "your_security_id",
  "tone": "Professional",
  "original_text": "Experienced software developer...",
  "JD": "Job description text..."
}
```

### 3. Experience Description (without JD)
Enhance work experience descriptions.

**Endpoint**: `POST /agent/Experience/Description`

**Request Body**:
```json
{
  "security_id": "your_security_id",
  "tone": "Impactful",
  "original_text": "Worked on web development..."
}
```

**Response**:
```json
{
  "summary": "Enhanced experience description...",
  "timestamp": 1234567890.123
}
```

### 4. Experience Description (with JD)
Enhance experience descriptions aligned with job requirements.

**Endpoint**: `POST /agent/JD/Experience/Description`

**Request Body**:
```json
{
  "security_id": "your_security_id",
  "tone": "Leadership",
  "original_text": "Led development team...",
  "JD": "Job description text..."
}
```

### 5. Missing Skills Analysis
Find skills required by the job description that are missing from your resume.

**Endpoint**: `POST /agent/JD/Missing/Skill`

**Request Body**:
```json
{
  "security_id": "your_security_id",
  "Skill": "Python, JavaScript, React, Node.js...",
  "JD": "Required skills: Python, Java, AWS, Docker..."
}
```

**Response**:
```json
{
  "missing_skills": [
    "Java",
    "AWS",
    "Docker"
  ],
  "timestamp": 1234567890.123
}
```

### Health Check
Check API health status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

## Tone Options

- `Professional` - Professional & Clean tone
- `Impactful` - Impactful & Strong tone
- `Leadership` - Leadership Tone

## Error Responses

All endpoints return appropriate HTTP status codes:

- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid security ID)
- `500` - Internal Server Error

**Error Response Format**:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Project Structure

```
resume_tool/
├── app.py                          # Main FastAPI application
├── Professional_s_agent.py         # Professional summary agent
├── Experience_d_agent.py           # Experience description agent
├── JD_Professional_s_agent.py      # JD-aligned professional summary
├── JD_Experience_d_agent.py        # JD-aligned experience description
├── JD_skill_agent.py               # Missing skills analysis
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dependencies

- FastAPI
- Pydantic
- LangChain
- OpenAI
- python-dotenv
- uvicorn

## Security

All endpoints require a valid `security_id` for authentication. Make sure to set your `Security_ID` in the `.env` file.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
