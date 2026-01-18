import os
import asyncio
from openai import OpenAI
import json
import fitz
import re

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text



SYSTEM_PROMPT = """You are an expert resume data extraction specialist. Your task is to extract ALL information from resumes with 100% accuracy and completeness.

CRITICAL RULES - ABSOLUTE REQUIREMENTS:
1. Extract EVERY piece of information present in the resume - miss NOTHING
2. Do NOT add, infer, derive, calculate, or hallucinate ANY information not explicitly stated
3. Do NOT extract information from one field and put it in another field
4. If a field is present but empty/unclear, include it as null or empty string
5. Preserve exact formatting of dates, numbers, percentages, and metrics
6. Maintain original spelling, capitalization, and punctuation
7. Extract ALL bullet points, achievements, and descriptions completely
8. Capture technical skills exactly as written (including version numbers, frameworks)
9. Include ALL contact information and links exactly as provided
10. ONLY populate a field if that EXACT information appears in that EXACT context in the resume

EXAMPLES OF WHAT NOT TO DO:
❌ DON'T extract "9+ yrs of experience" from summary and put it in "years_of_experience" field
❌ DON'T calculate years from dates
❌ DON'T infer employment_type if not stated
❌ DON'T separate achievements from responsibilities unless clearly separate sections
❌ DON'T categorize skills unless they are already categorized in the resume
❌ DON'T extract bullet points from "PROFILE SUMMARY" section and put them in "professional_summary.key_highlights" - keep them in additional_sections
❌ DON'T duplicate content across multiple fields - each piece of text should appear ONCE

WHAT TO DO INSTEAD:
✅ If "years_of_experience" is mentioned in a separate labeled field, extract it
✅ If it's only in the summary text, leave "years_of_experience" as null
✅ If employment type isn't stated, use empty string ""
✅ If everything is in bullet points, put all in "responsibilities" array
✅ Keep skills in the exact groups/categories as shown in the resume
✅ If resume has both a short summary AND a separate "PROFILE SUMMARY" section, put the short summary in professional_summary.summary_text and the detailed "PROFILE SUMMARY" in additional_sections
✅ Each piece of information should appear in EXACTLY ONE place in the JSON

EXTRACTION SCHEMA:
Return a valid JSON object with this exact structure:

{
  "personal_information": {
    "full_name": "string",
    "phone": "string or array of strings",
    "email": "string or array of strings",
    "location": {
      "city": "string",
      "state": "string",
      "country": "string",
      "full_address": "string"
    },
    "linkedin": "string (URL)",
    "github": "string (URL)",
    "portfolio": "string (URL)",
    "other_links": ["array of URLs"],
    "passport_number": "string",
    "date_of_birth": "string",
    "languages": ["array of languages"]
  },
  
  "professional_summary": {
    "summary_text": "string - ONLY the introductory paragraph/summary at the top of resume, NOT bullet points from 'Profile Summary' section",
    "years_of_experience": "ONLY if explicitly stated as a separate labeled field like 'Total Experience: X years', otherwise null",
    "key_highlights": ["ONLY if the PROFESSIONAL SUMMARY or OBJECTIVE section itself contains bullet points. Do NOT extract bullets from separate 'PROFILE SUMMARY' or 'ABOUT ME' sections - those go in additional_sections"]
  },
  
  "work_experience": [
    {
      "job_title": "string - exactly as written",
      "company_name": "string - exactly as written",
      "location": "string - exactly as written",
      "employment_type": "ONLY if explicitly stated (Full-time/Part-time/Contract/etc), otherwise empty string",
      "start_date": "string - preserve EXACT original format",
      "end_date": "string - preserve EXACT original format (or 'Present')",
      "duration": "ONLY if explicitly calculated/mentioned in resume, otherwise empty string",
      "responsibilities": ["array - each bullet point exactly as written, ALL points here unless separate sections exist"],
      "achievements": ["ONLY if resume has separate 'Achievements' section under this job, otherwise empty array"],
      "technologies": ["extract ONLY if in separate 'Tech stack used' line/section, otherwise empty array"],
      "key_metrics": ["extract ONLY if metrics are in separate section, otherwise empty array"]
    }
  ],
  
  "education": [
    {
      "degree": "string",
      "field_of_study": "string",
      "institution": "string",
      "location": "string",
      "graduation_date": "string",
      "start_date": "string",
      "gpa": "string",
      "percentage": "string",
      "achievements": ["array"],
      "relevant_coursework": ["array"]
    }
  ],
  
  "technical_skills": {
    "programming_languages": ["copy EXACTLY as categorized in resume"],
    "frameworks_libraries": ["copy EXACTLY as categorized in resume"],
    "databases": ["copy EXACTLY as categorized in resume"],
    "cloud_platforms": ["copy EXACTLY as categorized in resume"],
    "tools_technologies": ["copy EXACTLY as categorized in resume"],
    "ai_ml_specific": ["only if resume has this category"],
    "devops_tools": ["only if resume has this category"],
    "other_skills": ["only if resume has this category"],
    "categories": {
      "Use_Exact_Category_Name_From_Resume": ["skills under this exact category"]
    }
  },
  
  "projects": [
    {
      "project_name": "string",
      "description": "string",
      "technologies": ["array"],
      "role": "string",
      "duration": "string",
      "key_features": ["array"],
      "achievements": ["array"],
      "links": ["array of URLs"]
    }
  ],
  
  "certifications": [
    {
      "name": "string",
      "issuing_organization": "string",
      "issue_date": "string",
      "expiry_date": "string",
      "credential_id": "string",
      "credential_url": "string"
    }
  ],
  
  "achievements_awards": [
    {
      "title": "string",
      "description": "string",
      "date": "string",
      "issuer": "string"
    }
  ],
  
  "publications": [
    {
      "title": "string",
      "publisher": "string",
      "date": "string",
      "url": "string",
      "description": "string"
    }
  ],
  
  "soft_skills": ["array of soft skills"],
  
  "additional_sections": {
    "section_name": {
      "content": "string or array",
      "details": {}
    }
  },
  
  "metadata": {
    "total_experience_years": "ONLY extract if explicitly labeled as 'Total Experience', otherwise null",
    "current_role": "ONLY extract from most recent job, otherwise empty string",
    "current_company": "ONLY extract from most recent job, otherwise empty string",
    "preferred_location": "ONLY if explicitly stated in resume, otherwise empty string",
    "willing_to_relocate": "ONLY if explicitly stated, otherwise null",
    "notice_period": "ONLY if explicitly stated, otherwise empty string",
    "current_ctc": "ONLY if explicitly stated, otherwise empty string",
    "expected_ctc": "ONLY if explicitly stated, otherwise empty string"
  }
}

EXTRACTION INSTRUCTIONS - LITERAL COPYING ONLY:

1. READ COMPLETELY: Process the entire resume before extracting
2. PRESERVE EXACTNESS: Copy text verbatim - don't paraphrase, summarize, or reword
3. NO DERIVATION: Don't derive, infer, calculate, or move information between fields
4. CAPTURE AS-IS: If resume shows bullet points with metrics inline, keep them together
5. HANDLE DATES: Keep original date formats (MM/YYYY, Mon YYYY, etc.) - don't calculate durations
6. ARRAY ITEMS: Each bullet point should be a separate array element, word-for-word
7. NESTED INFO: If experience/projects have sub-bullets, preserve the exact hierarchy
8. TECHNICAL TERMS: Copy framework names, version numbers, APIs exactly (e.g., "GPT-4O mini", "LangChain")
9. ABBREVIATIONS: Keep as-is (API, CI/CD, ML, AWS, etc.)
10. SPECIAL CHARACTERS: Preserve symbols like +, %, $, @, etc.
11. EMPTY FIELDS: Use null or empty string/array if section doesn't exist or info isn't explicitly labeled
12. SECTION LABELS: Only populate nested fields (achievements, technologies, key_metrics) if they are SEPARATE labeled sections in the resume
13. CLEAN TEXT: Remove newline characters (\n) within text, but preserve paragraph breaks
14. SKIP HEADERS: Don't include section headers like "Key Responsibilities:", "Achievements:", "Tasks:" as bullet points

KEY PRINCIPLE: If you're unsure whether to extract something into a specific field, DON'T. 
Keep it in the main text/responsibilities field where it originally appears.

QUALITY CHECKS:
- Count bullet points in original vs extracted - must match EXACTLY
- Verify all contact info is captured
- Ensure all job roles are included
- Check that no achievements are missing
- Confirm all technical skills are listed
- Validate dates are in original format
- **CRITICAL: Verify NO duplicate content exists across different fields**
- **CRITICAL: Each sentence/bullet point should appear in EXACTLY ONE location in the JSON**

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown, no code blocks, no explanations.
Start directly with { and end with }."""



USER_PROMPT_TEMPLATE = """Extract all information from the following resume with 100% accuracy and completeness.

Resume Content:
{resume_text}

Remember:
- Extract EVERYTHING - miss nothing
- Add nothing that isn't explicitly stated
- Preserve exact wording, numbers, and formatting
- Don't include section headers (like "Key Responsibilities:") as bullet points
- DO NOT duplicate any content - each bullet point should appear in ONLY ONE location
- If there's a "PROFILE SUMMARY" section, keep it in additional_sections, NOT in professional_summary.key_highlights
- Return valid JSON only"""

def clean_extracted_data(data):
    """
    Post-process to clean up extraction issues and remove duplicates
    """
    # Remove newlines within text fields
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        return text
    
    # Track all seen content to detect duplicates
    seen_content = set()
    
    def is_duplicate(text):
        """Check if text is a duplicate"""
        if not text or len(text.strip()) < 20:  # Skip very short text
            return False
        clean = clean_text(text).lower()
        if clean in seen_content:
            return True
        seen_content.add(clean)
        return False
    
    # Clean and deduplicate professional summary
    if 'professional_summary' in data:
        summary = data['professional_summary']
        
        # Add summary_text to seen content
        if summary.get('summary_text'):
            seen_content.add(clean_text(summary['summary_text']).lower())
        
        # Remove duplicate key_highlights
        if 'key_highlights' in summary and summary['key_highlights']:
            summary['key_highlights'] = [
                clean_text(h) for h in summary['key_highlights']
                if h and not is_duplicate(h)
            ]
    
    # Clean work experience
    if 'work_experience' in data:
        for exp in data['work_experience']:
            if 'responsibilities' in exp:
                exp['responsibilities'] = [
                    clean_text(r) for r in exp['responsibilities']
                    if r and not r.strip().endswith(':') and not is_duplicate(r)
                ]
    
    # Clean education
    if 'education' in data:
        for edu in data['education']:
            for field in ['field_of_study', 'institution', 'degree']:
                if field in edu:
                    edu[field] = clean_text(edu[field])
    
    # Clean additional sections
    if 'additional_sections' in data:
        for section_name, section_data in data['additional_sections'].items():
            if isinstance(section_data.get('content'), list):
                section_data['content'] = [
                    clean_text(c) for c in section_data['content']
                    if c and not is_duplicate(c)
                ]
            elif isinstance(section_data.get('content'), str):
                section_data['content'] = clean_text(section_data['content'])
    
    # Clean achievements
    if 'achievements_awards' in data:
        data['achievements_awards'] = [
            {k: clean_text(v) if isinstance(v, str) else v for k, v in achievement.items()}
            for achievement in data['achievements_awards']
            if achievement.get('description') and not is_duplicate(achievement.get('description'))
        ]
    
    return data

async def resume_json(data:str):
    response = client.chat.completions.create(
        model="gpt-5.2",  # or "gpt-4-turbo"
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(resume_text=data)}
        ],
        temperature=0,  # Use 0 for maximum consistency and accuracy
        response_format={"type": "json_object"}  # Forces valid JSON output
    )

    extracted_data = json.loads(response.choices[0].message.content)
    # Apply deduplication and cleanup
    extracted_data = clean_extracted_data(extracted_data)
    
    # print(extracted_data)
    # extracted_data = json.loads(response.choices[0].message.content)
    # print(json.dumps(extracted_data, indent=2))
    return extracted_data