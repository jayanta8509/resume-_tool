import os
import asyncio
import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class professional_summary(BaseModel):
    summary: str


# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-5.2",  # Much faster than gpt-5-mini
    # temperature=0.1,        # very low for deterministic output
    max_tokens=800,    # Set explicit limit to prevent over-generation
    request_timeout=25,           # Add timeout to prevent hanging
    # streaming=False,      # Disable streaming for batch processing
    # max_retries=2
)


async def Professional_Clean_Tone(resume_data, job_description):
    Professional_P_Clean_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary from scratch based on the candidate's resume data and the job description.

    **Candidate's Resume Data:**
    {resume_data}

    **Job Description:** {job_description}

    Guidelines for creating the summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")

        2. Length: 60–80 words in a single, coherent paragraph.

        3. Content: Synthesize key information from the candidate's experience, technical skills, and education. Highlight their most relevant qualifications for the target role.

        4. Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.

        5. Clarity: Structure the summary to highlight the candidate's most important skills, achievements, and qualifications based on their resume data.

        6. ATS Optimization:
           - Use strong action verbs and relevant keywords from the JD
           - Avoid any special/unusual formatting that might confuse ATS parsers
           - Match job description terminology when describing skills and experience
           - Include hard skills, technologies, and domain-specific terms mentioned in the JD

        7. Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person or neutral perspective (avoid using "I" or "my").

    Now, CREATE a new professional summary in a professional and clean tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(resume_data, job_description):
    Impactful_P_Strong_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary from scratch based on the candidate's resume data and the job description, maintaining an impactful and strong tone.

    **Candidate's Resume Data:**
    {resume_data}

    **Job Description:** {job_description}

    Guidelines for creating the summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize high-value keywords that are frequently mentioned in the JD

        2. Length: 60–80 words, presented as a single concise paragraph.

        3. Content: Synthesize key information from the candidate's experience, technical skills, and education. Focus on achievements and measurable impact.

        4. Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone.

        5. Emphasis: Highlight the candidate's key accomplishments and results from their experience, emphasizing measurable impact or contributions (use quantifiable details if available).

        6. Clarity & Structure: Write clearly and coherently, ensuring the summary is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).

        7. ATS Optimization:
           - Match job description terminology exactly when describing skills and experience
           - Include hard skills, technologies, and domain-specific terms mentioned in the JD
           - Use power verbs that align with the job requirements
           - Ensure all JD-integrated keywords feel natural, not forced

        8. Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my").

    Now, CREATE a new professional summary in an impactful and strong tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Impactful_P_Strong_Tone

async def Leadership_Tone(resume_data, job_description):
    Leadership_P_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary from scratch based on the candidate's resume data and the job description, maintaining a leadership-focused tone.

    **Candidate's Resume Data:**
    {resume_data}

    **Job Description:** {job_description}

    Guidelines for creating the summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize leadership-related keywords and management terms from the JD

        2. Length: 60–80 words in one well-crafted paragraph.

        3. Content: Synthesize key information from the candidate's experience, technical skills, and education, focusing on leadership qualities and strategic impact.

        4. Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.

        5. Emphasis: Highlight the candidate's leadership achievements and responsibilities from their experience (e.g. leading teams, driving strategic initiatives, mentoring others, delivering results). Showcase strategic impact and management experience where applicable.

        6. Clarity & Appeal: Ensure the summary is clear and structured, making it easy for recruiters to identify the candidate's value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.

        7. ATS Optimization:
           - Match job description terminology exactly when describing leadership skills and experience
           - Include leadership-specific keywords from the JD (e.g., "cross-functional leadership," "strategic planning," "team development")
           - Use power verbs that demonstrate leadership and align with job requirements
           - Ensure all JD-integrated keywords feel natural and reinforce leadership capabilities

        8. Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person or an implied first-person style typical of resumes (no "I" statements or possessive "my").

    Now, CREATE a new professional summary in a leadership-oriented tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Leadership_P_Tone


async def professional_responce(tone: str, resume_data: str, job_description: str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(resume_data, job_description)
          agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(resume_data, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(resume_data, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    context_message = """Please CREATE a new professional summary from the resume data provided, following the guidelines and incorporating relevant keywords from the job description to maximize ATS score."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     resume_data = """EXPERIENCE
# AI/ML Engineer
# Iksen India Pvt Ltd
# Jul 2024 – Present
# Kolkata, India
# • Architected AI Question Generation System using GPT-4O mini, OpenCV, and Swarm framework with AWS S3 integration,
# implementing pattern recognition and context-aware question generation following SDLC best practices.
# • Built production-grade Virtual Try-On AI platform using Fooocus model and FastAPI, deployed on RunPod GPU servers,
# generating photorealistic fashion visualizations with 95%+ accuracy for e-commerce applications.
# • Engineered end-to-end AI video generation pipeline integrating GPT-4O for scripting, Eleven Labs for voice synthesis,
# Stable Diffusion for image creation, and WAN 2.1 for animation, deployed on scalable RunPod infrastructure.
# • Developed enterprise recruitment intelligence platform with three specialized GPT-4o AI agents for resume parsing, JD analysis,
# and candidate matching, delivering 40% faster screening via FastAPI REST backend.
# • Created multi-agent Resume Maker Tool with GPT-4o-mini, aggregating data from LinkedIn, GitHub, and portfolios to
# generate ATS-optimized resumes with 95%+ compliance through async processing and structured JSON outputs.
# Machine Learning Engineer
# Paythrough Softwares and Solutions Pvt Ltd
# Jun 2023 – Jun 2024
# Kolkata, India
# • Deployed AI Financial Advisor Platform integrating LangChain, CrewAI, AutoGen agents with OpenAI-ada-002
# embeddings, Pinecone vector DB, IBM Watson transcription API, and Twilio REST API for real-time
# advisor-client communication.
# • Fine-tuned Mistral-7B on e-commerce FAQ dataset using PEFT with LoRA and Supervised Fine-tuning Trainer, achieving 30%
# improvement in query understanding and response accuracy for customer support automation.
# • Built production loan prediction and repayment models using SGD algorithm, NumPy, Pandas with comprehensive EDA and
# feature engineering, achieving 85%+ accuracy in credit risk assessment.
# • Designed dual-mode recommendation engine using SVD algorithm, delivering personalized product suggestions for 10K+ users
# with selection sort optimization for new and existing customer segments.

# TECHNICAL SKILLS
# Languages: Python, C/C++, SQL
# AI/ML Frameworks: LangChain, RAG, LangGraph, CrewAI, AutoGen, Swarm, Pydantic AI, MLOps, Scikit-Learn, TensorFlow,
# Keras, PyTorch
# LLM & Models: OpenAI GPT-4/4o, Gemini, DeepSeek, Anthropic Claude, Grok, Mistral, BERT, T5, Stable Diffusion, Hugging
# Face
# Databases: PostgreSQL, MySQL, Redis, Pinecone, FAISS, Chroma, Qdrant (Vector DBs)
# DevOps & Cloud: Docker, Git, AWS (SageMaker, EC2, Lambda, S3, LightSail), RunPod GPU Servers, CI/CD Pipelines
# APIs & Web: FastAPI, Flask, Quart, Django REST, RESTful APIs, OpenCV, Beautiful Soup
# ML Techniques: NLP, Computer Vision, Supervised/Unsupervised Learning, Feature Engineering, XGBoost, SGD, PEFT, LoRA,
# Model Fine-tuning

# EDUCATION
# Narula Institute of Technology (MAKAUT)
# Bachelor of Technology in Computer Science and Engineering; CGPA: 8.10/10.0
# South Calcutta Polytechnic (WBSCTE)
# Diploma in Computer Science and Technology; Percentage: 71.90%"""

#     job_description = """We are looking for a Senior Software Engineer with experience in:
# - Python, Java, or Go programming languages
# - Distributed systems and microservices architecture
# - Machine Learning and AI/ML frameworks
# - Cloud platforms (AWS, GCP, or Azure)
# - Data engineering and ETL pipelines
# - Agile development methodologies
# - Leading technical teams and mentoring junior developers"""

#     tone = "Leadership Tone"
#     output = asyncio.run(professional_responce(tone, resume_data, job_description))
#     print(output.summary)