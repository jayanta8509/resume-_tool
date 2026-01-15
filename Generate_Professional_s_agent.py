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


async def Professional_Clean_Tone(resume_data: str):
    Professional_P_Clean_Tone = f"""You are a professional resume consultant. Generate a compelling professional summary based on the candidate's resume data.

    **Resume Data:**
    {resume_data}

    Guidelines for the generated summary:
        Length: 60–80 words in a single, coherent paragraph.
        Content: Focus primarily on professional experience - years of experience, core skills, industry expertise, and key achievements. Integrate educational background naturally only if it's prestigious/relevant (e.g., "IIT graduate," "Stanford CS"), otherwise omit it to maintain flow.
        Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.
        Clarity: Structure the summary to highlight the candidate's most important qualifications, skills, and achievements in an easy-to-read format.
        ATS/Recruiter Appeal: Ensure the language is ATS-friendly and appealing to hiring managers (use strong action verbs and relevant keywords; avoid any special/unusual formatting).
        Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person perspective (avoid using "I" or "my"). Do not invent facts not present in the provided data. Avoid tacking on education details at the end like an afterthought.
        Now, generate a professional summary in a professional and clean tone based on these guidelines."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(resume_data: str):
    Impactful_P_Strong_Tone = f"""You are a professional resume consultant. Generate a compelling professional summary based on the candidate's resume data.
    **Resume Data:**
    {resume_data}

    Guidelines for the generated summary:
        Length: 60–80 words, presented as a single concise paragraph.
        Content: Focus primarily on professional impact - total years of experience, technical/industry expertise, and measurable achievements. Weave in educational credentials naturally only if they're notable (e.g., top-tier institutions), otherwise exclude them to keep the focus on impact.
        Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone.
        Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions (use quantifiable details if available from the data).
        Clarity & Structure: Write clearly and coherently, ensuring the summary is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).
        Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my"). Do not invent facts not present in the provided data. Never append education as a fragment at the end.
        Now, generate a professional summary in an impactful and strong tone following these guidelines."""
    return Impactful_P_Strong_Tone

async def Leadership_Tone(resume_data: str):
    Leadership_P_Tone = f"""You are a professional resume consultant. Generate a compelling professional summary based on the candidate's resume data.
    **Resume Data:**
    {resume_data}

    Guidelines for the generated summary:
        Length: 60–80 words in one well-crafted paragraph.
        Content: Focus on leadership experience - roles, team management, strategic initiatives, and career progression. Integrate education naturally only if from a prestigious institution (e.g., "IIT Delhi alumnus"), otherwise omit to maintain professional flow.
        Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.
        Emphasis: Highlight the candidate's leadership achievements and responsibilities (e.g. leading teams, driving strategic initiatives, mentoring others, delivering results, organizational impact). Showcase strategic impact and management experience where applicable.
        Clarity & Appeal: Ensure the summary is clear and structured, making it easy for recruiters to identify the candidate's leadership value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.
        Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person perspective typical of resumes (no "I" statements or possessive "my"). Do not invent facts not present in the provided data. Never tack on education as an afterthought fragment.
        Now, generate a professional summary in a leadership-oriented tone, following these guidelines."""
    return Leadership_P_Tone


async def Generate_professional_responce(tone: str, resume_data: str):
    """
    Generate a professional summary based on experience and education data.

    Args:
        tone: The tone style ("Professional & Clean", "Impactful & Strong", or "Leadership Tone")
        resume_data: String containing resume with EXPERIENCE and EDUCATION sections

    Returns:
        professional_summary object with generated summary
    """
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(resume_data)
          agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(resume_data)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(resume_data)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    context_message = """Please generate a professional summary according to the guidelines provided."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     resume_data = """EXPERIENCE 
# AI/ML  Engineer 
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
# Diploma in Computer Science and Technology; Percentage: 71.90% 
#     """

#     tone = "Impactful & Strong"
#     output = asyncio.run(Generate_professional_responce(tone, resume_data))
#     print(output.summary)