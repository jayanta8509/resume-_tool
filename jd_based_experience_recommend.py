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


class experience_description_recommend(BaseModel):
    recomment: str




# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-5.2",  # Much faster than gpt-5-mini
    # temperature=0.1,        # very low for deterministic output
    max_tokens=800,    # Set explicit limit to prevent over-generation
    request_timeout=25,           # Add timeout to prevent hanging
    # streaming=False,      # Disable streaming for batch processing
    # max_retries=2
)


async def gap_prompt(experience_text: str, job_description: str):
    return f"""
You are an AI agent that identifies UNMATCHED technical skills.

TASK:
Compare Job Description (JD) vs candidate Experience Description.
Return ONLY the JD technical skills that are NOT present in experience_text.

STRICT MATCHING:
- Match ONLY if the exact term appears OR a very close synonym appears.
- "Distributed systems" is matched only if experience_text mentions "distributed".
- "Microservices" is matched only if experience_text mentions "microservice" or "microservices".
- "ETL pipelines" is matched only if experience_text mentions "ETL" explicitly.
- "Cloud platforms (AWS/GCP/Azure)" is matched only if experience_text mentions AWS or GCP or Azure or "cloud".
- "AI/ML frameworks" is matched only if experience_text mentions a framework name OR the exact phrase "AI/ML framework(s)".

NORMALIZATION:
Output skills using these exact labels when applicable:
Python, Java, Go, Microservices, AI/ML Frameworks, AWS, GCP, Azure, ETL

OUTPUT RULES:
- Output ONLY valid JSON.
- JSON must contain ONLY one key: "recomment".
- Format must be exactly:
  This JD needs :- <comma-separated unmatched skills>
- No explanations. No extra words.

Experience Description:
{experience_text}

Job Description:
{job_description}

OUTPUT (STRICT JSON ONLY):
{{"recomment":"This JD needs :- <unmatched skills>"}}
""".strip()









async def experience_response_gap(experience_text: str, job_description: str):
    system_prompt = await gap_prompt(experience_text, job_description)

    agent = create_agent(
        model,
        response_format=ToolStrategy(experience_description_recommend),
        system_prompt=system_prompt
    )

    context_message = f"""
Experience Text:
{experience_text}

Job Description:
{job_description}
""".strip()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )

    return result["structured_response"]




# if __name__ == "__main__":
#     experience_text = """
# Architected AI Question Generation System using GPT-4O mini, OpenCV, and Swarm framework with AWS S3 integration, 
# implementing pattern recognition and context-aware question generation following SDLC best practices. 
# • Built production-grade Virtual Try-On AI platform using Fooocus model and FastAPI, deployed on RunPod GPU servers, 
# generating photorealistic fashion visualizations with 95%+ accuracy for e-commerce applications. 
# • Engineered end-to-end AI video generation pipeline integrating GPT-4O for scripting, Eleven Labs for voice synthesis, 
# Stable Diffusion for image creation, and WAN 2.1 for animation, deployed on scalable RunPod infrastructure. 
# • Developed enterprise recruitment intelligence platform with three specialized GPT-4o AI agents for resume parsing, JD analysis, 
# and candidate matching, delivering 40% faster screening via FastAPI REST backend. 
# • Created multi-agent Resume Maker Tool with GPT-4o-mini, aggregating data from LinkedIn, GitHub, and portfolios to 
# generate ATS-optimized resumes with 95%+ compliance through async processing and structured JSON outputs.
# """

#     job_description = """
# We are looking for a Senior Software Engineer with experience in:
# - Python, Java, or Go programming languages
# - Distributed systems and microservices architecture
# - Machine Learning and AI/ML frameworks
# - Cloud platforms (AWS, GCP, or Azure)
# - Data engineering and ETL pipelines
# """

#     output = asyncio.run(experience_response(experience_text, job_description))
#     print(output.recomment)
