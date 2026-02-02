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


class jd_gap_recommend(BaseModel):
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


async def prompt(summary: str, job_description: str):
    return f"""
You are an AI agent that finds SKILL GAPS between a candidate summary and a Job Description.

TASK:
1) Extract technical skills from the Job Description (JD Skills).
2) Extract technical skills from the candidate Summary (Candidate Skills).
3) Return ONLY the JD skills that are NOT present in Candidate Skills (Unmatched JD Skills).

RULES:
- Output ONLY valid JSON with ONLY this key: "recomment"
- "recomment" must be EXACTLY in this format:
  This JD needs :- <comma-separated list of unmatched JD skills>

- Include ONLY technical items: languages, frameworks, cloud, tools, architecture, data/ML areas.
- Exclude soft skills: Agile, mentoring, leadership, collaboration, communication.
- Do NOT include company names, years of experience.
- Do NOT invent skills.
- Matching rules:
  - Treat "AI/ML Frameworks" as matched if Candidate Skills mention frameworks by name OR mention "AI/ML frameworks".
  - Treat "Machine Learning" as matched if Candidate Skills mention "ML" or "Machine Learning".
  - Treat "Cloud platforms" as matched ONLY if candidate explicitly mentions AWS/GCP/Azure or "cloud".
  - Treat "Microservices" as matched ONLY if candidate explicitly mentions "microservices".
  - Treat "Distributed Systems" as matched if candidate mentions "distributed systems".
  - Treat "Data Engineering" as matched if candidate mentions "data engineering".
  - Treat "ETL Pipelines" as matched if candidate mentions "ETL" or "pipelines".
  - Treat "Python/Java/Go" as matched only if explicitly mentioned in candidate summary.

- If all JD skills are matched, output:
  {{"recomment":"All core JD skills are already matched in the summary."}}

INPUTS:

Candidate Summary:
{summary}

Job Description:
{job_description}

OUTPUT (STRICT JSON ONLY):
{{"recomment":"This JD needs :- <unmatched skills>"}}
""".strip()





async def professional_responce_gap(summary: str, job_description: str):
    system_prompt = await prompt(summary, job_description)

    agent = create_agent(
    model,
    response_format=ToolStrategy(jd_gap_recommend),
    system_prompt=system_prompt
)

    context_message = f"""
Summary:
{summary}

Job Description:
{job_description}
""".strip()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )

    return result["structured_response"]




# if __name__ == "__main__":
#     summary = """Results-driven AI/ML Engineer with 2+ years building production AI systems serving 10K+ users. Specialized in LLM applications, multi-agent 
# architectures, and end-to-end ML pipelines using GPT-4, LangChain, and modern frameworks. Proven expertise delivering 95%+ model accuracy 
# and 40% efficiency improvements through scalable solutions with FastAPI, AWS, and GPU infrastructure."""

#     job_description = """We are looking for a Senior Software Engineer with experience in:
# - Python, Java, or Go programming languages
# - Distributed systems and microservices architecture
# - Machine Learning and AI/ML frameworks
# - Cloud platforms (AWS, GCP, or Azure)
# - Data engineering and ETL pipelines
# - Agile development methodologies
# - Leading technical teams and mentoring junior developers"""

#     output = asyncio.run(professional_responce(summary, job_description))
#     print(output.recomment)