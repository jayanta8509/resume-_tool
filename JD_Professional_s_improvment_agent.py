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


async def Professional_Clean_Tone(summary, JD_needs, job_description):
    Professional_P_Clean_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary based on the candidate's existing summary and the job requirements.

    **Candidate's Current Professional Summary:**
    {summary}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new summary
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently where ATS scanners will detect them

    2. Length: 70–100 words in a single, coherent paragraph (may be slightly longer to accommodate all required keywords).

    3. Content: Build upon the candidate's existing summary while ensuring ALL JD_needs skills are naturally integrated.

    4. Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.

    5. Structure: Ensure the summary flows naturally while incorporating all required skills.

    6. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Use strong action verbs and relevant keywords from the JD
       - Avoid any special/unusual formatting that might confuse ATS parsers

    7. Avoid: No generic filler phrases or clichés. Write in third person or neutral perspective (avoid using "I" or "my").

    Now, CREATE a new professional summary that includes ALL skills from JD_needs, maintaining a professional and clean tone."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(summary, JD_needs, job_description):
    Impactful_P_Strong_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary based on the candidate's existing summary and the job requirements, maintaining an impactful and strong tone.

    **Candidate's Current Professional Summary:**
    {summary}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new summary
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently where ATS scanners will detect them
       - Prioritize high-value keywords at the beginning of the summary

    2. Length: 70–100 words in a single, powerful paragraph (may be slightly longer to accommodate all required keywords).

    3. Content: Build upon the candidate's existing summary while ensuring ALL JD_needs skills are naturally integrated. Focus on achievements and measurable impact.

    4. Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone.

    5. Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions.

    6. Clarity & Structure: Write clearly and coherently, ensuring the summary grabs attention while incorporating all required skills.

    7. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Use power verbs and terminology that align with the job requirements
       - Ensure all keywords feel natural and reinforce impact

    8. Avoid: Steer clear of generic clichés or buzzwords. Maintain a professional resume style in third person (no "I" or "my").

    Now, CREATE a new professional summary that includes ALL skills from JD_needs, maintaining an impactful and strong tone."""

    return Impactful_P_Strong_Tone

async def Leadership_Tone(summary, JD_needs, job_description):
    Leadership_P_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new professional summary based on the candidate's existing summary and the job requirements, maintaining a leadership-focused tone.

    **Candidate's Current Professional Summary:**
    {summary}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new summary
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently where ATS scanners will detect them
       - Prioritize leadership-related and management keywords at the beginning

    2. Length: 70–100 words in one authoritative paragraph (may be slightly longer to accommodate all required keywords).

    3. Content: Build upon the candidate's existing summary while ensuring ALL JD_needs skills are naturally integrated. Focus on leadership qualities and strategic impact.

    4. Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. Professional, decisive, and inspiring.

    5. Emphasis: Highlight the candidate's leadership achievements and responsibilities (e.g., leading teams, driving strategic initiatives, mentoring others, delivering results). Showcase strategic impact.

    6. Clarity & Appeal: Ensure the summary is clear and structured, making it easy for recruiters to identify leadership value. Use ATS-friendly language.

    7. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Include leadership-specific keywords from the JD
       - Use power verbs that demonstrate leadership capabilities
       - Ensure all keywords reinforce leadership presence

    8. Avoid: Avoid generic or overused phrases. Write in third person (no "I" statements or possessive "my").

    Now, CREATE a new professional summary that includes ALL skills from JD_needs, maintaining a leadership-oriented tone."""

    return Leadership_P_Tone


async def professional_improve_responce(tone: str, summary: str, JD_needs: str, job_description: str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(summary, JD_needs, job_description)
          agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(summary, JD_needs, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(summary, JD_needs, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    context_message = """Please CREATE a new professional summary that includes ALL skills from JD_needs, following the guidelines provided to maximize ATS score."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans


# if __name__ == "__main__":
#     SUMMARY = """
#     Results-driven AI/ML Engineer with 2+ years building production AI systems serving 10K+ users. Specialized in LLM applications, multi-agent
# architectures, and end-to-end ML pipelines using GPT-4, LangChain, and modern frameworks. Proven expertise delivering 95%+ model accuracy
# and 40% efficiency improvements through scalable solutions with FastAPI, AWS, and GPU infrastructure.
#     """
#     JD_needs = """HTML, CSS, JavaScript, TypeScript, HTML5, CSS3, Bootstrap, jQuery, Git, MySQL, MongoDB, SEO"""

#     job_description = """We are looking for a Senior AI/ML Engineer with experience in:
#     - LangChain, LangGraph, and LLM framework development
#     - RAG (Retrieval Augmented Generation) pipelines and vector databases
#     - Python development and AI agent architecture
#     - AWS cloud services and deployment
#     - Building scalable AI/ML solutions
#     - Experience with vector databases (Pinecone, Weaviate, etc.)
#     - Full-stack development with React
#     - Product engineering and automation tools
#     - Leading AI product development from concept to deployment"""

#     tone = "Leadership Tone"
#     output = asyncio.run(professional_responce(tone, SUMMARY, JD_needs, job_description))
#     print(output.summary)