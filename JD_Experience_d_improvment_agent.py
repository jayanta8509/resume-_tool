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


class Experience_description(BaseModel):
    description: list[str]


# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-5.2",  # Much faster than gpt-5-mini
    # temperature=0.1,        # very low for deterministic output
    max_tokens=1500,   # Increased for detailed experience descriptions with bullet points
    request_timeout=30,           # Add timeout to prevent hanging
    # streaming=False,      # Disable streaming for batch processing
    # max_retries=2
)


async def Professional_Clean_Tone(original_experience, JD_needs, job_description):
    Professional_P_Clean_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new work experience description based on the candidate's existing experience and the job requirements.

    **Candidate's Original Experience Description:**
    {original_experience}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new experience description
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently within bullets where ATS scanners will detect them

    2. Format: Present as 4-6 concise bullet points (each bullet: 15-25 words).

    3. Content: Build upon the candidate's existing experience while ensuring ALL JD_needs skills are naturally integrated.

    4. Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.

    5. Action Verbs: Start each bullet with a strong action verb (e.g., Developed, Implemented, Managed, Optimized, Led, Designed, Created).

    6. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Match job description terminology exactly when describing technologies and tools
       - Ensure all keywords appear naturally within relevant bullet points

    7. Avoid: No generic filler phrases or clichés. Write in third person or neutral perspective (avoid using "I" or "my").

    8. Quantifiable Impact: Preserve and highlight any numbers/metrics from the original.

    Now, CREATE a new experience description that includes ALL skills from JD_needs, maintaining a professional and clean tone."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(original_experience, JD_needs, job_description):
    Impactful_P_Strong_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new work experience description based on the candidate's existing experience and the job requirements, maintaining an impactful and strong tone.

    **Candidate's Original Experience Description:**
    {original_experience}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new experience description
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently within bullets where ATS scanners will detect them
       - Prioritize high-value keywords at the beginning of bullets

    2. Format: Present as 4-6 powerful bullet points (each bullet: 15-25 words).

    3. Content: Build upon the candidate's existing experience while ensuring ALL JD_needs skills are naturally integrated. Focus on achievements and measurable impact.

    4. Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone. Each bullet should feel powerful and decisive.

    5. Emphasis: Highlight key accomplishments and results, emphasizing measurable impact or contributions.

    6. Action Verbs: Start each bullet with high-impact action verbs (e.g., Spearheaded, Engineered, Transformed, Accelerated, Revolutionized, Maximized, Delivered).

    7. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Use power verbs and terminology that align with the job requirements
       - Ensure all keywords appear naturally and reinforce impact

    8. Clarity & Structure: Write clearly and coherently, ensuring the description grabs attention while incorporating all required skills.

    9. Avoid: Steer clear of generic clichés or buzzwords. Maintain a professional resume style in third person (no "I" or "my").

    10. Quantifiable Impact: Preserve and highlight any numbers/metrics from the original.

    Now, CREATE a new experience description that includes ALL skills from JD_needs, maintaining an impactful and strong tone."""

    return Impactful_P_Strong_Tone

async def Leadership_Tone(original_experience, JD_needs, job_description):
    Leadership_P_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to CREATE a new work experience description based on the candidate's existing experience and the job requirements, maintaining a leadership-focused tone.

    **Candidate's Original Experience Description:**
    {original_experience}

    **JD Required Skills (MUST INCLUDE ALL):**
    {JD_needs}

    **Job Description:** {job_description}

    CRITICAL REQUIREMENTS:
    1. **MANDATORY KEYWORD INTEGRATION:**
       - You MUST include ALL skills and keywords listed in "JD Required Skills"
       - Every single skill/keyword from JD_needs must appear in the new experience description
       - Use the EXACT terminology as provided in JD_needs (do not abbreviate or modify)
       - Place keywords prominently within bullets where ATS scanners will detect them
       - Prioritize leadership-related and management keywords at the beginning

    2. Format: Present as 4-6 authoritative bullet points (each bullet: 15-25 words).

    3. Content: Build upon the candidate's existing experience while ensuring ALL JD_needs skills are naturally integrated. Focus on leadership qualities and strategic impact.

    4. Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. Professional, decisive, and inspiring.

    5. Emphasis: Highlight leadership achievements and responsibilities (e.g., leading teams, driving strategic initiatives, mentoring others, delivering results). Showcase strategic impact.

    6. Action Verbs: Start each bullet with leadership-oriented action verbs (e.g., Led, Directed, Orchestrated, Mentored, Guided, Championed, Established, Cultivated).

    7. ATS Optimization:
       - All JD_needs keywords must be present verbatim
       - Include leadership-specific keywords from the JD
       - Use power verbs that demonstrate leadership capabilities
       - Ensure all keywords reinforce leadership presence

    8. Clarity & Appeal: Ensure the description is clear and structured, making it easy for recruiters to identify leadership value.

    9. Avoid: Avoid generic or overused phrases. Write in third person (no "I" statements or possessive "my").

    10. Strategic Focus: Frame accomplishments to show strategic thinking and decision-making authority.

    Now, CREATE a new experience description that includes ALL skills from JD_needs, maintaining a leadership-oriented tone."""

    return Leadership_P_Tone


async def Experience_improve_result(tone: str, original_experience: str, JD_needs: str, job_description: str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(original_experience, JD_needs, job_description)
          agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(original_experience, JD_needs, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(original_experience, JD_needs, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)

    context_message = """Please CREATE a new experience description that includes ALL skills from JD_needs, following the guidelines provided to maximize ATS score."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":


#     description = """• Architected AI Question Generation System using GPT-4O mini, OpenCV, and Swarm framework with AWS S3 integration,
# implementing pattern recognition and context-aware question generation following SDLC best practices.
# • Built production-grade Virtual Try-On AI platform using Fooocus model and FastAPI, deployed on RunPod GPU servers,
# generating photorealistic fashion visualizations with 95%+ accuracy for e-commerce applications.
# • Engineered end-to-end AI video generation pipeline integrating GPT-4O for scripting, Eleven Labs for voice synthesis,
# Stable Diffusion for image creation, and WAN 2.1 for animation, deployed on scalable RunPod infrastructure.
# • Developed enterprise recruitment intelligence platform with three specialized GPT-4o AI agents for resume parsing, JD analysis,
# and candidate matching, delivering 40% faster screening via FastAPI REST backend.
# • Created multi-agent Resume Maker Tool with GPT-4o-mini, aggregating data from LinkedIn, GitHub, and portfolios to
# generate ATS-optimized resumes with 95%+ compliance through async processing and structured JSON outputs.
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
#     output = asyncio.run(Experience_improve_result(tone, description, JD_needs, job_description))
#     r = output.description
#     print(r)
#     print(len(r))