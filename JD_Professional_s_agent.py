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


async def Professional_Clean_Tone(Summary, job_description):
    Professional_P_Clean_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the professional summary to maximize ATS score by strategically incorporating relevant keywords from the job description.

    **Original Professional Summary:** {Summary}

    **Job Description:** {job_description}

    Guidelines for the rewritten summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")

        2. Length: 60–80 words in a single, coherent paragraph.

        3. Content: Preserve all factual details and key experiences from the original (do not add any new information not already in the summary).

        4. Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.

        5. Clarity: Improve the structure and wording for easy readability, highlighting the candidate's most important skills and achievements.

        6. ATS Optimization:
           - Use strong action verbs and relevant keywords from the JD
           - Avoid any special/unusual formatting that might confuse ATS parsers
           - Match job description terminology when describing skills and experience
           - Include hard skills, technologies, and domain-specific terms mentioned in the JD

        7. Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person or neutral perspective (avoid using "I" or "my").

    Now, rewrite the professional summary in a professional and clean tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(Summary, job_description):
    Impactful_P_Strong_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the professional summary to maximize ATS score by strategically incorporating relevant keywords from the job description while maintaining an impactful and strong tone.

    **Original Professional Summary:** {Summary}

    **Job Description:** {job_description}

    Guidelines for the rewritten summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize high-value keywords that are frequently mentioned in the JD

        2. Length: 60–80 words, presented as a single concise paragraph.

        3. Content: Retain all factual details and key experiences (do not introduce new information not in the original).

        4. Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone.

        5. Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions (use quantifiable details if available).

        6. Clarity & Structure: Write clearly and coherently, ensuring the summary is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).

        7. ATS Optimization:
           - Match job description terminology exactly when describing skills and experience
           - Include hard skills, technologies, and domain-specific terms mentioned in the JD
           - Use power verbs that align with the job requirements
           - Ensure all JD-integrated keywords feel natural, not forced

        8. Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my").

    Now, produce the rewritten professional summary in an impactful and strong tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Impactful_P_Strong_Tone

async def Leadership_Tone(Summary, job_description):
    Leadership_P_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the professional summary to maximize ATS score by strategically incorporating relevant keywords from the job description while maintaining a leadership-focused tone.

    **Original Professional Summary:** {Summary}

    **Job Description:** {job_description}

    Guidelines for the rewritten summary:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, certifications, and qualifications from the job description
           - Naturally integrate 3-5 of the most relevant JD keywords that align with the candidate's actual experience
           - Place keywords strategically where ATS scanners are most likely to detect them (beginning, middle)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize leadership-related keywords and management terms from the JD

        2. Length: 60–80 words in one well-crafted paragraph.

        3. Content: Keep all factual details and key experiences from the original summary (do not add information that isn't provided).

        4. Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.

        5. Emphasis: Highlight the candidate's leadership achievements and responsibilities (e.g. leading teams, driving strategic initiatives, mentoring others, delivering results). Showcase strategic impact and management experience where applicable.

        6. Clarity & Appeal: Ensure the rewritten summary is clear and structured, making it easy for recruiters to identify the candidate's value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.

        7. ATS Optimization:
           - Match job description terminology exactly when describing leadership skills and experience
           - Include leadership-specific keywords from the JD (e.g., "cross-functional leadership," "strategic planning," "team development")
           - Use power verbs that demonstrate leadership and align with job requirements
           - Ensure all JD-integrated keywords feel natural and reinforce leadership capabilities

        8. Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person or an implied first-person style typical of resumes (no "I" statements or possessive "my").

    Now, provide the rewritten professional summary in a leadership-oriented tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Leadership_P_Tone


async def professional_responce(tone: str, summary: str, job_description: str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(summary, job_description)
          agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(summary, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(summary, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(professional_summary),
                  system_prompt=prompt)

    context_message = """Please rewrite the professional summary according to the guidelines provided, incorporating relevant keywords from the job description to maximize ATS score."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     summary = """9+ yrs of experience in Software Developemt,
# Backend, Distributed Systems, AI, LLMs, and Data
# Engineering. Worked with D.E Shaw, Deliveroo, Ula
# like companies in India and Europe. """

#     job_description = """We are looking for a Senior Software Engineer with experience in:
# - Python, Java, or Go programming languages
# - Distributed systems and microservices architecture
# - Machine Learning and AI/ML frameworks
# - Cloud platforms (AWS, GCP, or Azure)
# - Data engineering and ETL pipelines
# - Agile development methodologies
# - Leading technical teams and mentoring junior developers"""

#     tone = "Leadership Tone"
#     output = asyncio.run(professional_responce(tone, summary, job_description))
#     print(output.summary)