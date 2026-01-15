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


async def Professional_Clean_Tone(description, job_description):
    Professional_P_Clean_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the work experience description to maximize ATS score by strategically incorporating relevant keywords from the job description.

    **Original Experience Description:** {description}

    **Job Description:** {job_description}

    Guidelines for the rewritten experience description:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, tools, frameworks, certifications, and qualifications from the job description
           - Naturally integrate 5-8 of the most relevant JD keywords across the bullet points that align with the candidate's actual experience
           - Place keywords prominently within bullets where ATS scanners are most likely to detect them (beginning of bullets, mid-bullet context)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize hard skills, technical terms, and domain-specific language mentioned in the JD
           - Ensure tech stack keywords match JD terminology exactly

        2. Format: Present as 4-6 concise bullet points (each bullet: 15-25 words).

        3. Content: Preserve ALL factual details, technologies, tools, achievements, and responsibilities from the original (do not add any new information not already in the description).

        4. Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.

        5. Clarity: Improve the structure and wording for easy readability, highlighting the candidate's most important skills and contributions.

        6. Action Verbs: Start each bullet with a strong action verb (e.g., Developed, Implemented, Managed, Optimized, Led, Designed, Created).

        7. ATS Optimization:
           - Match job description terminology exactly when describing technologies, tools, and methodologies
           - Include specific skills and competencies mentioned in the JD
           - Use industry-standard terminology that aligns with the job requirements
           - Ensure all JD-integrated keywords appear naturally within relevant bullet points
           - Highlight technologies and frameworks that are specifically mentioned in the JD

        8. Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person or neutral perspective (avoid using "I" or "my").

        9. Quantifiable Impact: If the original contains numbers/metrics, preserve and highlight them. If not, focus on clear, specific outcomes.

    Now, rewrite the experience description in a professional and clean tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(description, job_description):
    Impactful_P_Strong_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the work experience description to maximize ATS score by strategically incorporating relevant keywords from the job description while maintaining an impactful and strong tone.

    **Original Experience Description:** {description}

    **Job Description:** {job_description}

    Guidelines for the rewritten experience description:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, tools, frameworks, certifications, and qualifications from the job description
           - Naturally integrate 5-8 of the most relevant JD keywords across the bullet points that align with the candidate's actual experience
           - Place keywords prominently within bullets where ATS scanners are most likely to detect them (beginning of bullets, mid-bullet context)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize high-value keywords and technical terms that are frequently mentioned in the JD
           - Ensure tech stack and framework keywords match JD terminology exactly

        2. Format: Present as 4-6 powerful bullet points (each bullet: 15-25 words).

        3. Content: Retain ALL factual details, technologies, tools, achievements, and responsibilities from the original (do not introduce new information not in the description).

        4. Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone. Each bullet should feel powerful and decisive.

        5. Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions. Prioritize achievements over routine responsibilities.

        6. Action Verbs: Start each bullet with high-impact action verbs (e.g., Spearheaded, Engineered, Transformed, Accelerated, Revolutionized, Maximized, Delivered).

        7. Quantifiable Impact: If the original contains numbers/metrics (%, $, time saved, etc.), ensure they're prominently featured. If not, frame outcomes in impactful terms.

        8. ATS Optimization:
           - Match job description terminology exactly when describing technologies, tools, and methodologies
           - Include specific skills and competencies mentioned in the JD
           - Use power verbs and terminology that align with the job requirements
           - Ensure all JD-integrated keywords appear naturally and reinforce impact
           - Highlight technologies and frameworks that are specifically mentioned in the JD

        9. Clarity & Structure: Write clearly and coherently, ensuring the description is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).

        10. Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my").

    Now, produce the rewritten experience description in an impactful and strong tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Impactful_P_Strong_Tone

async def Leadership_Tone(description, job_description):
    Leadership_P_Tone = f"""You are an expert ATS-optimized resume consultant. Your task is to rewrite the work experience description to maximize ATS score by strategically incorporating relevant keywords from the job description while maintaining a leadership-focused tone.

    **Original Experience Description:** {description}

    **Job Description:** {job_description}

    Guidelines for the rewritten experience description:
        1. KEYWORD INTEGRATION (CRITICAL FOR ATS):
           - Extract and identify the most important skills, technologies, tools, frameworks, certifications, qualifications, and leadership competencies from the job description
           - Naturally integrate 5-8 of the most relevant JD keywords across the bullet points that align with the candidate's actual experience
           - Place keywords prominently within bullets where ATS scanners are most likely to detect them (beginning of bullets, mid-bullet context)
           - Use exact keyword terminology from the JD when possible (e.g., if JD says "machine learning" use that, not "ML")
           - Prioritize leadership-related keywords, management terms, and strategic competencies from the JD
           - Ensure tech stack and framework keywords match JD terminology exactly

        2. Format: Present as 4-6 authoritative bullet points (each bullet: 15-25 words).

        3. Content: Keep ALL factual details, technologies, tools, achievements, and responsibilities from the original description (do not add information that isn't provided).

        4. Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.

        5. Emphasis: Highlight the candidate's leadership achievements and responsibilities (e.g., leading teams, driving strategic initiatives, mentoring others, delivering results, managing stakeholders, overseeing projects). Showcase strategic impact and management experience where applicable.

        6. Action Verbs: Start each bullet with leadership-oriented action verbs (e.g., Led, Directed, Orchestrated, Mentored, Guided, Championed, Established, Cultivated).

        7. Strategic Focus: Frame accomplishments to show strategic thinking, decision-making authority, and ability to drive organizational impact.

        8. ATS Optimization:
           - Match job description terminology exactly when describing leadership skills, technologies, and methodologies
           - Include leadership-specific keywords from the JD (e.g., "cross-functional leadership," "strategic planning," "team development," "stakeholder management")
           - Use power verbs that demonstrate leadership and align with job requirements
           - Ensure all JD-integrated keywords appear naturally and reinforce leadership capabilities
           - Highlight technologies and frameworks that are specifically mentioned in the JD

        9. Clarity & Appeal: Ensure the rewritten description is clear and structured, making it easy for recruiters to identify the candidate's leadership value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.

        10. Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person (no "I" statements or possessive "my").

    Now, provide the rewritten experience description in a leadership-oriented tone, strategically incorporating relevant keywords from the job description to maximize ATS score."""

    return Leadership_P_Tone


async def Experience_result(tone: str, description: str, job_description: str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(description, job_description)
          agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(description, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)
    else:
        prompt = await Leadership_Tone(description, job_description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt=prompt)

    context_message = """Please rewrite the experience description according to the guidelines provided, incorporating relevant keywords from the job description to maximize ATS score."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     description = """• Deployed AI Financial Advisor Platform integrating LangChain, CrewAI, AutoGen agents with OpenAI-ada-002 
# embeddings, Pinecone vector DB, IBM Watson transcription API, and Twilio REST API for real-time 
# advisor-client communication. 
# • Fine-tuned Mistral-7B on e-commerce FAQ dataset using PEFT with LoRA and Supervised Fine-tuning Trainer, achieving 30% 
# improvement in query understanding and response accuracy for customer support automation. 
# • Built production loan prediction and repayment models using SGD algorithm, NumPy, Pandas with comprehensive EDA and 
# feature engineering, achieving 85%+ accuracy in credit risk assessment. 
# • Designed dual-mode recommendation engine using SVD algorithm, delivering personalized product suggestions for 10K+ users 
# with selection sort optimization for new and existing customer segments. """

#     job_description = """We are looking for a Senior AI/ML Engineer with experience in:
# - LangChain, LangGraph, and LLM framework development
# - RAG (Retrieval Augmented Generation) pipelines and vector databases
# - Python development and AI agent architecture
# - AWS cloud services and deployment
# - Building scalable AI/ML solutions
# - Experience with vector databases (Pinecone, Weaviate, etc.)
# - Full-stack development with React
# - Product engineering and automation tools
# - Leading AI product development from concept to deployment"""

#     tone = "Leadership Tone"
#     output = asyncio.run(Experience_result(tone, description, job_description))
#     r = output.description
#     print(len(r))