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
    description: str


# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-5.2",  # Much faster than gpt-5-mini
    # temperature=0.1,        # very low for deterministic output
    max_tokens=1500,   # Increased for detailed experience descriptions with bullet points
    request_timeout=30,           # Add timeout to prevent hanging
    # streaming=False,      # Disable streaming for batch processing
    # max_retries=2
)


async def Professional_Clean_Tone(description):
    Professional_P_Clean_Tone = f"""You are a professional resume consultant. Analyze the following work experience description and rewrite it in a professional and clean tone.
    **Original Experience Description:** {description}

    Guidelines for the rewritten experience description:
        Format: Present as 4-6 concise bullet points (each bullet: 15-25 words).
        Content: Preserve ALL factual details, technologies, tools, achievements, and responsibilities from the original (do not add any new information not already in the description).
        Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.
        Clarity: Improve the structure and wording for easy readability, highlighting the candidate's most important skills and contributions.
        Action Verbs: Start each bullet with a strong action verb (e.g., Developed, Implemented, Managed, Optimized, Led, Designed, Created).
        ATS/Recruiter Appeal: Ensure the language is ATS-friendly and appealing to hiring managers (use relevant keywords; avoid any special/unusual formatting).
        Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person or neutral perspective (avoid using "I" or "my").
        Quantifiable Impact: If the original contains numbers/metrics, preserve and highlight them. If not, focus on clear, specific outcomes.
        Now, rewrite the experience description in a professional and clean tone based on these guidelines."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(description):
    Impactful_P_Strong_Tone = f"""You are a professional resume consultant. Analyze the following work experience description and rewrite it in an impactful and strong tone.
    **Original Experience Description:** {description}

    Guidelines for the rewritten experience description:
        Format: Present as 4-6 powerful bullet points (each bullet: 15-25 words).
        Content: Retain ALL factual details, technologies, tools, achievements, and responsibilities from the original (do not introduce new information not in the description).
        Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone. Each bullet should feel powerful and decisive.
        Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions. Prioritize achievements over routine responsibilities.
        Action Verbs: Start each bullet with high-impact action verbs (e.g., Spearheaded, Engineered, Transformed, Accelerated, Revolutionized, Maximized, Delivered).
        Quantifiable Impact: If the original contains numbers/metrics (%, $, time saved, etc.), ensure they're prominently featured. If not, frame outcomes in impactful terms.
        Clarity & Structure: Write clearly and coherently, ensuring the description is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).
        Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my").
        Now, produce the rewritten experience description in an impactful and strong tone following these guidelines."""
    return Impactful_P_Strong_Tone

async def Leadership_Tone(description):
    Leadership_P_Tone = f"""You are a professional resume consultant. Analyze the following work experience description and rewrite it in a leadership-focused tone.
    **Original Experience Description:** {description}

    Guidelines for the rewritten experience description:
        Format: Present as 4-6 authoritative bullet points (each bullet: 15-25 words).
        Content: Keep ALL factual details, technologies, tools, achievements, and responsibilities from the original description (do not add information that isn't provided).
        Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.
        Emphasis: Highlight the candidate's leadership achievements and responsibilities (e.g., leading teams, driving strategic initiatives, mentoring others, delivering results, managing stakeholders, overseeing projects). Showcase strategic impact and management experience where applicable.
        Action Verbs: Start each bullet with leadership-oriented action verbs (e.g., Led, Directed, Orchestrated, Mentored, Guided, Championed, Established, Cultivated).
        Strategic Focus: Frame accomplishments to show strategic thinking, decision-making authority, and ability to drive organizational impact.
        Clarity & Appeal: Ensure the rewritten description is clear and structured, making it easy for recruiters to identify the candidate's leadership value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.
        Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person (no "I" statements or possessive "my").
        Now, provide the rewritten experience description in a leadership-oriented tone, following these guidelines."""
    return Leadership_P_Tone


async def Experience_result(tone :str, description:str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(description)
          agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)
    else:
        prompt = await Leadership_Tone(description)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)

    context_message = """Please rewrite the experience description according to the guidelines provided."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     description = """ Bootstrapped an AI Colpilot for B2B SaaS companies to acquire 
# customers at global conferences. 
#  Built LangGraph-powered agent workflows and orchestrating 
# RAG pipelines with Pinecone for vector retrieval. 
#  Also, built LLM micro-agents to automate event-intelligence 
# tasks feature extraction, lead scoring, and personalized 
# outreach at scale. 
#  Tech stack : Langchain, Python, Pinecone DB, AWS, React, LLMs 
#  Product stack : MixPanel, Airtable, n8n, Gumloop, Zapier, etc. """
    
#     tone = "Leadership Tone"
#     output = asyncio.run( Experience_result(tone,description))
#     print(output.description)