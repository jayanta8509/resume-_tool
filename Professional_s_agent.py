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


async def Professional_Clean_Tone(Summary):
    Professional_P_Clean_Tone = f"""You are a professional resume consultant. Analyze the following professional summary and rewrite it in a professional and clean tone.
    **Original Professional Summary:** {Summary}

    Guidelines for the rewritten summary:
        Length: 60–80 words in a single, coherent paragraph.
        Content: Preserve all factual details and key experiences from the original (do not add any new information not already in the summary).
        Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.
        Clarity: Improve the structure and wording for easy readability, highlighting the candidate’s most important skills and achievements.
        ATS/Recruiter Appeal: Ensure the language is ATS-friendly and appealing to hiring managers (use strong action verbs and relevant keywords; avoid any special/unusual formatting).
        Avoid: No generic filler phrases or clichés (e.g., “hard-working individual,” “results-driven professional,” etc.). Write in third person or neutral perspective (avoid using “I” or “my”).
        Now, rewrite the professional summary in a professional and clean tone based on these guidelines."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(Summary):
    Impactful_P_Strong_Tone = f"""You are a professional resume consultant. Analyze the following professional summary and rewrite it in an impactful and strong tone.
    **Original Professional Summary:** {Summary}

    Guidelines for the rewritten summary:
        Length: 60–80 words, presented as a single concise paragraph.
        Content: Retain all factual details and key experiences (do not introduce new information not in the original).
        Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone.
        Emphasis: Highlight the candidate’s key accomplishments and results, emphasizing measurable impact or contributions (use quantifiable details if available).
        Clarity & Structure: Write clearly and coherently, ensuring the summary is well-structured and grabs the recruiter’s attention. Keep the wording ATS-friendly (plain text and relevant keywords).
        Avoid: Steer clear of generic clichés or buzzwords (e.g., “hard-working team player,” “go-getter,” etc.). Maintain a professional resume style in third person (no “I” or “my”).
        Now, produce the rewritten professional summary in an impactful and strong tone following these guidelines."""
    return Impactful_P_Strong_Tone

async def Leadership_Tone(Summary):
    Leadership_P_Tone = f"""You are a professional resume consultant. Analyze the following professional summary and rewrite it in a leadership-focused tone.
    **Original Professional Summary:** {Summary}

    Guidelines for the rewritten summary:
        Length: 60–80 words in one well-crafted paragraph.
        Content: Keep all factual details and key experiences from the original summary (do not add information that isn’t provided).
        Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.
        Emphasis: Highlight the candidate’s leadership achievements and responsibilities (e.g. leading teams, driving strategic initiatives, mentoring others, delivering results). Showcase strategic impact and management experience where applicable.
        Clarity & Appeal: Ensure the rewritten summary is clear and structured, making it easy for recruiters to identify the candidate’s value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.
        Avoid: Avoid generic or overused phrases (no clichés like “natural born leader” or “dynamic people person,” etc.). Write in third person or an implied first-person style typical of resumes (no “I” statements or possessive “my”).
        Now, provide the rewritten professional summary in a leadership-oriented tone, following these guidelines."""
    return Leadership_P_Tone


async def professional_responce(tone :str, summary:str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(summary)
          agent = create_agent(model, 
                  response_format=ToolStrategy(professional_summary), 
                  system_prompt =prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(summary)
        agent = create_agent(model, 
                  response_format=ToolStrategy(professional_summary), 
                  system_prompt =prompt)
    else:
        prompt = await Leadership_Tone(summary)
        agent = create_agent(model, 
                  response_format=ToolStrategy(professional_summary), 
                  system_prompt =prompt)
    
    context_message = """Please rewrite the professional summary according to the guidelines provided."""
    
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
    
#     tone = "Leadership Tone"
#     output = asyncio.run( professional_responce(tone,summary))
#     print(output.summary)