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


async def Professional_Clean_Tone(resume_data):
    Professional_P_Clean_Tone = f"""You are a professional resume consultant. Generate compelling work experience descriptions based on the candidate's resume data.

    **Resume Data:**
    {resume_data}

    Guidelines for generating the experience description:
        Format: Present as 4-6 concise bullet points (each bullet: 15-25 words).
        Content: Synthesize key information from the PROFESSIONAL SUMMARY and TECHNICAL SKILLS sections to create realistic, relevant experience descriptions. Highlight technologies, tools, achievements, and responsibilities mentioned in the resume.
        Tone/Style: Use a formal, clear, and polished writing style that feels professional and straightforward.
        Clarity: Structure each bullet to highlight specific skills, projects, or contributions. Make them concrete and meaningful.
        Action Verbs: Start each bullet with a strong action verb (e.g., Developed, Implemented, Managed, Optimized, Led, Designed, Created).
        ATS/Recruiter Appeal: Ensure the language is ATS-friendly and appealing to hiring managers (use relevant keywords from the skills section; avoid any special/unusual formatting).
        Avoid: No generic filler phrases or clichés (e.g., "hard-working individual," "results-driven professional," etc.). Write in third person or neutral perspective (avoid using "I" or "my").
        Quantifiable Impact: Include reasonable metrics based on the experience level mentioned (years, scope). Frame outcomes in specific, achievable terms.
        Now, generate the experience description in a professional and clean tone based on these guidelines."""

    return Professional_P_Clean_Tone

async def Impactful_Strong_Tone(resume_data):
    Impactful_P_Strong_Tone = f"""You are a professional resume consultant. Generate compelling work experience descriptions based on the candidate's resume data.
    **Resume Data:**
    {resume_data}

    Guidelines for generating the experience description:
        Format: Present as 4-6 powerful bullet points (each bullet: 15-25 words).
        Content: Synthesize key information from the PROFESSIONAL SUMMARY and TECHNICAL SKILLS sections to create impactful, relevant experience descriptions. Emphasize technologies, tools, achievements, and scope mentioned in the resume.
        Tone/Style: Use confident, dynamic language with strong action verbs to convey an impactful and assertive tone. Each bullet should feel powerful and decisive.
        Emphasis: Highlight the candidate's key accomplishments and results, emphasizing measurable impact or contributions. Frame outcomes to show significant value delivered.
        Action Verbs: Start each bullet with high-impact action verbs (e.g., Spearheaded, Engineered, Transformed, Accelerated, Revolutionized, Maximized, Delivered, Architectured).
        Quantifiable Impact: Include strong metrics based on experience level (%, improvements, scale). Make each bullet feel results-oriented and high-impact.
        Clarity & Structure: Write clearly and coherently, ensuring the description is well-structured and grabs the recruiter's attention. Keep the wording ATS-friendly (plain text and relevant keywords).
        Avoid: Steer clear of generic clichés or buzzwords (e.g., "hard-working team player," "go-getter," etc.). Maintain a professional resume style in third person (no "I" or "my").
        Now, generate the experience description in an impactful and strong tone following these guidelines."""
    return Impactful_P_Strong_Tone

async def Leadership_Tone(resume_data):
    Leadership_P_Tone = f"""You are a professional resume consultant. Generate compelling work experience descriptions based on the candidate's resume data.
    **Resume Data:**
    {resume_data}

    Guidelines for generating the experience description:
        Format: Present as 4-6 authoritative bullet points (each bullet: 15-25 words).
        Content: Synthesize key information from the PROFESSIONAL SUMMARY and TECHNICAL SKILLS sections to create leadership-oriented experience descriptions. Emphasize technologies, strategic initiatives, team leadership, and high-level responsibilities.
        Tone/Style: Use a confident, authoritative tone that emphasizes leadership qualities and vision. The style should reflect a leadership presence — professional, decisive, and inspiring.
        Emphasis: Highlight leadership achievements and responsibilities (e.g., leading teams, driving strategic initiatives, mentoring others, delivering results, managing stakeholders, overseeing projects, architectural decisions).
        Action Verbs: Start each bullet with leadership-oriented action verbs (e.g., Led, Directed, Orchestrated, Mentored, Guided, Championed, Established, Cultivated, Spearheaded).
        Strategic Focus: Frame accomplishments to show strategic thinking, decision-making authority, and ability to drive organizational impact. Emphasize scope and influence.
        Clarity & Appeal: Ensure the description is clear and structured, making it easy for recruiters to identify the candidate's leadership value. Use language that is ATS-friendly and includes strong keywords relevant to leadership roles.
        Avoid: Avoid generic or overused phrases (no clichés like "natural born leader" or "dynamic people person," etc.). Write in third person (no "I" statements or possessive "my").
        Now, generate the experience description in a leadership-oriented tone, following these guidelines."""
    return Leadership_P_Tone


async def Generate_Experience_result(tone :str, resume_data:str):
    if tone == "Professional & Clean":
          prompt = await Professional_Clean_Tone(resume_data)
          agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)

    elif tone == "Impactful & Strong":
        prompt = await Impactful_Strong_Tone(resume_data)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)
    else:
        prompt = await Leadership_Tone(resume_data)
        agent = create_agent(model,
                  response_format=ToolStrategy(Experience_description),
                  system_prompt =prompt)

    context_message = """Please generate the experience description according to the guidelines provided."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     resume_data = """PROFESSIONAL SUMMARY
# AI/ML Engineer with 2+ years of experience building production-grade LLM and computer vision systems across e-commerce, finance, and recruitment. Delivered a virtual try-on platform with 95%+ accuracy, a three-agent screening solution that cut hiring workflows by 40%, and a fine-tuned Mistral-7B model improving FAQ accuracy by 30%. Proficient in Python, FastAPI, LangChain/RAG, vector databases, and AWS/RunPod deployment.

# TECHNICAL SKILLS
# Languages: Python, C/C++, SQL
# AI/ML Frameworks: LangChain, RAG, LangGraph, CrewAI, AutoGen, Swarm, Pydantic AI, MLOps, Scikit-Learn, TensorFlow, Keras, PyTorch
# LLM & Models: OpenAI GPT-4/4o, Gemini, DeepSeek, Anthropic Claude, Grok, Mistral, BERT, T5, Stable Diffusion, Hugging Face
# Databases: PostgreSQL, MySQL, Redis, Pinecone, FAISS, Chroma, Qdrant (Vector DBs)
# DevOps & Cloud: Docker, Git, AWS (SageMaker, EC2, Lambda, S3, LightSail), RunPod GPU Servers, CI/CD Pipelines
# APIs & Web: FastAPI, Flask, Quart, Django REST, RESTful APIs, OpenCV, Beautiful Soup
# ML Techniques: NLP, Computer Vision, Supervised/Unsupervised Learning, Feature Engineering, XGBoost, SGD, PEFT, LoRA, Model Fine-tuning
# """

#     tone = "Impactful & Strong"
#     output = asyncio.run(Experience_result(tone, resume_data))
#     r = output.description
#     print(len(r))
#     print(r)