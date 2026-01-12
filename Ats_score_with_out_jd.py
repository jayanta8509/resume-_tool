import os
import asyncio
import json
from re import S
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class ats_score_with_out_jd(BaseModel):
    ats_score: int
    improvement_guide: str


# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=500,
    request_timeout=20,
)

SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) specialist.

**TASK:**
Analyze RESUME for general market readiness and ATS compatibility. Infer target role from resume.
Provide ATS score (0-100) and clear, user-friendly improvement guide.
If previous score provided, mention progress briefly in the guide.

**SCORING (0-100):**
- Role Alignment (30pts): Clear career path, consistent role focus
- Impact & Results (25pts): Quantified achievements, metrics
- Clarity & Structure (20pts): Organization, formatting
- Skills Quality (15pts): Relevant skills proven by experience
- Language (5pts): Grammar, active voice
- ATS Compatibility (5pts): Standard sections, simple format

**RULES:**
- Be strict. Most resumes score 60-75. Cap excellent at 85.
- Infer target role from job titles, skills, experience
- Don't assume missing info.
- Penalize vague claims.
- Write improvement_guide in simple, friendly language.

**OUTPUT FORMAT:**
{
  "ats_score": [X],
  "improvement_guide": "Your resume scores [X]/100 for [inferred role] positions. [IF PREVIOUS: Previously scored [Y]/100 - improved by [Z] points!]

**Target Role:** [Most likely role based on resume]
**Experience Level:** [Fresher/Mid-level/Senior]

**Top 3 Improvements:**
1. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]
2. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]
3. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]

**What's Working:**
[list 2-3 strong points]

**What's Missing:**
[list 2-3 gaps or weaknesses]

**Market Readiness:**
• Startups: [Ready/Needs work]
• MNCs/Enterprise: [Ready/Needs work]
• Product Companies: [Ready/Needs work]

**Best Fix for Quick Score Boost:**
[One specific, actionable change that will add 5-10 points]

After these fixes, your score could reach: [Y]/100"
}"""


async def ATs_score_with_out_jd(resume: str, previous_ats_score: int | None = None):
    """
    Analyze resume for general ATS compatibility and provide score.

    Args:
        resume: User's resume text
        previous_ats_score: Optional previous ATS score for comparison (0-100)

    Returns:
        Dictionary with ats_score and improvement_guide
    """
    agent = create_agent(model,
            response_format=ToolStrategy(ats_score_with_out_jd),
            system_prompt=SYSTEM_PROMPT)

    # Build context message with previous score if provided
    previous_score_section = ""
    if previous_ats_score is not None:
        previous_score_section = f"""
Previous ATS Score: {previous_ats_score}/100

Please compare the current analysis with this previous score and highlight:
1. What improvements have been made (if any)
2. Any areas that may have declined
3. Specific changes that impacted the score
"""

    context_message = f"""User's Resume:
{resume}
{previous_score_section}
Please analyze and return ats_score and improvement_guide according to resume data."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]

    return ans

# if __name__ == "__main__":
#     resume = """Jayanta Roy
# AI/ML Engineer — LLM & Multi-Agent Systems Specialist
# 8017021283 | jayantameslova@gmail.com | LinkedIn | GitHub | Kolkata, India
# PROFESSIONAL SUMMARY
# Results-driven AI/ML Engineer with 2+ years building production AI systems serving 10K+ users. Specialized in LLM applications, multi-agent
# architectures, and end-to-end ML pipelines using GPT-4, LangChain, and modern frameworks. Proven expertise delivering 95%+ model accuracy
# and 40% efficiency improvements through scalable solutions with FastAPI, AWS, and GPU infrastructure.
# EXPERIENCE
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
# KEY ACHIEVEMENTS
# Kolkata, India
# Graduated 2023
# Kolkata, India
# Graduated 2020
# • Deployed 8+ production AI systems serving 10K+ users with 95%+ model accuracy and ATS compliance
# • Expertise in multi-agent AI architectures, LLM fine-tuning, and scalable ML pipeline development
# • Proficient in full-stack AI deployment: model training, API development, cloud infrastructure, and CI/CD automation"""


#     output = asyncio.run(ATs_score_with_out_jd(resume, 80))
#     print(output.ats_score)
#     print(output.improvement_guide)
