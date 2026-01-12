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


class ats_score_with_jd(BaseModel):
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
Analyze RESUME against JOB DESCRIPTION. Provide ATS score (0-100) and clear, user-friendly improvement guide.
If previous score provided, mention progress briefly in the guide.

**SCORING (0-100):**
- Role Alignment (30pts): Keywords match, relevant experience
- Impact & Results (25pts): Quantified achievements, metrics
- Clarity & Structure (20pts): Organization, formatting
- Skills Quality (15pts): Required skills present
- Language (5pts): Grammar, active voice
- ATS Compatibility (5pts): Standard sections, simple format

**RULES:**
- Be strict. Most resumes score 60-75. Cap excellent at 85.
- Don't assume missing info.
- Penalize vague claims.
- Write improvement_guide in simple, friendly language.

**OUTPUT FORMAT:**
{
  "ats_score": [X],
  "improvement_guide": "Your resume scores [X]/100. [IF PREVIOUS: Previously scored [Y]/100 - improved by [Z] points!]

**Top 3 Improvements:**
1. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]
2. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]
3. **[Issue]:** [Simple explanation] → **Do this:** [Specific action]

**What's Working:**
[list 2-3 things that match the JD well]

**What's Missing:**
[list 2-3 key requirements from JD not in resume]

**Best Fix for Quick Score Boost:**
[One specific, actionable change that will add 5-10 points]

After these fixes, your score could reach: [Y]/100"
}"""


async def ATs_score(resume: str, JD: str, previous_ats_score: int | None = None):
    """
    Analyze resume against job description and provide ATS score.

    Args:
        resume: User's resume text
        JD: Job description text
        previous_ats_score: Optional previous ATS score for comparison (0-100)

    Returns:
        Dictionary with ats_score, previous_score, score_change, and improvement_guide
    """
    agent = create_agent(model,
            response_format=ToolStrategy(ats_score_with_jd),
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

Job Description:
{JD}
{previous_score_section}
Please analyze and return ats_score, previous_score (if provided), score_change, and improvement_guide according to JD and resume data."""

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
# • Proficient in full-stack AI deployment: model training, API development, cloud infrastructure, and CI/CD automation """

#     jd = """Lead the design, development, and deployment of production-grade AI/ML solutions that drive measurable business impact across Syngenta's operations. You will own end-to-end ML pipelines, from problem definition through production deployment, while collaborating with cross-functional teams to transform agricultural and business challenges into scalable AI applications.

# Accountabilities

# Design, develop, and deploy production ready AI/ML models and applications that solve critical business problems
# Own the complete ML lifecycle: data pipeline design, feature engineering, model training, evaluation, deployment, and monitoring
# Collaborate with cross-functional teams to understand business requirements and translate them into technical solutions
# Implement MLOps best practices including model versioning, CI/CD pipelines, and automated retraining
# Translate complex business requirements into technical AI/ML solutions with clear success metrics
# Conduct code reviews and establish engineering best practices for AI projects
# Evaluate and integrate emerging AI technologies (LLMs, GenAI, RAG systems) into Syngenta's ecosystem
# Lead POCs and MVPs using design thinking methodologies to validate solution feasibility
# Optimize model performance, latency, and cost-efficiency for production systems
# Contribute to Syngenta's AI platform capabilities and reusable component libraries

# Knowledge, Experience & Capabilities

# 3-5 years of hands-on experience in AI/ML engineering or related roles
# Proven track record of deploying models to production environments
# Proficiency in Python
# Good understanding of AWS cloud architecture including Sagemaker and Bedrock
# Ability to use identify and re-use GitHub projects to solve business problems
# Experience working with cross-functional teams and translating business needs into technical solutions
# Demonstrated ability to manage multiple projects and deliver results in agile environments

# Critical success factors & key challenges

# Strong algorithm design, analysis and reasoning skills
# Ability to deliver POCs, MVPs, Experiments, technology evaluations following design thinking practices
# Ability to orchestrate efforts needed to prioritize business initiatives across complex change agendas
# Excellent communication and stakeholder management skills to explain technical information to individuals who don't have the same technical background
# Problem solving and decision-making skills
# Risk assessment and mitigation for AI/ML projects
# Proactive in identifying opportunities for AI-driven improvements
# Teamwork, team management and leadership skills

# Innovations

# Employee may, as part of his/her role and maybe through multifunctional teams, participate in the creation and design of innovative solutions. In this context, Employee may contribute to inventions, designs, other work product, including know-how, copyrights, software, innovations, solutions, and other intellectual assets.

# Qualifications

# Degree in Computer Science, AI, ML, Data Science, Engineering, or related fields.

# Additional Information

# Note: Syngenta is an Equal Opportunity Employer and does not discriminate in recruitment, hiring, training, promotion or any other employment practices for reasons of race, color, religion, gender, national origin, age, sexual orientation, gender identity, marital or veteran status, disability, or any other legally protected status."""

#     output = asyncio.run(ATs_score(resume, jd))
#     print(output.ats_score)
#     print(output.improvement_guide)