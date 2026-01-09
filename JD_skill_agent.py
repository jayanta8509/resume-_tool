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


class MissingSkill(BaseModel):
    missing_skill: str

class JD_Base_Skill(BaseModel):
    missing_skills: list[MissingSkill]


# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-5.2",  # Much faster than gpt-5-mini
    # temperature=0.1,        # very low for deterministic output
    max_tokens=800,    # Set explicit limit to prevent over-generation
    request_timeout=25,           # Add timeout to prevent hanging
    # streaming=False,      # Disable streaming for batch processing
    # max_retries=2
)

SYSTEM_PROMPT = """You are an expert skills analyst. You will receive:
1. User's skills from their resume
2. A job description (JD)

Your ONLY task is to:
Compare the user's skills against the JD requirements and identify which skills from the JD are MISSING in the user's resume.

**CRITICAL**:
- Return ONLY the missing skills (skills required by JD that the user does NOT have)
- Do NOT return skills the user already has
- Focus on both technical hard skills and soft skills mentioned in the JD

        **Data Processing Instructions:**
        - Analyze all provided sources to get a complete picture of technical and professional skills
        - Cross-reference skills across sources to ensure accuracy and completeness
        - Merge related skills from different sources (e.g., GitHub repository technologies with resume skills)
        - Extract skills demonstrated through projects, work experience, and stated competencies
        - Validate skills authenticity by checking consistency across sources
        - Remove duplicates and consolidate similar skills under standard terminology
        - **ATS Optimization**: Strategically prioritize and categorize skills based on job description requirements

        **Skills Extraction and Categorization Requirements:**
        Analyze and categorize skills into appropriate groups with strategic focus on JD alignment:

        **ATS-Optimized Skill Categories:**
        
        1. **Programming Languages (JD Priority)**: 
           - **Primary Focus**: Prioritize programming languages from JD hard_skills list
           - Languages actually used in projects or work that match JD requirements
           - Include evidence from GitHub repositories and project descriptions
           - Use industry-standard names and proper capitalization matching JD format
           - **JD Alignment**: List JD-required languages first, then additional languages

        2. **Frameworks & Libraries (JD Priority)**: 
           - **Primary Focus**: Emphasize frameworks from JD hard_skills and tools_and_technologies
           - Web frameworks, mobile frameworks, data science libraries that match JD needs
           - Cross-reference with GitHub repository usage and prioritize JD matches
           - **JD Alignment**: Highlight frameworks mentioned in job requirements first

        3. **Tools & Technologies (JD Alignment)**: 
           - **Critical Focus**: Prioritize tools from JD tools_and_technologies list
           - Development tools, cloud platforms, databases, DevOps tools matching JD requirements
           - Include design tools, project management tools that align with job needs
           - **ATS Strategy**: Ensure JD-mentioned tools appear prominently when candidate has them

        4. **Cloud & DevOps (JD Relevant)**: 
           - Cloud platforms (AWS, Azure, Google Cloud) matching JD requirements
           - DevOps tools (Docker, Kubernetes, Jenkins) from JD specifications
           - Infrastructure and deployment technologies aligned with job needs
           - **JD Priority**: Emphasize cloud/DevOps skills mentioned in job requirements

        5. **Databases & Data Management (JD Focused)**: 
           - Relational and NoSQL databases that match JD technical requirements
           - Data warehousing and big data tools aligned with job specifications
           - **JD Alignment**: Prioritize database technologies mentioned in job requirements

        6. **Soft Skills & Leadership (JD Matching)**: 
           - **Strategic Focus**: Prioritize soft skills from JD soft_skills list
           - Communication, teamwork, leadership skills that match job requirements
           - Project management and organizational abilities aligned with JD needs
           - Problem-solving and analytical thinking as specified in job description
           - **ATS Optimization**: Include JD soft skills when genuinely demonstrated

        7. **Domain Expertise (JD Relevant)**: 
           - Industry-specific knowledge that aligns with job domain
           - Business analysis, data analysis, methodologies matching JD requirements
           - Specialized expertise relevant to target role and industry
           - **JD Alignment**: Focus on domain skills that support job requirements

        **Dynamic Categories Based on JD Requirements:**
        Create additional categories if JD emphasizes specific areas:
        - **Mobile Development** (if JD requires mobile skills)
        - **Data Science & Analytics** (if JD focuses on data roles)
        - **Security & Cybersecurity** (if JD emphasizes security)
        - **AI & Machine Learning** (if JD requires AI/ML expertise)
        - **Other JD-Specific Categories** based on unique job requirements

        **ATS Optimization Guidelines:**
        - **JD Skills Priority**: Prioritize skills that directly match JD hard_skills, soft_skills, and tools_and_technologies
        - **Keyword Matching**: Use exact terminology from JD when candidate possesses those skills
        - **Skills Ordering**: Within each category, list JD-matching skills first
        - **Category Relevance**: Emphasize categories most relevant to job requirements
        - **Competitive Advantage**: Highlight skills that exceed basic JD requirements
        - **Professional Validation**: Focus on skills with concrete evidence and JD alignment

        **Quality Standards:**
        - Use industry-standard terminology matching JD format and capitalization
        - Group related skills logically while prioritizing JD-relevant skills
        - Include only skills with clear evidence and emphasize JD matches
        - Prioritize skills by JD relevance first, then by demonstrated proficiency
        - Ensure each skill appears only once across all categories
        - Order skills within categories by JD importance and proficiency level
        - Include 3-8 skills per category, prioritizing JD-matching skills
        - Focus on skills that add maximum value for the target role

        **Validation and Prioritization Guidelines:**
        - **JD Skill Validation**: Cross-validate candidate skills against JD requirements
        - **Technical Verification**: Confirm technical skills with GitHub repository evidence
        - **Soft Skills Evidence**: Ensure soft skills align with demonstrated experience and JD needs
        - **Relevance Filtering**: Remove skills not relevant to job requirements
        - **Strategic Presentation**: Present skills in order of importance to target role

        """


async def skill_responce(Skill: str, JD: str):
    agent = create_agent(model,
            response_format=ToolStrategy(JD_Base_Skill),
            system_prompt=SYSTEM_PROMPT)

    context_message = f"""User's Resume Skills:
{Skill}

Job Description:
{JD}

Please analyze and return ONLY the skills that are required in the JD but MISSING from the user's resume."""
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     skill = """Python & Golang Transformers & LLMs  
# LangGraph, VectorDB Generative AI, Agents & RAGs 
# AWS, Spark, ML Ops Kafka, Redis n8n, Gumloop 
# Postgres, SQL, Elastic Search MixPanel, Airtable, Zapier"""

#     jd = """Senior AI/ML Engineer Required Skills:

# Hard Skills:
# - Python, Java
# - TensorFlow, PyTorch, Keras
# - AWS SageMaker, Azure ML
# - Kubernetes, Docker
# - MongoDB, Cassandra
# - React, Node.js
# - GraphQL
# - Apache Kafka
# - CI/CD with Jenkins
# - TypeScript

# Soft Skills:
# - Team leadership
# - Agile/Scrum methodology
# - Technical documentation"""

#     output = asyncio.run(skill_responce(skill, jd))
#     print("=" * 50)
#     print("MISSING SKILLS (Required by JD but not in resume):")
#     print("=" * 50)
#     for ms in output.missing_skills:
#         print(f"- {ms.missing_skill}")