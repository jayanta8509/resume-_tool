import logging
import os
import time
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from Professional_s_agent import professional_responce
from Experience_d_agent import Experience_result
from JD_Professional_s_agent import professional_responce as jd_professional_responce
from JD_Experience_d_agent import Experience_result as jd_Experience_result
from JD_skill_agent import skill_responce
from Ats_score_with_jd import ATs_score
from Ats_score_with_out_jd import ATs_score_with_out_jd
from Generate_Experience_d_agent import Generate_Experience_result
from Generate_Professional_s_agent import Generate_professional_responce

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SECURITY_ID = os.getenv("Security_ID")
TONE_MAPPING = {
    "Professional": "Professional & Clean",
    "Impactful": "Impactful & Strong",
    "Leadership": "Leadership Tone",
}

# Initialize FastAPI app
app = FastAPI(
    title="Resume Builder API",
    description="AI-powered Resume Enhancement",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Tone(str, Enum):
    PROFESSIONAL = "Professional"
    IMPACTFUL = "Impactful"
    LEADERSHIP = "Leadership"


class JDAgentRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Security authentication ID")
    tone: Tone = Field(default=Tone.PROFESSIONAL, description="Tone for summary generation")
    original_text: str = Field(..., min_length=1, description="Original text to enhance")
    JD: str = Field(..., min_length=1, description="Job Description")

    @field_validator("security_id", "original_text", "JD")
    @classmethod
    def validate_non_empty_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()

class AgentRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Security authentication ID")
    tone: Tone = Field(default=Tone.PROFESSIONAL, description="Tone for summary generation")
    original_text: str = Field(..., min_length=1, description="Original text to enhance")

    @field_validator("security_id")
    @classmethod
    def validate_security_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Security ID cannot be empty")
        return v.strip()

    @field_validator("original_text")
    @classmethod
    def validate_original_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Original text cannot be empty")
        return v.strip()
    
class SkillAgentRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Security authentication ID")
    Skill: str = Field(..., min_length=1, description="Original text to enhance")
    JD: str = Field(..., min_length=1, description="Job Description")

    @field_validator("security_id", "Skill", "JD")
    @classmethod
    def validate_non_empty_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()

class AtsJDAgentRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Security authentication ID")
    resume_data: str = Field(..., min_length=1, description="Resume data")
    Ats_score: Optional[int] = Field(None, description="privous ATS score")
    JD: str = Field(..., min_length=1, description="Job Description")

    @field_validator("security_id", "resume_data", "JD")
    @classmethod
    def validate_non_empty_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()

class AtsAgentRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Security authentication ID")
    resume_data: str = Field(..., min_length=1, description="Resume data")
    Ats_score: Optional[int] = Field(None, description="privous ATS score")

    @field_validator("security_id", "resume_data")
    @classmethod
    def validate_non_empty_fields(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("This field cannot be empty")
        return v.strip()


@app.post(
    "/agent/professional/summary",
    status_code=status.HTTP_200_OK,
    summary="Generate professional summary",
    description="Enhances resume text using AI-powered professional tone",
    tags=["Resume API With Improvement"]

)
async def generate_summary(request: AgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await professional_responce(tone, request.original_text)

        return {
            "summary": result.summary,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.post(
    "/agent/Generate/professional/summary",
    status_code=status.HTTP_200_OK,
    summary="Generate professional summary with all Resume data",
    description="Enhances resume text using AI-powered professional tone",
    tags=["Resume API With Generate"]

)
async def generate_p_summary(request: AgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await Generate_professional_responce(tone, request.original_text)

        return {
            "summary": result.summary,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.post(
    "/agent/JD/professional/summary",
    status_code=status.HTTP_200_OK,
    summary="Generate professional summary",
    description="Enhances resume text using AI-powered professional tone",
    tags=["Resume API With Job Description"]

)
async def generate_summary_with_JD(request: JDAgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await jd_professional_responce(tone, request.original_text ,request.JD)

        return {
            "summary": result.summary,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@app.post(
    "/agent/Experience/Description",
    status_code=status.HTTP_200_OK,
    summary="Generate Experience Description",
    description="Enhances resume text using AI-powered professional tone",
    tags=["Resume API With Improvement"]
)
async def generate_Experience_Description(request: AgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await Experience_result(tone, request.original_text)

        return {
            "summary": result.description,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@app.post(
    "/agent/Generate/Experience/Description",
    status_code=status.HTTP_200_OK,
    summary="Generate Experience Description with All Resume Data",
    description="Enhances resume text using AI-powered professional tone",
    tags=["Resume API With Generate"]
)
async def generate_P_Experience_Description(request: AgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await Generate_Experience_result(tone, request.original_text)

        return {
            "summary": result.description,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.post(
    "/agent/JD/Experience/Description",
    status_code=status.HTTP_200_OK,
    summary="Generate Experience Description with JD",
    description="Enhances resume experience using AI-powered professional tone with job description",
    tags=["Resume API With Job Description"]
)
async def generate_Experience_Description_with_JD(request: JDAgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        tone = TONE_MAPPING.get(request.tone.value, TONE_MAPPING[Tone.PROFESSIONAL])
        result = await jd_Experience_result(tone, request.original_text, request.JD)

        return {
            "summary": result.description,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )



@app.post(
    "/agent/JD/Missing/Skill",
    status_code=status.HTTP_200_OK,
    summary="Find missing skills from job description",
    description="Analyzes user skills against job description and returns missing required skills",
    tags=["Resume API Missing Skill using Job Description"]
)
async def missing_skill(request: SkillAgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )

        result = await skill_responce(request.Skill, request.JD)

        return {
            "missing_skills": [skill.missing_skill for skill in result.missing_skills],
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding missing skills: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )



@app.post(
    "/agent/ATS/Score/With/JD",
    status_code=status.HTTP_200_OK,
    summary="ATS Score with JD",
    description="Analyzes user resume against job description and returns Ats Score",
    tags=["ATS Score With JD"]
)
async def ATS_score_with_jd(request: AtsJDAgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )
        if request.Ats_score is None:
            result = await ATs_score(request.resume_data, request.JD)
        else :
            result = await ATs_score(request.resume_data, request.JD, request.Ats_score)

        return {
            "ATS_Score": result.ats_score,
            "Improvment_Guide" : result.improvement_guide,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding missing skills: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@app.post(
    "/agent/ATS/Score",
    status_code=status.HTTP_200_OK,
    summary="ATS Score with Out JD",
    description="Analyzes user resume returns Ats Score",
    tags=["ATS Score With Out JD"]
)
async def ATS_score_with_jd(request: AtsAgentRequest) -> dict:
    try:
        if request.security_id != SECURITY_ID:
            logger.warning(f"Failed authentication attempt with security_id: {request.security_id[:4]}***")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid security credentials"
            )
        if request.Ats_score is None:
            result = await ATs_score_with_out_jd(request.resume_data)
        else :
            result = await ATs_score_with_out_jd(request.resume_data, request.Ats_score)

        return {
            "ATS_Score": result.ats_score,
            "Improvment_Guide" : result.improvement_guide,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding missing skills: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@app.get("/", tags=["General"])
async def root() -> dict:
    return {
        "name": "Resume Builder API",
        "version": "2.0.0",
        "endpoints": {
            "professional_summary": "POST /agent/professional/summary",
            "jd_professional_summary": "POST /agent/JD/professional/summary",
            "experience_description": "POST /agent/Experience/Description",
            "jd_experience_description": "POST /agent/JD/Experience/Description",
            "missing_skill": "POST /agent/JD/Missing/Skill",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["General"])
async def health_check() -> dict:
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )