from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="Aura Score API")

# Define answer options for each question
class Q1Answer(str, Enum):
    CALM = "Calm"
    BIT_NERVOUS = "Bit nervous"
    VERY_ANXIOUS = "Very anxious"

class Q2Answer(str, Enum):
    LEAD = "Lead"
    SPEAK_WHEN_PROMPTED = "Speak when prompted"
    STAY_QUIET = "Stay quiet"

class Q3Answer(str, Enum):
    EXPLAIN_CLEARLY = "Explain clearly"
    HESITATE = "Hesitate"
    FREEZE = "Freeze"

class Q4Answer(str, Enum):
    ENERGIZED = "Energized"
    WISH_BETTER = "Wish better"
    DRAINED = "Drained"

class Q5Answer(str, Enum):
    NEVER = "Never"
    RARELY = "Rarely"
    SOMETIMES = "Sometimes"
    OFTEN = "Often"
    ALWAYS = "Always"

class Q6Answer(str, Enum):
    NEVER = "Never"
    RARELY = "Rarely"
    SOMETIMES = "Sometimes"
    OFTEN = "Often"
    ALWAYS = "Always"

class QuestionnaireInput(BaseModel):
    q1: Q1Answer = Field(..., description="How do you feel in social situations?")
    q2: Q2Answer = Field(..., description="In group discussions, do you...")
    q3: Q3Answer = Field(..., description="When explaining something complex...")
    q4: Q4Answer = Field(..., description="After social interactions, you feel...")
    q5: Q5Answer = Field(..., description="How often do people seek your company?")
    q6: Q6Answer = Field(..., description="How often do you inspire others?")

class SubScores(BaseModel):
    confidence: float = Field(..., description="Self-assurance and calm under pressure (0-100)")
    presence: float = Field(..., description="How captivating and energetic you are (0-100)")
    clarity: float = Field(..., description="How clearly you express ideas (0-100)")
    charisma: float = Field(..., description="Warmth + leadership aura (0-100)")
    eloquence: float = Field(..., description="Vocabulary richness and verbal smoothness (0-100)")

class AuraScoreResponse(BaseModel):
    aura_score: float = Field(..., description="Final aura score normalized to 0-100")
    sub_scores: SubScores = Field(..., description="Individual criteria scores")
    breakdown: dict = Field(..., description="Individual question scores and raw questionnaire score")

def get_q1_score(answer: Q1Answer) -> int:
    """Q1: Calm (10) / Bit nervous (6) / Very anxious (2)"""
    scores = {
        Q1Answer.CALM: 10,
        Q1Answer.BIT_NERVOUS: 6,
        Q1Answer.VERY_ANXIOUS: 2
    }
    return scores[answer]

def get_q2_score(answer: Q2Answer) -> int:
    """Q2: Lead (10) / Speak when prompted (6) / Stay quiet (2)"""
    scores = {
        Q2Answer.LEAD: 10,
        Q2Answer.SPEAK_WHEN_PROMPTED: 6,
        Q2Answer.STAY_QUIET: 2
    }
    return scores[answer]

def get_q3_score(answer: Q3Answer) -> int:
    """Q3: Explain clearly (10) / Hesitate (6) / Freeze (2)"""
    scores = {
        Q3Answer.EXPLAIN_CLEARLY: 10,
        Q3Answer.HESITATE: 6,
        Q3Answer.FREEZE: 2
    }
    return scores[answer]

def get_q4_score(answer: Q4Answer) -> int:
    """Q4: Energized (10) / Wish better (6) / Drained (2)"""
    scores = {
        Q4Answer.ENERGIZED: 10,
        Q4Answer.WISH_BETTER: 6,
        Q4Answer.DRAINED: 2
    }
    return scores[answer]

def get_q5_score(answer: Q5Answer) -> int:
    """Q5: Never (0), Rarely (2), Sometimes (5), Often (8), Always (10)"""
    scores = {
        Q5Answer.NEVER: 0,
        Q5Answer.RARELY: 2,
        Q5Answer.SOMETIMES: 5,
        Q5Answer.OFTEN: 8,
        Q5Answer.ALWAYS: 10
    }
    return scores[answer]

def get_q6_score(answer: Q6Answer) -> int:
    """Q6: Never (0), Rarely (2), Sometimes (5), Often (8), Always (10)"""
    scores = {
        Q6Answer.NEVER: 0,
        Q6Answer.RARELY: 2,
        Q6Answer.SOMETIMES: 5,
        Q6Answer.OFTEN: 8,
        Q6Answer.ALWAYS: 10
    }
    return scores[answer]

def calculate_scores(input_data: QuestionnaireInput) -> AuraScoreResponse:
    """Calculate aura score based on questionnaire answers"""
    
    # Get individual question scores
    q1_score = get_q1_score(input_data.q1)
    q2_score = get_q2_score(input_data.q2)
    q3_score = get_q3_score(input_data.q3)
    q4_score = get_q4_score(input_data.q4)
    q5_score = get_q5_score(input_data.q5)
    q6_score = get_q6_score(input_data.q6)
    
    # Calculate questionnaire score (0-60 range)
    questionnaire_score = q1_score + q2_score + q3_score + q4_score + q5_score + q6_score
    
    # Normalize questionnaire score to 0-100 range
    # AuraScore = QuestionnaireScore normalized to 0-100
    aura_score = (questionnaire_score / 60) * 100
    aura_score = round(max(0, min(100, aura_score)), 2)  # Clamp and round
    
    # Calculate sub-scores (0-100 range) - these are separate metrics
    # Confidence: avg(Q1, Q4) × 10
    confidence = ((q1_score + q4_score) / 2) * 10
    
    # Presence: avg(Q2, Q5) × 10
    presence = ((q2_score + q5_score) / 2) * 10
    
    # Clarity: Q3 × 10
    clarity = q3_score * 10
    
    # Charisma: avg(Q2, Q6) × 10
    charisma = ((q2_score + q6_score) / 2) * 10
    
    # Eloquence: Q3 × 10
    eloquence = q3_score * 10
    
    return AuraScoreResponse(
        aura_score=aura_score,
        sub_scores=SubScores(
            confidence=round(confidence, 2),
            presence=round(presence, 2),
            clarity=round(clarity, 2),
            charisma=round(charisma, 2),
            eloquence=round(eloquence, 2)
        ),
        breakdown={
            "questionnaire_score_raw": questionnaire_score,
            "questionnaire_score_max": 60,
            "q1_score": q1_score,
            "q2_score": q2_score,
            "q3_score": q3_score,
            "q4_score": q4_score,
            "q5_score": q5_score,
            "q6_score": q6_score
        }
    )

@app.get("/")
async def root():
    return {
        "message": "Aura Score API - Questionnaire Version",
        "endpoints": {
            "POST /calculate-aura": "Calculate aura score from questionnaire answers",
            "GET /questions": "Get all questions and answer options"
        }
    }

@app.get("/questions")
async def get_questions():
    """Return all questions and their possible answers"""
    return {
        "questions": [
            {
                "id": "q1",
                "question": "How do you feel in social situations?",
                "options": [
                    {"value": "Calm", "score": 10},
                    {"value": "Bit nervous", "score": 6},
                    {"value": "Very anxious", "score": 2}
                ]
            },
            {
                "id": "q2",
                "question": "In group discussions, do you...",
                "options": [
                    {"value": "Lead", "score": 10},
                    {"value": "Speak when prompted", "score": 6},
                    {"value": "Stay quiet", "score": 2}
                ]
            },
            {
                "id": "q3",
                "question": "When explaining something complex...",
                "options": [
                    {"value": "Explain clearly", "score": 10},
                    {"value": "Hesitate", "score": 6},
                    {"value": "Freeze", "score": 2}
                ]
            },
            {
                "id": "q4",
                "question": "After social interactions, you feel...",
                "options": [
                    {"value": "Energized", "score": 10},
                    {"value": "Wish better", "score": 6},
                    {"value": "Drained", "score": 2}
                ]
            },
            {
                "id": "q5",
                "question": "How often do people seek your company?",
                "options": [
                    {"value": "Never", "score": 0},
                    {"value": "Rarely", "score": 2},
                    {"value": "Sometimes", "score": 5},
                    {"value": "Often", "score": 8},
                    {"value": "Always", "score": 10}
                ]
            },
            {
                "id": "q6",
                "question": "How often do you inspire others?",
                "options": [
                    {"value": "Never", "score": 0},
                    {"value": "Rarely", "score": 2},
                    {"value": "Sometimes", "score": 5},
                    {"value": "Often", "score": 8},
                    {"value": "Always", "score": 10}
                ]
            }
        ],
        "scoring": {
            "aura_score": "Main score based on questionnaire (0-100)",
            "sub_scores": {
                "confidence": "Self-assurance and calm under pressure (Q1, Q4)",
                "presence": "How captivating and energetic you are (Q2, Q5)",
                "clarity": "How clearly you express ideas (Q3)",
                "charisma": "Warmth + leadership aura (Q2, Q6)",
                "eloquence": "Vocabulary richness and verbal smoothness (Q3)"
            }
        }
    }

@app.post("/calculate-aura", response_model=AuraScoreResponse)
async def calculate_aura_score(input_data: QuestionnaireInput):
    """
    Calculate aura score based on questionnaire answers
    
    The aura_score is the normalized questionnaire score (0-100).
    Sub-scores are calculated separately for additional insights.
    """
    try:
        return calculate_scores(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating aura score: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}