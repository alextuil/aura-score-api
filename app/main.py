from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends, Security
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict
import math

from app.config import get_settings
import openai
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

app = FastAPI(title="Aura Score API")

# --- Security: API Key + Rate Limiting (IP+Key) ---
api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False)
def _get_allowed_api_keys() -> set[str]:
    settings = get_settings()
    csv = getattr(settings, "api_keys_csv", None) or ""
    keys = {k.strip() for k in csv.split(",") if k.strip()}
    return keys

def require_api_key(request: Request, provided: Optional[str] = Security(api_key_scheme)) -> None:
    if provided is None:
        provided = request.headers.get("x-api-key")
    allowed = _get_allowed_api_keys()
    if not allowed:
        # If no keys configured, deny by default in production; allow only for local dev
        if request.client and request.client.host in {"127.0.0.1", "::1"}:
            return
        raise HTTPException(status_code=401, detail="API key required")
    if not provided or provided not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})

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

class AudioSubScores(BaseModel):
    confidence: float = Field(..., description="Tone firmness, loudness, stability (0-8)")
    fluency: float = Field(..., description="Flow, hesitations, fillers per minute (0-8)")
    vocabulary_richness: float = Field(..., description="Lexical diversity (0-8)")
    clarity: float = Field(..., description="Enunciation/pronunciation proxy (0-8)")
    engagement: float = Field(..., description="Energy/enthusiasm proxy (0-8)")

class AuraScoreAudioResponse(BaseModel):
    aura_score: float = Field(..., description="Weighted aura score (0-100) with audio if provided")
    questionnaire_only_score: float = Field(..., description="Questionnaire-only score (0-100)")
    audio_score_raw: Optional[float] = Field(None, description="Audio sub-scores total (0-40) if audio provided")
    audio_sub_scores: Optional[AudioSubScores] = Field(None, description="Audio metrics (0-8 each)")
    sub_scores: SubScores = Field(..., description="Questionnaire sub-scores (0-100 each)")
    breakdown: Dict[str, float] = Field(..., description="Question-level scores and totals")

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

def _init_openai_client():
    try:
        settings = get_settings()
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required to use audio transcription")
        # Newer openai SDK uses client object; keep compatibility with common usage
        try:
            from openai import OpenAI  # type: ignore
            return OpenAI(api_key=api_key)  # prefer client-style
        except Exception:
            openai.api_key = api_key  # fallback to legacy global
            return None
    except HTTPException:
        raise
    except Exception as e:
        # If settings validation fails, surface as 400 with guidance
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required to use audio transcription")

def _transcribe_with_whisper(file: UploadFile) -> str:
    client = _init_openai_client()
    # Use new client if available
    try:
        if client is not None:
            # Read file bytes for the request
            content = file.file.read()
            from io import BytesIO
            bio = BytesIO(content)
            bio.name = file.filename or "audio.mp3"
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=bio
            )
            return transcript.text or ""
        # Legacy SDK path
        audio_bytes = file.file.read()
        from io import BytesIO
        stream = BytesIO(audio_bytes)
        stream.name = file.filename or "audio.mp3"
        result = openai.Audio.transcriptions.create(model="whisper-1", file=stream)  # type: ignore
        return getattr(result, "text", "") or ""
    finally:
        try:
            file.file.seek(0)
        except Exception:
            pass

def _compute_audio_metrics(transcript: str, audio_duration_seconds: float = 30.0) -> AudioSubScores:
    text = (transcript or "").lower()
    words = [w for w in ''.join([c if c.isalnum() or c.isspace() else ' ' for c in text]).split() if w]
    total_words = len(words)
    unique_words = len(set(words)) if words else 0

    # Heuristics (light proxies without raw audio analysis):
    filler_tokens = {"uh", "um", "erm", "hmm", "like", "youknow", "you", "know"}
    fillers = sum(1 for w in words if w in filler_tokens)
    fillers_per_min = (fillers / (audio_duration_seconds / 60.0)) if audio_duration_seconds > 0 else fillers

    # Fluency (0-8) – fewer fillers is better
    # Map fillers/min from 0..20+ to 8..0 with floor
    fluency = max(0.0, 8.0 - min(20.0, fillers_per_min) * (8.0 / 20.0))

    # Vocabulary richness (0-8) – unique/total ratio
    diversity_ratio = (unique_words / total_words) if total_words > 0 else 0.0
    vocabulary_richness = max(0.0, min(8.0, diversity_ratio * 8.0))

    # Confidence (0-8) – proxy using steady speech rate (words/sec near 2.0 is ideal)
    wps = (total_words / audio_duration_seconds) if audio_duration_seconds > 0 else 0.0
    # Ideal range ~1.5–2.5 wps. Score drops outside.
    if wps <= 0.5:
        confidence = 1.0
    elif wps >= 4.0:
        confidence = 2.0
    else:
        # Peak near 2.0
        confidence = max(0.0, 8.0 - abs(wps - 2.0) * 4.0)

    # Clarity (0-8) – proxy using presence of clear sentence boundaries and longer words
    long_words = sum(1 for w in words if len(w) >= 7)
    long_word_ratio = (long_words / total_words) if total_words > 0 else 0.0
    clarity = max(0.0, min(8.0, 2.0 + long_word_ratio * 12.0))  # cap at 8

    # Engagement (0-8) – proxy using exclamation/varied punctuation cues in transcript
    exclamations = transcript.count("!") if transcript else 0
    questions = transcript.count("?") if transcript else 0
    engagement_raw = exclamations * 1.5 + questions * 1.0 + (1 if total_words > 0 and (total_words / audio_duration_seconds) > 1.2 else 0)
    engagement = max(0.0, min(8.0, 2.0 + engagement_raw))

    return AudioSubScores(
        confidence=round(confidence, 2),
        fluency=round(fluency, 2),
        vocabulary_richness=round(vocabulary_richness, 2),
        clarity=round(clarity, 2),
        engagement=round(engagement, 2)
    )

def _calculate_weighted_aura(questionnaire_only_score_0_100: float, audio_sub_scores: Optional[AudioSubScores]) -> (float, Optional[float]):
    if audio_sub_scores is None:
        return round(questionnaire_only_score_0_100, 2), None
    audio_total_0_40 = (
        audio_sub_scores.confidence
        + audio_sub_scores.fluency
        + audio_sub_scores.vocabulary_richness
        + audio_sub_scores.clarity
        + audio_sub_scores.engagement
    )
    audio_norm_0_100 = (audio_total_0_40 / 40.0) * 100.0
    weighted = 0.6 * questionnaire_only_score_0_100 + 0.4 * audio_norm_0_100
    return round(weighted, 2), round(audio_total_0_40, 2)

def _blend_sub_scores_with_audio(base_breakdown: Dict[str, float], audio: AudioSubScores) -> SubScores:
    """
    Blend questionnaire sub-scores with audio metrics using provided weights.
    Audio metrics are normalized from 0–8 to 0–10 before blending.
    All outputs are clamped to 0–100 and rounded to 2 decimals.
    """
    # Questionnaire per-question scores (0–10 each)
    q1 = float(base_breakdown.get("q1_score", 0))
    q2 = float(base_breakdown.get("q2_score", 0))
    q3 = float(base_breakdown.get("q3_score", 0))
    q4 = float(base_breakdown.get("q4_score", 0))
    q5 = float(base_breakdown.get("q5_score", 0))
    q6 = float(base_breakdown.get("q6_score", 0))

    # Normalize audio (0–8) → (0–10)
    def norm8_to10(value: float) -> float:
        return max(0.0, min(10.0, (value / 8.0) * 10.0))

    audio_conf = norm8_to10(audio.confidence)
    audio_eng = norm8_to10(audio.engagement)
    audio_cla = norm8_to10(audio.clarity)
    audio_vocab = norm8_to10(audio.vocabulary_richness)
    audio_flu = norm8_to10(audio.fluency)

    # Formulas (result then ×10 to map 0–10 → 0–100)
    confidence = (( (q1 + q4) / 2.0 ) * 0.6 + audio_conf * 0.4) * 10.0
    presence = (( (q2 + q5) / 2.0 ) * 0.5 + audio_eng * 0.5) * 10.0
    clarity = ( q3 * 0.4 + audio_cla * 0.6 ) * 10.0
    charisma = (( (q2 + q6) / 2.0 ) * 0.5 + audio_eng * 0.5) * 10.0
    eloquence = ( q3 * 0.3 + audio_vocab * 0.4 + audio_flu * 0.3 ) * 10.0

    def clamp100(x: float) -> float:
        return round(max(0.0, min(100.0, x)), 2)

    return SubScores(
        confidence=clamp100(confidence),
        presence=clamp100(presence),
        clarity=clamp100(clarity),
        charisma=clamp100(charisma),
        eloquence=clamp100(eloquence)
    )

@app.get("/")
async def root():
    return {
        "message": "Aura Score API - Questionnaire Version",
        "endpoints": {
            "POST /calculate-aura": "Calculate aura score from questionnaire answers",
            "POST /calculate-aura-audio": "Calculate aura score from questionnaire answers + optional audio (multipart)",
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
@limiter.limit("60/minute")
async def calculate_aura_score(request: Request, input_data: QuestionnaireInput, _: None = Depends(require_api_key)):
    """
    Calculate aura score based on questionnaire answers
    
    The aura_score is the normalized questionnaire score (0-100).
    Sub-scores are calculated separately for additional insights.
    """
    try:
        return calculate_scores(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating aura score: {str(e)}")

@app.post("/calculate-aura-audio", response_model=AuraScoreAudioResponse)
@limiter.limit("60/minute")
async def calculate_aura_with_audio(
    request: Request,
    q1: Q1Answer = Form(...),
    q2: Q2Answer = Form(...),
    q3: Q3Answer = Form(...),
    q4: Q4Answer = Form(...),
    q5: Q5Answer = Form(...),
    q6: Q6Answer = Form(...),
    audio: UploadFile = File(..., description="Required 30s MP3 introducing yourself"),
    _: None = Depends(require_api_key)
):
    """
    Calculate aura score with optional audio enhancement.

    - Questionnaire generates a 0–100 score (unchanged logic)
    - If audio provided (30s mp3), compute 5 audio metrics (0–8 each, total 0–40),
      normalize to 0–100, and combine: 0.6 × Questionnaire + 0.4 × Audio
    """
    try:
        input_model = QuestionnaireInput(q1=q1, q2=q2, q3=q3, q4=q4, q5=q5, q6=q6)
        base = calculate_scores(input_model)

        audio_sub: Optional[AudioSubScores] = None
        audio_total_0_40: Optional[float] = None
        if (audio.content_type or "").lower() not in {"audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mpeg-3"}:
            raise HTTPException(status_code=400, detail="Audio file must be an MP3")
        transcript = _transcribe_with_whisper(audio)
        audio_sub = _compute_audio_metrics(transcript, audio_duration_seconds=30.0)

        weighted_0_100, audio_total_0_40 = _calculate_weighted_aura(base.aura_score, audio_sub)

        # Blend sub-scores with audio (mandatory in this endpoint)
        final_sub_scores = _blend_sub_scores_with_audio(base.breakdown, audio_sub)

        return AuraScoreAudioResponse(
            aura_score=weighted_0_100,
            questionnaire_only_score=base.aura_score,
            audio_score_raw=audio_total_0_40,
            audio_sub_scores=audio_sub,
            sub_scores=final_sub_scores,
            breakdown=base.breakdown
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating weighted aura score: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}