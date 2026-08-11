import io
import os
import uuid
import certifi
import logging

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
try:
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import ALLOWED_ORIGINS, HF_MODEL_ID
from web_rag_agent import query_chat_bot, get_suggested_questions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-ai-backend")

app = FastAPI(title="Skin AI Clinical Decision Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model loading (once, at process start) -------------------------------

_processor = None
_model = None


def get_model():
    global _processor, _model
    if _model is None:
        logger.info("Loading DINOv2 classifier: %s", HF_MODEL_ID)
        _processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        _model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID, device_map="auto")
        _model.eval()
    return _processor, _model


@app.on_event("startup")
def warm_up_model():
    # Load the classifier eagerly so the first real request isn't slow.
    get_model()


# --- Schemas ----------------------------------------------------------------

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    suggested_questions: list[str]


class ChatRequest(BaseModel):
    message: str
    predicted_label: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str


class NewSessionResponse(BaseModel):
    session_id: str


# --- Routes -------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=NewSessionResponse)
def new_session():
    """Frontend calls this once per browser session to get a chat session id."""
    return {"session_id": str(uuid.uuid4())}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    try:
        processor, model = get_model()
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            conf, pred_id = torch.max(probs, dim=1)

        predicted_class = model.config.id2label[pred_id.item()]
        confidence = conf.item() * 100

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    return PredictResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 1),
        suggested_questions=get_suggested_questions(predicted_class),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        answer = query_chat_bot(
            user_input=req.message,
            predicted_label=req.predicted_label,
            session_id=req.session_id,
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {e}")

    return ChatResponse(answer=answer)
