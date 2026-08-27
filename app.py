import json
import os
import uuid

# ============================================================
# SSL FIX (Must be placed before any other imports)
# ============================================================

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)

# ============================================================
# IMPORTS
# ============================================================

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from web_rag_agent import (
    clear_session_history,
    generate_initial_content,
    query_chat_bot,
)

# ============================================================
# MODEL SETUP
# ============================================================

HF_MODEL_ID = "shindevaishnavi304/dinov2-finetuned-skin-disease"

print("=" * 65)
print("Loading skin disease classifier...")
print("=" * 65)

processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
model.eval()

print("Model loaded successfully.")
print("=" * 65)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def new_session():
    return str(uuid.uuid4())


def parse_llm_response(briefing_raw, questions_raw):
    clean_briefing = briefing_raw
    clean_questions = questions_raw or []

    if isinstance(briefing_raw, dict):
        clean_briefing = briefing_raw.get("briefing", briefing_raw.get("answer", str(briefing_raw)))
        clean_questions = briefing_raw.get("questions", clean_questions)
    elif isinstance(briefing_raw, str) and briefing_raw.strip().startswith("{"):
        try:
            parsed = json.loads(briefing_raw)
            clean_briefing = parsed.get("briefing", parsed.get("answer", briefing_raw))
            if "questions" in parsed and not clean_questions:
                clean_questions = parsed.get("questions", [])
        except Exception:
            clean_briefing = briefing_raw.replace('\\"', '"')

    if isinstance(clean_briefing, str):
        clean_briefing = clean_briefing.replace("\\n", "\n").strip()

    if isinstance(clean_questions, list):
        clean_questions = [str(q).strip() for q in clean_questions if str(q).strip()]
    else:
        clean_questions = []

    return clean_briefing, clean_questions


# ============================================================
# CLASSIFY IMAGE & START CONSULTATION
# ============================================================


def classify_and_start_chat(image, old_session_id):
    if image is None:
        return (
            [],
            "Please upload a dermoscopic skin image first.",
            None,
            old_session_id,
            gr.update(choices=[], value=None),
        )

    if old_session_id:
        clear_session_history(old_session_id)

    fresh_session_id = new_session()

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, prediction_id = torch.max(probabilities, dim=1)

    predicted_class = model.config.id2label[prediction_id.item()]
    confidence_percentage = confidence.item() * 100

    result_md = (
        f"**Predicted Condition:** `{predicted_class}`  \n"
        f"**Model Confidence:** `{confidence_percentage:.1f}%`"
    )

    try:
        raw_briefing, raw_questions = generate_initial_content(
            disease_label=predicted_class,
            session_id=fresh_session_id,
        )
        briefing, questions = parse_llm_response(raw_briefing, raw_questions)
    except Exception as e:
        print("Initial briefing error:", e)
        briefing = "Clinical briefing could not be generated."
        questions = []

    welcome_message = (
        "### 🔬 Lesion Analysis Complete\n\n"
        f"- **Predicted Condition:** `{predicted_class}`\n"
        f"- **Model Confidence:** `{confidence_percentage:.1f}%`\n\n"
        "---\n\n"
        f"{briefing}\n\n"
        "---\n\n"
        "💡 **Suggested Questions**\n"
        "Click any suggestion below or ask your own question below."
    )

    history = [{"role": "assistant", "content": welcome_message}]
    question_update = gr.update(choices=questions, value=None)

    return (
        history,
        result_md,
        predicted_class,
        fresh_session_id,
        question_update,
    )


# ============================================================
# CHAT HANDLERS
# ============================================================


def answer_suggested_question(question, history, predicted_class, session_id):
    if not question:
        return history, gr.update()

    if not predicted_class:
        return (
            history + [{"role": "assistant", "content": "⚠️ Please classify an image first."}],
            gr.update(choices=[], value=None),
        )

    try:
        raw_answer, raw_questions = query_chat_bot(
            user_input=question,
            disease_label=predicted_class,
            session_id=session_id,
        )
        answer, questions = parse_llm_response(raw_answer, raw_questions)
    except Exception as e:
        print("Suggested question error:", e)
        answer = "Sorry, I could not generate an answer."
        questions = []

    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return new_history, gr.update(choices=questions, value=None)


def respond(message, history, predicted_class, session_id):
    if not message or not message.strip():
        return history, "", gr.update()

    if not predicted_class:
        return (
            history + [{"role": "assistant", "content": "⚠️ Please classify an image first."}],
            "",
            gr.update(choices=[], value=None),
        )

    try:
        raw_answer, raw_questions = query_chat_bot(
            user_input=message,
            disease_label=predicted_class,
            session_id=session_id,
        )
        answer, questions = parse_llm_response(raw_answer, raw_questions)
    except Exception as e:
        print("Chatbot error:", e)
        answer = "Sorry, I could not generate an answer."
        questions = []

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return new_history, "", gr.update(choices=questions, value=None)


def reset_app(session_id):
    if session_id:
        clear_session_history(session_id)

    return (
        None,
        "",
        [],
        None,
        new_session(),
        gr.update(choices=[], value=None),
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Skin AI — Clinical Decision Support",
    theme=gr.themes.Soft(),
) as demo:

    session_id_state = gr.State(new_session())
    predicted_class_state = gr.State(None)

    gr.Markdown(
        """
# 👨‍⚕️ Skin AI — Clinical Decision Support

Upload a dermoscopic skin image to classify the condition.

The system provides an immediate clinical briefing covering **Severity, Malignancy, Diagnostic Tests, and First-line Treatments**.
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Image Classification")
            image_input = gr.Image(
                type="pil",
                label="Upload Dermoscopic Image",
            )
            classify_btn = gr.Button(
                "🔬 Classify & Start Consultation",
                variant="primary",
            )
            result_md = gr.Markdown()
            reset_btn = gr.Button(
                "🔄 Reset Consultation",
                variant="secondary",
            )

        with gr.Column(scale=3):
            gr.Markdown("### 2. Clinical AI Assistant")
            chatbot = gr.Chatbot(
                height=480,
                label="Clinical AI Assistant",
            )
            suggested_questions = gr.Radio(
                choices=[],
                value=None,
                label="💡 Dynamic Suggested Questions",
                interactive=True,
            )
            msg_input = gr.Textbox(
                placeholder="Ask about tests, symptoms, alternatives, recovery...",
                label=None,
            )
            send_btn = gr.Button("Send Question", variant="primary")

    # Event Listeners
    classify_btn.click(
        fn=classify_and_start_chat,
        inputs=[image_input, session_id_state],
        outputs=[
            chatbot,
            result_md,
            predicted_class_state,
            session_id_state,
            suggested_questions,
        ],
    )

    suggested_questions.change(
        fn=answer_suggested_question,
        inputs=[
            suggested_questions,
            chatbot,
            predicted_class_state,
            session_id_state,
        ],
        outputs=[chatbot, suggested_questions],
    )

    send_btn.click(
        fn=respond,
        inputs=[
            msg_input,
            chatbot,
            predicted_class_state,
            session_id_state,
        ],
        outputs=[chatbot, msg_input, suggested_questions],
    )

    msg_input.submit(
        fn=respond,
        inputs=[
            msg_input,
            chatbot,
            predicted_class_state,
            session_id_state,
        ],
        outputs=[chatbot, msg_input, suggested_questions],
    )

    reset_btn.click(
        fn=reset_app,
        inputs=[session_id_state],
        outputs=[
            image_input,
            result_md,
            chatbot,
            predicted_class_state,
            session_id_state,
            suggested_questions,
        ],
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    demo.launch()