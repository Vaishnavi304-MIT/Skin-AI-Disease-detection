import os
import re
import uuid
from pathlib import Path

# ============================================================
# SSL FIX
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

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from web_rag_agent import (
    clear_session_history,
    generate_initial_content_stream,
    query_chat_bot_stream,
)

# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSS_FILE = BASE_DIR / "style.css"

with open(CSS_FILE, "r", encoding="utf-8") as f:
    CUSTOM_CSS = f.read()

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
# HELPERS
# ============================================================

def new_session():
    return str(uuid.uuid4())


# ============================================================
# CLASSIFY IMAGE
# ============================================================

@spaces.GPU
def classify_only(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction_id = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = model.config.id2label[
        prediction_id.item()
    ]

    confidence_percentage = (
        confidence.item() * 100
    )

    return predicted_class, confidence_percentage


# ============================================================
# FORMATTING HELPERS
# ============================================================

def strip_followup_section(text):
    """
    Removes a Follow-up Questions section if the LLM
    accidentally includes one in its response.
    """

    if not text:
        return text

    heading_pattern = re.compile(
        r"\n{0,2}#{0,4}\s*\*\\*\*\s*"
        r"(follow[\s-]?up questions|"
        r"questions a clinician might ask|"
        r"suggested (clinical )?(questions|inquiries)|"
        r"you might also ask)"
        r"[^\n]*\*\\*\*\s*\n",
        re.IGNORECASE,
    )

    match = heading_pattern.search(text)

    if match:
        return text[:match.start()].rstrip()

    return text


def _confidence_tier(pct):

    if pct >= 85:
        return "High confidence", "chip-high"

    if pct >= 60:
        return "Moderate confidence", "chip-mid"

    return "Low confidence", "chip-low"


def render_result_card(
    predicted_class,
    confidence_percentage,
    risk_badge=None,
    loading=False
):

    tier_label, tier_class = _confidence_tier(
        confidence_percentage
    )

    condition_display = (
        predicted_class
        .replace("_", " ")
        .title()
    )

    if loading:

        risk_html = (
            '<div class="risk-row risk-loading">'
            '<span class="risk-dot"></span>'
            '<span>Assessing risk level…</span>'
            '</div>'
        )

    elif risk_badge:

        risk_html = (
            '<div class="risk-row risk-ready">'
            f'{risk_badge}'
            '</div>'
        )

    else:

        risk_html = (
            '<div class="risk-row risk-unknown">'
            '<span class="risk-dot"></span>'
            '<span>Risk level unclear</span>'
            '</div>'
        )

    return f"""
<div class="result-card">

  <div class="result-card-label">
    Model Prediction
  </div>

  <div class="result-condition">
    {condition_display}
  </div>

  <div class="confidence-row">

    <div class="confidence-track">
      <div
        class="confidence-fill"
        style="width:{confidence_percentage:.1f}%"
      ></div>
    </div>

    <span class="confidence-chip {tier_class}">
      {confidence_percentage:.1f}% · {tier_label}
    </span>

  </div>

  {risk_html}

</div>
"""


# ============================================================
# CLASSIFY + STREAM CONSULTATION
# ============================================================

def classify_and_start_chat(
    image,
    old_session_id
):

    if image is None:

        yield (
            [],
            '<div class="result-card result-card-empty">'
            '⚠️ Please upload a dermoscopic skin image first.'
            '</div>',
            None,
            old_session_id,
            gr.update(
                choices=[],
                value=None
            ),
        )

        return

    if old_session_id:
        clear_session_history(old_session_id)

    fresh_session_id = new_session()

    predicted_class, confidence_percentage = (
        classify_only(image)
    )

    result_html = render_result_card(
        predicted_class,
        confidence_percentage,
        loading=True
    )

    history = [
        {
            "role": "assistant",
            "content": (
                "🔬 Analyzing the lesion and pulling "
                "current clinical guidance…"
            ),
        }
    ]

    yield (
        history,
        result_html,
        predicted_class,
        fresh_session_id,
        gr.update(
            choices=[],
            value=None
        ),
    )

    briefing = ""
    questions = []
    risk_badge = None

    try:

        for (
            partial_text,
            partial_questions,
            partial_risk
        ) in generate_initial_content_stream(
            disease_label=predicted_class,
            session_id=fresh_session_id
        ):

            briefing = strip_followup_section(
                partial_text
            )

            if partial_questions is not None:
                questions = partial_questions

            if partial_risk is not None:
                risk_badge = partial_risk

            welcome_message = (
                "### 🔬 Lesion Analysis Complete\n\n"
                f"{briefing}"
            )

            history = [
                {
                    "role": "assistant",
                    "content": welcome_message,
                }
            ]

            live_result_html = render_result_card(
                predicted_class,
                confidence_percentage,
                risk_badge=risk_badge,
                loading=risk_badge is None,
            )

            yield (
                history,
                live_result_html,
                predicted_class,
                fresh_session_id,
                gr.update(),
            )

    except Exception as e:

        print(
            "Initial briefing error:",
            e
        )

        history = [
            {
                "role": "assistant",
                "content": (
                    "⚠️ Clinical briefing could not "
                    "be generated. You can still ask "
                    "questions below."
                ),
            }
        ]

        fallback_html = render_result_card(
            predicted_class,
            confidence_percentage,
            risk_badge=risk_badge
        )

        yield (
            history,
            fallback_html,
            predicted_class,
            fresh_session_id,
            gr.update(
                choices=[],
                value=None
            ),
        )

        return

    final_result_html = render_result_card(
        predicted_class,
        confidence_percentage,
        risk_badge=risk_badge
    )

    yield (
        history,
        final_result_html,
        predicted_class,
        fresh_session_id,
        gr.update(
            choices=questions,
            value=None
        ),
    )


# ============================================================
# CHAT HANDLERS
# ============================================================

def _stream_turn(
    user_message,
    history,
    predicted_class,
    session_id
):

    if not predicted_class:

        yield (
            history
            + [
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ Please classify an image first."
                    ),
                },
            ],
            gr.update(
                choices=[],
                value=None
            ),
        )

        return

    working_history = history + [
        {
            "role": "user",
            "content": user_message,
        },
        {
            "role": "assistant",
            "content": "",
        },
    ]

    questions = []

    try:

        for (
            partial_answer,
            partial_questions
        ) in query_chat_bot_stream(
            user_input=user_message,
            disease_label=predicted_class,
            session_id=session_id,
        ):

            working_history[-1]["content"] = (
                strip_followup_section(
                    partial_answer
                )
            )

            if partial_questions is not None:
                questions = partial_questions

            yield (
                working_history,
                gr.update()
            )

    except Exception as e:

        print(
            "Chat error:",
            e
        )

        working_history[-1]["content"] = (
            "Sorry, I could not generate an answer."
        )

        yield (
            working_history,
            gr.update(
                choices=[],
                value=None
            ),
        )

        return

    yield (
        working_history,
        gr.update(
            choices=questions,
            value=None
        ),
    )


def answer_suggested_question(
    question,
    history,
    predicted_class,
    session_id
):

    if not question:

        yield (
            history,
            gr.update()
        )

        return

    yield from _stream_turn(
        question,
        history,
        predicted_class,
        session_id
    )


def respond(
    message,
    history,
    predicted_class,
    session_id
):

    if not message or not message.strip():

        yield (
            history,
            "",
            gr.update()
        )

        return

    for (
        new_history,
        question_update
    ) in _stream_turn(
        message,
        history,
        predicted_class,
        session_id
    ):

        yield (
            new_history,
            "",
            question_update
        )


def reset_app(session_id):

    if session_id:
        clear_session_history(session_id)

    return (
        None,
        "",
        [],
        None,
        new_session(),
        gr.update(
            choices=[],
            value=None
        ),
    )


# ============================================================
# THEME
# ============================================================

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,

    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],

    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "monospace",
    ],

).set(

    body_background_fill="#F4F7F7",

    block_background_fill="#FFFFFF",

    block_border_width="1px",

    block_border_color="#E3E9E8",

    block_radius="16px",

    block_shadow=(
        "0 1px 3px rgba(15, 61, 59, 0.06)"
    ),

    button_primary_background_fill="#0F6B67",

    button_primary_background_fill_hover="#0C5652",

    button_primary_text_color="#FFFFFF",

    button_secondary_background_fill="#FFFFFF",

    button_secondary_border_color="#0F6B67",

    button_secondary_text_color="#0F6B67",

    input_background_fill="#FFFFFF",

    input_border_color="#D8E2E1",

    input_border_color_focus="#0F6B67",
)


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Skin AI — Clinical Decision Support"
) as demo:

    session_id_state = gr.State(
        new_session()
    )

    predicted_class_state = gr.State(
        None
    )

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="app-header">

          <h1>
            🩺 Skin AI — Clinical Decision Support
          </h1>

          <p>
            Upload a dermoscopic skin image for instant
            classification, a risk read, recommended
            diagnostic tests, and first-line treatment
            guidance — built for clinician use.
          </p>

          <div class="badge-row">

            <span class="pill">
              Real-time classification
            </span>

            <span class="pill">
              Evidence-linked briefing
            </span>

            <span class="pill">
              Interactive follow-up Q&amp;A
            </span>

          </div>

        </div>
        """
    )

    # ========================================================
    # MAIN ROW
    # ========================================================

    with gr.Row(equal_height=False):

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column(
            scale=2,
            elem_id="upload-col"
        ):

            gr.Markdown(
                '<div class="section-label">'
                '<span class="step-num">1</span>'
                ' Image Classification'
                '</div>'
            )

            image_input = gr.Image(
                type="pil",
                label="Upload dermoscopic image",
                elem_id="image-input",
            )

            classify_btn = gr.Button(
                "🔬 Classify & Start Consultation",
                variant="primary",
                elem_id="classify-btn",
            )

            result_md = gr.HTML(
                '<div class="result-card '
                'result-card-empty">'
                "Upload an image to see the model's "
                "prediction here."
                "</div>"
            )

            reset_btn = gr.Button(
                "🔄 Reset Consultation",
                variant="secondary",
                elem_id="reset-btn",
            )

        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column(scale=3):

            gr.Markdown(
                '<div class="section-label">'
                '<span class="step-num">2</span>'
                ' Clinical AI Assistant'
                '</div>'
            )

            chatbot = gr.Chatbot(
                height=460,
                label="Clinical AI Assistant",
                elem_id="clinical-chatbot",
            )

            suggested_questions = gr.Radio(
                choices=[],
                value=None,
                label="💡 Suggested clinical inquiries",
                interactive=True,
                elem_id="suggested-questions",
            )

            with gr.Row(
                elem_id="msg-row"
            ):

                msg_input = gr.Textbox(
                    placeholder=(
                        "Ask about tests, symptoms, "
                        "alternatives, precautions…"
                    ),
                    label="Your question",
                    show_label=False,
                    scale=5,
                    elem_id="msg-input",
                )

                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_id="send-btn"
                )

    # ========================================================
    # EVENT LISTENERS
    # ========================================================

    classify_btn.click(
        fn=classify_and_start_chat,

        inputs=[
            image_input,
            session_id_state,
        ],

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

        outputs=[
            chatbot,
            suggested_questions,
        ],
    )

    send_btn.click(
        fn=respond,

        inputs=[
            msg_input,
            chatbot,
            predicted_class_state,
            session_id_state,
        ],

        outputs=[
            chatbot,
            msg_input,
            suggested_questions,
        ],
    )

    msg_input.submit(
        fn=respond,

        inputs=[
            msg_input,
            chatbot,
            predicted_class_state,
            session_id_state,
        ],

        outputs=[
            chatbot,
            msg_input,
            suggested_questions,
        ],
    )

    reset_btn.click(
        fn=reset_app,

        inputs=[
            session_id_state,
        ],

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
    demo.launch(
        theme=THEME,
        css=CUSTOM_CSS
    )