import os
import re
import uuid

# ============================================================
# SSL FIX (Must be placed before external imports)
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
# Fast, GPU, no LLM call here
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

    confidence_percentage = confidence.item() * 100

    return predicted_class, confidence_percentage


# ============================================================
# FORMATTING HELPERS
# ============================================================

def strip_followup_section(text):
    """
    Safety net: if the LLM ever writes its own
    "Follow-up Questions" section directly into the
    briefing/answer text despite being told not to,
    cut that section out of the chat text so it can't
    duplicate the clickable chip list below the chat.

    With web_rag_agent.py generating questions in a
    fully separate call now, this should rarely trigger.
    """

    if not text:
        return text

    heading_pattern = re.compile(
        r"\n{0,2}#{0,4}\s*\*\*\s*"
        r"(follow[\s-]?up questions|"
        r"questions a clinician might ask|"
        r"suggested (clinical )?(questions|inquiries)|"
        r"you might also ask)"
        r"[^\n]*\*\*\s*\n",
        re.IGNORECASE,
    )

    match = heading_pattern.search(text)

    if match:
        return text[:match.start()].rstrip()

    return text


def _confidence_tier(pct):
    """
    Returns:
        (label, css-class)
    for a confidence value.
    """

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
    """
    Builds the HTML for the model-prediction card.
    Kept separate from the streaming logic so the
    visual design lives in one place.
    """

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
    """
    Generator:
    Shows the classification result the instant it's ready
    then streams the clinical briefing in as it's generated.
    """

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

    # Show prediction immediately.
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
# Streamed
# ============================================================

def _stream_turn(
    user_message,
    history,
    predicted_class,
    session_id
):
    """
    Shared streaming logic for both the textbox and
    suggested-question flows.
    """

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
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');


/* ============================================================
   GLOBAL
   ============================================================ */

:root {

  color-scheme: light !important;

  --teal-900: #0B3D3B;
  --teal-700: #0F6B67;
  --teal-500: #2E9490;
  --teal-100: #E6F1F0;

  --coral-600: #D9503F;
  --coral-100: #FBE7E3;

  --amber-600: #C77E1E;
  --amber-100: #FBF0DC;

  --green-600: #2F9E58;
  --green-100: #E4F5EA;

  --ink-900: #16241F;
  --ink-600: #4B5A57;
}


.gradio-container {

  font-family:
    'Inter',
    ui-sans-serif,
    system-ui,
    sans-serif !important;

  max-width:
    1320px !important;

  margin-left:
    auto !important;

  margin-right:
    auto !important;

  color-scheme:
    light !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.app-header {

  background:
    linear-gradient(
      120deg,
      var(--teal-900) 0%,
      var(--teal-700) 65%,
      var(--teal-500) 100%
    );

  border-radius:
    20px;

  padding:
    32px 36px;

  margin-bottom:
    22px;

  color:
    #F4FAF9 !important;

  box-shadow:
    0 10px 30px
    rgba(11, 61, 59, 0.18);

  text-align:
    center;
}


.app-header h1 {

  font-family:
    'Fraunces',
    serif !important;

  font-weight:
    600 !important;

  font-size:
    2.1rem !important;

  margin:
    0 0 6px 0 !important;

  color:
    #FFFFFF !important;

  -webkit-text-fill-color:
    #FFFFFF !important;

  letter-spacing:
    -0.01em;
}


.app-header p {

  color:
    #D9EEEC !important;

  -webkit-text-fill-color:
    #D9EEEC !important;

  font-size:
    1rem !important;

  max-width:
    760px;

  line-height:
    1.55 !important;

  margin:
    0 auto !important;
}


.app-header .badge-row {

  display:
    flex;

  justify-content:
    center;

  gap:
    8px;

  margin-top:
    16px;

  flex-wrap:
    wrap;
}


.app-header .pill {

  background:
    rgba(255,255,255,0.12);

  border:
    1px solid
    rgba(255,255,255,0.25);

  color:
    #F4FAF9 !important;

  -webkit-text-fill-color:
    #F4FAF9 !important;

  padding:
    5px 12px;

  border-radius:
    999px;

  font-size:
    0.78rem;

  font-weight:
    500;
}


/* ============================================================
   SECTION LABELS
   ============================================================ */

.section-label {

  font-family:
    'Fraunces',
    serif !important;

  font-weight:
    600 !important;

  font-size:
    1.15rem !important;

  color:
    var(--teal-900) !important;

  -webkit-text-fill-color:
    var(--teal-900) !important;

  display:
    flex;

  align-items:
    center;

  gap:
    8px;

  margin-bottom:
    4px !important;
}


.section-label .step-num {

  display:
    inline-flex;

  align-items:
    center;

  justify-content:
    center;

  width:
    24px;

  height:
    24px;

  border-radius:
    50%;

  background:
    var(--teal-100);

  color:
    var(--teal-700) !important;

  -webkit-text-fill-color:
    var(--teal-700) !important;

  font-family:
    'JetBrains Mono',
    monospace;

  font-size:
    0.75rem;

  font-weight:
    600;
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.result-card {

  background:
    #FFFFFF !important;

  border:
    1px solid #E3E9E8;

  border-radius:
    16px;

  padding:
    20px 22px;

  margin-top:
    4px;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;

  color-scheme:
    light !important;
}


.result-card-empty {

  color:
    var(--coral-600) !important;

  -webkit-text-fill-color:
    var(--coral-600) !important;

  font-weight:
    500;

  text-align:
    center;

  padding:
    28px 20px;
}


.result-card .result-card-label {

  text-transform:
    uppercase;

  letter-spacing:
    0.08em;

  font-size:
    0.72rem;

  font-weight:
    600;

  color:
    var(--ink-600) !important;

  -webkit-text-fill-color:
    var(--ink-600) !important;

  margin-bottom:
    6px;
}


.result-card .result-condition {

  font-family:
    'Fraunces',
    serif !important;

  font-size:
    1.5rem;

  font-weight:
    600;

  color:
    var(--teal-900) !important;

  -webkit-text-fill-color:
    var(--teal-900) !important;

  margin-bottom:
    14px;
}


/* ============================================================
   CONFIDENCE
   ============================================================ */

.confidence-row {

  display:
    flex;

  align-items:
    center;

  gap:
    12px;

  margin-bottom:
    16px;
}


.confidence-track {

  flex:
    1;

  height:
    8px;

  background:
    var(--teal-100);

  border-radius:
    999px;

  overflow:
    hidden;
}


.confidence-fill {

  height:
    100%;

  background:
    linear-gradient(
      90deg,
      var(--teal-500),
      var(--teal-700)
    );

  border-radius:
    999px;
}


.confidence-chip {

  font-family:
    'JetBrains Mono',
    monospace;

  font-size:
    0.75rem;

  font-weight:
    600;

  padding:
    4px 10px;

  border-radius:
    999px;

  white-space:
    nowrap;
}


.chip-high {

  background:
    var(--green-100) !important;

  color:
    var(--green-600) !important;

  -webkit-text-fill-color:
    var(--green-600) !important;
}


.chip-mid {

  background:
    var(--amber-100) !important;

  color:
    var(--amber-600) !important;

  -webkit-text-fill-color:
    var(--amber-600) !important;
}


.chip-low {

  background:
    var(--coral-100) !important;

  color:
    var(--coral-600) !important;

  -webkit-text-fill-color:
    var(--coral-600) !important;
}


/* ============================================================
   RISK
   ============================================================ */

.risk-row {

  border-top:
    1px dashed #E3E9E8;

  padding-top:
    12px;

  font-size:
    0.92rem;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;

  display:
    flex;

  align-items:
    center;

  gap:
    8px;
}


.risk-row span {

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


.risk-dot {

  width:
    8px;

  height:
    8px;

  border-radius:
    50%;

  background:
    var(--ink-600) !important;

  flex-shrink:
    0;
}


.risk-loading .risk-dot {

  background:
    var(--amber-600) !important;

  animation:
    pulse 1.1s ease-in-out infinite;
}


@keyframes pulse {

  0%, 100% {
    opacity: 1;
  }

  50% {
    opacity: 0.35;
  }
}


/* ============================================================
   UPLOAD + BUTTONS
   ============================================================ */

#upload-col .image-container,
#upload-col [data-testid="image"] {

  border-radius:
    14px !important;
}


button#classify-btn,
.primary-cta {

  font-weight:
    600 !important;

  font-size:
    1rem !important;

  border-radius:
    12px !important;

  padding:
    12px 18px !important;

  letter-spacing:
    0.01em;
}


button#reset-btn {

  border-radius:
    12px !important;

  font-weight:
    500 !important;
}


/* ============================================================
   CHATBOT OUTER CONTAINER
   ============================================================ */

#clinical-chatbot {

  border-radius:
    16px !important;

  border:
    1px solid #E3E9E8 !important;

  color-scheme:
    light !important;

  overflow:
    hidden !important;

  width:
    100% !important;

  box-sizing:
    border-box !important;
}


/* ============================================================
   CHAT MESSAGE WRAPPER
   ============================================================ */

#clinical-chatbot .message-wrap {

  padding:
    4px 8px !important;

  width:
    100% !important;

  max-width:
    100% !important;

  box-sizing:
    border-box !important;
}


/* ============================================================
   CHAT MESSAGE
   ============================================================ */

#clinical-chatbot .message {

  border-radius:
    14px !important;

  font-size:
    0.98rem !important;

  line-height:
    1.6 !important;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;

  max-width:
    100% !important;

  min-width:
    0 !important;

  box-sizing:
    border-box !important;

  overflow:
    hidden !important;
}


/* Bot message */

#clinical-chatbot .bot {

  background:
    var(--teal-100) !important;

  border:
    1px solid #D3E7E5 !important;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


/* User message */

#clinical-chatbot .user {

  background:
    #FFFFFF !important;

  border:
    1px solid #E3E9E8 !important;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


/* ============================================================
   CHAT MARKDOWN
   ============================================================ */

#clinical-chatbot .message p,
#clinical-chatbot .message li,
#clinical-chatbot .message span,
#clinical-chatbot .message strong,
#clinical-chatbot .message em {

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


#clinical-chatbot .message h1,
#clinical-chatbot .message h2,
#clinical-chatbot .message h3,
#clinical-chatbot .message h4,
#clinical-chatbot .message h5,
#clinical-chatbot .message h6 {

  color:
    var(--teal-900) !important;

  -webkit-text-fill-color:
    var(--teal-900) !important;
}


#clinical-chatbot .message a {

  color:
    var(--teal-700) !important;

  -webkit-text-fill-color:
    var(--teal-700) !important;
}


/* ============================================================
   MARKDOWN TABLES
   ============================================================ */

/*
   IMPORTANT:

   Wide tables must NOT be squeezed into the phone width.

   The table is allowed to keep a readable width.
*/


#clinical-chatbot .message table {

  border-collapse:
    collapse !important;

  table-layout:
    auto !important;

  width:
    max-content !important;

  min-width:
    100% !important;

  max-width:
    none !important;
}


#clinical-chatbot .message th,
#clinical-chatbot .message td {

  padding:
    8px 10px !important;

  min-width:
    120px !important;

  white-space:
    normal !important;

  word-break:
    normal !important;

  overflow-wrap:
    normal !important;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


/* ============================================================
   TABLE SCROLL CONTAINER
   ============================================================ */

/*
   Markdown-generated tables can be wider than the phone.

   This prevents:

   T
   y
   pi
   ca
   l

   and instead gives the user horizontal scrolling.
*/

#clinical-chatbot .message .table-wrapper,
#clinical-chatbot .message .prose {

  min-width:
    0 !important;

  max-width:
    100% !important;
}


/*
   If the Markdown renderer wraps the table directly,
   allow horizontal overflow.
*/

#clinical-chatbot .message {

  overflow-x:
    hidden !important;
}


/*
   Table itself can extend beyond the content width.
*/

#clinical-chatbot .message table {

  overflow:
    visible !important;
}


/* ============================================================
   CODE
   ============================================================ */

#clinical-chatbot .message pre {

  max-width:
    100% !important;

  overflow-x:
    auto !important;

  white-space:
    pre !important;
}


#clinical-chatbot .message code {

  word-break:
    normal !important;

  overflow-wrap:
    normal !important;
}


/* ============================================================
   SUGGESTED QUESTIONS
   ============================================================ */

#suggested-questions {

  color-scheme:
    light !important;

  width:
    100% !important;
}


#suggested-questions .wrap {

  display:
    flex !important;

  flex-wrap:
    wrap !important;

  gap:
    8px !important;

  width:
    100% !important;

  box-sizing:
    border-box !important;
}


#suggested-questions label {

  display:
    flex !important;

  align-items:
    center !important;

  box-sizing:
    border-box !important;

  border:
    1px solid var(--teal-500) !important;

  background:
    #FFFFFF !important;

  color:
    var(--teal-900) !important;

  -webkit-text-fill-color:
    var(--teal-900) !important;

  border-radius:
    999px !important;

  padding:
    8px 16px !important;

  font-size:
    0.88rem !important;

  font-weight:
    500 !important;

  line-height:
    1.35 !important;

  transition:
    background 0.15s ease,
    color 0.15s ease;

  cursor:
    pointer !important;
}


#suggested-questions label span {

  color:
    var(--teal-900) !important;

  -webkit-text-fill-color:
    var(--teal-900) !important;

  white-space:
    normal !important;

  word-break:
    normal !important;

  overflow-wrap:
    break-word !important;

  line-height:
    1.35 !important;
}


#suggested-questions label:hover {

  background:
    var(--teal-100) !important;
}


#suggested-questions input[type="radio"] {

  width:
    1px !important;

  height:
    1px !important;

  opacity:
    0 !important;

  position:
    absolute !important;
}


#suggested-questions
input[type="radio"]:checked + span {

  color:
    var(--teal-700) !important;

  -webkit-text-fill-color:
    var(--teal-700) !important;
}


#suggested-questions
input[type="radio"]:focus-visible ~ * {

  outline:
    2px solid var(--teal-700);

  outline-offset:
    2px;
}


/* ============================================================
   MESSAGE INPUT
   ============================================================ */

#msg-row {

  align-items:
    flex-end;

  gap:
    8px;
}


#msg-input {

  min-width:
    0 !important;
}


#msg-input textarea {

  border-radius:
    12px !important;

  font-size:
    0.98rem !important;

  background:
    #FFFFFF !important;

  color:
    var(--ink-900) !important;

  -webkit-text-fill-color:
    var(--ink-900) !important;
}


#msg-input textarea::placeholder {

  color:
    #6B7A77 !important;

  -webkit-text-fill-color:
    #6B7A77 !important;

  opacity:
    1 !important;
}


/* ============================================================
   ACCESSIBILITY
   ============================================================ */

button:focus-visible,
textarea:focus-visible,
input:focus-visible,
[role="radio"]:focus-visible {

  outline:
    3px solid var(--teal-700) !important;

  outline-offset:
    2px !important;
}


/* ============================================================
   REDUCED MOTION
   ============================================================ */

@media (prefers-reduced-motion: reduce) {

  .risk-loading .risk-dot {

    animation:
      none;
  }
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

  .app-header {

    padding:
      24px 20px;
  }

  .app-header h1 {

    font-size:
      1.6rem !important;
  }
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {


  /* ----------------------------------------------------------
     LIGHT COLOR SCHEME
     ---------------------------------------------------------- */

  :root {

    color-scheme:
      light !important;
  }

  .gradio-container {

    color-scheme:
      light !important;
  }


  /* ----------------------------------------------------------
     RESULT CARD
     ---------------------------------------------------------- */

  .result-card {

    background:
      #FFFFFF !important;

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;

    color-scheme:
      light !important;
  }


  .result-card .result-card-label {

    color:
      #4B5A57 !important;

    -webkit-text-fill-color:
      #4B5A57 !important;
  }


  .result-card .result-condition {

    color:
      #0B3D3B !important;

    -webkit-text-fill-color:
      #0B3D3B !important;
  }


  .result-card .risk-row,
  .result-card .risk-row span {

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;
  }


  /* ----------------------------------------------------------
     CONFIDENCE
     ---------------------------------------------------------- */

  .result-card .chip-high {

    background:
      #E4F5EA !important;

    color:
      #2F9E58 !important;

    -webkit-text-fill-color:
      #2F9E58 !important;
  }


  .result-card .chip-mid {

    background:
      #FBF0DC !important;

    color:
      #C77E1E !important;

    -webkit-text-fill-color:
      #C77E1E !important;
  }


  .result-card .chip-low {

    background:
      #FBE7E3 !important;

    color:
      #D9503F !important;

    -webkit-text-fill-color:
      #D9503F !important;
  }


  /* ----------------------------------------------------------
     CHATBOT
     ---------------------------------------------------------- */

  #clinical-chatbot {

    width:
      100% !important;

    max-width:
      100% !important;

    color-scheme:
      light !important;

    box-sizing:
      border-box !important;

    overflow:
      hidden !important;
  }


  /*
     Keep chatbot messages as wide as possible.
  */

  #clinical-chatbot .message-wrap {

    width:
      100% !important;

    max-width:
      100% !important;

    padding:
      5px 8px !important;

    box-sizing:
      border-box !important;
  }


  #clinical-chatbot .message {

    width:
      100% !important;

    max-width:
      100% !important;

    min-width:
      0 !important;

    box-sizing:
      border-box !important;

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;

    font-size:
      0.96rem !important;

    line-height:
      1.55 !important;
  }


  /* ----------------------------------------------------------
     CHAT TEXT
     ---------------------------------------------------------- */

  #clinical-chatbot .message p,
  #clinical-chatbot .message li,
  #clinical-chatbot .message span,
  #clinical-chatbot .message strong,
  #clinical-chatbot .message em {

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;
  }


  #clinical-chatbot .message h1,
  #clinical-chatbot .message h2,
  #clinical-chatbot .message h3,
  #clinical-chatbot .message h4 {

    color:
      #0B3D3B !important;

    -webkit-text-fill-color:
      #0B3D3B !important;
  }


  /* ----------------------------------------------------------
     TABLE FIX
     ----------------------------------------------------------

     Tables should NOT compress their columns until every
     word becomes one character.

     They keep a readable minimum width.
  */

  #clinical-chatbot .message table {

    width:
      max-content !important;

    min-width:
      520px !important;

    max-width:
      none !important;

    table-layout:
      auto !important;

    border-collapse:
      collapse !important;
  }


  #clinical-chatbot .message th,
  #clinical-chatbot .message td {

    min-width:
      120px !important;

    padding:
      8px 10px !important;

    white-space:
      normal !important;

    word-break:
      normal !important;

    overflow-wrap:
      normal !important;

    font-size:
      0.88rem !important;

    line-height:
      1.45 !important;

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;
  }


  /*
     Markdown prose/table container.

     If Gradio places the table inside .prose, the
     prose area can scroll horizontally.
  */

  #clinical-chatbot .message .prose {

    width:
      100% !important;

    max-width:
      100% !important;

    min-width:
      0 !important;

    overflow-x:
      auto !important;

    overflow-y:
      visible !important;

    -webkit-overflow-scrolling:
      touch !important;
  }


  /*
     If a direct table wrapper exists.
  */

  #clinical-chatbot .message .table-wrapper {

    width:
      100% !important;

    max-width:
      100% !important;

    overflow-x:
      auto !important;

    overflow-y:
      visible !important;

    -webkit-overflow-scrolling:
      touch !important;
  }


  /*
     Prevent long URLs/code from destroying the layout.
  */

  #clinical-chatbot .message pre {

    max-width:
      100% !important;

    overflow-x:
      auto !important;
  }


  /* ----------------------------------------------------------
     SUGGESTED QUESTIONS
     ---------------------------------------------------------- */

  #suggested-questions {

    width:
      100% !important;

    color-scheme:
      light !important;

    box-sizing:
      border-box !important;
  }


  #suggested-questions .wrap {

    display:
      flex !important;

    flex-direction:
      column !important;

    flex-wrap:
      nowrap !important;

    width:
      100% !important;

    max-width:
      100% !important;

    gap:
      8px !important;

    box-sizing:
      border-box !important;
  }


  /*
     Each suggested question is a pill.

     Long questions wrap to a second line INSIDE
     the pill instead of becoming a textbox.
  */

  #suggested-questions label {

    display:
      flex !important;

    align-items:
      center !important;

    justify-content:
      flex-start !important;

    width:
      100% !important;

    max-width:
      100% !important;

    min-height:
      44px !important;

    box-sizing:
      border-box !important;

    padding:
      9px 18px !important;

    border-radius:
      999px !important;

    border:
      1px solid #2E9490 !important;

    background:
      #FFFFFF !important;

    color:
      #0B3D3B !important;

    -webkit-text-fill-color:
      #0B3D3B !important;

    font-size:
      0.88rem !important;

    font-weight:
      500 !important;

    line-height:
      1.35 !important;

    overflow:
      hidden !important;
  }


  #suggested-questions label span {

    display:
      block !important;

    width:
      100% !important;

    color:
      #0B3D3B !important;

    -webkit-text-fill-color:
      #0B3D3B !important;

    white-space:
      normal !important;

    word-break:
      normal !important;

    overflow-wrap:
      break-word !important;

    line-height:
      1.35 !important;
  }


  #suggested-questions label:hover {

    background:
      #E6F1F0 !important;
  }


  /* ----------------------------------------------------------
     MESSAGE INPUT
     ---------------------------------------------------------- */

  #msg-row {

    width:
      100% !important;

    max-width:
      100% !important;

    box-sizing:
      border-box !important;

    align-items:
      flex-end !important;

    gap:
      8px !important;
  }


  #msg-input {

    min-width:
      0 !important;

    flex:
      1 1 auto !important;
  }


  #msg-input textarea {

    width:
      100% !important;

    background:
      #FFFFFF !important;

    color:
      #16241F !important;

    -webkit-text-fill-color:
      #16241F !important;

    font-size:
      0.95rem !important;

    border-radius:
      12px !important;
  }


  #msg-input textarea::placeholder {

    color:
      #6B7A77 !important;

    -webkit-text-fill-color:
      #6B7A77 !important;

    opacity:
      1 !important;
  }
}
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Skin AI — Clinical Decision Support",
    theme=THEME,
    css=CUSTOM_CSS
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
    demo.launch()