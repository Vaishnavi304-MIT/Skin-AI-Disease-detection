import os
import re
import functools

# ============================================================
# SSL FIX (Must run before network imports)
# ============================================================

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)

# ============================================================
# IMPORTS
# ============================================================

from config import GROQ_API_KEY
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# ============================================================
# MODEL SETUP
# ============================================================
# Two model instances: a larger-budget one for the first, structured
# briefing, and a leaner one for fast follow-up chat turns. Follow-up
# answers don't need 900 tokens, so cutting max_tokens there directly
# cuts latency.
#
# NOTE: neither model's max_tokens budget has to cover a trailing
# "===QUESTIONS===" block anymore -- question generation is a fully
# separate, unlinked call (see _generate_followups below). That was
# the root cause of suggested questions silently falling back to
# generic text: long answers (especially ones with web context) would
# eat the whole token budget before the model ever reached the
# questions section.

briefing_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1.0,
    max_tokens=800,
    groq_api_key=GROQ_API_KEY,
)

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=450,
    groq_api_key=GROQ_API_KEY,
)

search = DuckDuckGoSearchRun()

# ============================================================
# SESSION MEMORY
# ============================================================

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def clear_session_history(session_id: str):
    store.pop(session_id, None)


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are Skin AI, a clinical decision support assistant for DERMATOLOGISTS.
The end user is a licensed clinician, not a patient — you may name specific
drugs, dosages, routes and durations for their professional review.

Condition identified: {disease_label}

CRITICAL RULES:
- Always use easy language which is understandable.
- Use clean Markdown with bullet points and bold section headers.
- Be concise and scannable — clinicians skim. No filler sentences.
- Extract and cite the latest diagnostic tests / treatment facts from the
  web context when it is provided. If web context is empty, answer from
  established clinical knowledge and say so briefly.
- Never output raw JSON braces or JSON keys.
- The Risk Level line MUST use exactly one of these three words:
  Benign, Pre-malignant, or Malignant.
- Answer ONLY the clinical question. Do not write your own "follow-up
  questions" section — that is generated separately.

WEB CONTEXT:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ============================================================
# DUCKDUCKGO SEARCH — cached + combined to cut calls/tokens
# ============================================================

# Words that signal a follow-up question actually needs fresh web facts.
# Kept broad on purpose: false negatives (skipping a needed search) hurt
# clinical accuracy more than the cost of an occasional extra search.
SEARCH_TRIGGER_KEYWORDS = {
    "test", "biopsy", "treatment", "cure", "drug", "medication", "medicine",
    "dose", "dosage", "malignan", "cancer", "guideline", "latest", "recent",
    "study", "trial", "protocol", "fda", "approved", "alternative",
    "side effect", "interaction", "recurrence", "prognosis", "staging",
}

# Short conversational turns that never need a search, checked first so
# they short-circuit before the trigger-keyword scan even runs.
NO_SEARCH_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|sure|got it|great|cool|yes|no)\W*\s*$",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=128)
def _cached_search(query: str) -> str:
    print("=" * 65)
    print("DUCKDUCKGO SEARCH")
    print("Query:", query)
    try:
        res = search.run(query)
        return res[:2000] if res else ""
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return ""


def search_initial_briefing(disease: str) -> str:
    """One combined query instead of two — halves the initial-load search
    latency and DuckDuckGo calls while still covering tests, severity and
    treatment in a single pass."""
    query = (
        f"{disease} dermatology diagnostic tests biopsy dermoscopy "
        f"severity malignancy first-line treatment guidelines"
    )
    return _cached_search(query)


def needs_search(user_input: str) -> bool:
    if NO_SEARCH_PATTERNS.match(user_input):
        return False
    text = user_input.lower()
    return any(kw in text for kw in SEARCH_TRIGGER_KEYWORDS)


def search_followup(disease: str, question: str) -> str:
    query = f"{disease} {question} dermatology clinical guidance"
    return _cached_search(query)


# ============================================================
# OUTPUT PARSING (kept for any legacy/non-streaming callers)
# ============================================================


def parse_clean_output(text: str, fallback_questions: list):
    """Cleanly splits main briefing/answer from questions without JSON parsing
    issues. Only relevant if a prompt still asks the model to emit a trailing
    ===QUESTIONS=== block (the streaming paths below no longer do this)."""
    text = re.sub(r"```json|```", "", text).strip()

    if "===QUESTIONS===" in text:
        parts = text.split("===QUESTIONS===")
        main_content = parts[0].strip()
        raw_qs = parts[1].strip().split("\n")
        questions = [
            re.sub(r"^[-*0-9.]+\s*", "", q).strip()
            for q in raw_qs
            if q.strip() and not q.strip().startswith("{")
        ][:4]
        return main_content, questions if questions else fallback_questions

    return text.strip(), fallback_questions


_RISK_PATTERN = re.compile(
    r"risk level\**:?\**\s*(benign|pre-malignant|malignant)", re.IGNORECASE
)

RISK_BADGES = {
    "benign": "🟢 **Low Risk — Benign**",
    "pre-malignant": "🟡 **Moderate Risk — Pre-malignant**",
    "malignant": "🔴 **High Risk — Malignant**",
}


def extract_risk_level(text: str) -> str:
    match = _RISK_PATTERN.search(text)
    if match:
        return RISK_BADGES[match.group(1).lower()]
    return "⚪ **Risk level unclear — review manually**"


def _generate_followups(disease_label, last_question, last_answer, default_qs):
    """Grounded ONLY in the most recent exchange (ignores earlier history
    on purpose). Isolated model.invoke() call — no memory read/write."""
    followup_prompt = f"""
Condition: {disease_label}
Most recent clinician question: "{last_question}"
Most recent AI answer: "{last_answer[:1200]}"

Based ONLY on the exchange above (ignore any earlier conversation),
list exactly 4 short follow-up questions a CLINICIAN would to AI next, remember doctor is asking ai for help not askking question to patient
One per line. No numbering, no headers, no JSON, no extra commentary.
"""
    try:
        resp = chat_model.invoke(followup_prompt)
        raw_qs = [q.strip() for q in resp.content.split("\n") if q.strip()]
        qs = [re.sub(r"^[-*0-9.]+\s*", "", q).strip() for q in raw_qs]
        qs = [q for q in qs if q][:4]
        return qs if qs else default_qs
    except Exception as e:
        print("Follow-up generation error:", e)
        return default_qs


# ============================================================
# CHAIN HELPERS
# ============================================================


def _make_chat(model):
    chain = prompt | model
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# ============================================================
# INITIAL BRIEFING PROMPT (shared by streaming + non-streaming)
# ============================================================

_BRIEFING_REQUEST_TEMPLATE = """
Using the web context provided, create a structured clinical briefing for '{disease_label}'.

Format the response using this exact structure:

### ⚠️ Severity & Malignancy
- **Risk Level:** Benign / Pre-malignant / Malignant (pick exactly one word)
- **Clinical Severity:** Expected course and key warning signs.

### 🧪 Diagnostic Tests Required
- 2 to 4 current diagnostic/confirmatory tests (e.g., Punch Biopsy, Dermoscopy pattern, Direct Immunofluorescence, Blood panel). One line each on why it's indicated.

### 💊 Recommended Treatments & Medications
- **First-line / Topical:** Name specific drugs, typical strength, and regimen (e.g., "Imiquimod 5% cream, 3x/week for 6 weeks").
- **Systemic / Procedural:** Current clinical advancement in treatment and second-line drugs or procedures with typical dosing/route.

### 🛡️ Key Precautions
- 2 critical precautions or infection-control measures.
"""

_DEFAULT_INITIAL_QUESTION_LABEL = "Give me the initial clinical briefing for this condition."


def _default_initial_questions(disease_label):
    return [
        f"What are the main differential diagnoses for {disease_label}?",
        f"What specific biopsy technique is indicated for {disease_label}?",
        f"What are the standard dosages for first-line therapies?",
        f"What complications should be monitored?",
    ]


def _default_followup_questions(disease_label):
    return [
        f"What are the differential diagnoses for {disease_label}?",
        f"What additional confirmatory tests should be ordered?",
        f"What are common side effects of this treatment?",
        f"What precautions help prevent recurrence?",
    ]


# ============================================================
# INITIAL BRIEFING (non-streaming, kept for compatibility)
# ============================================================


def generate_initial_content(disease_label: str, session_id: str):
    briefing, questions, _risk = generate_initial_content_full(
        disease_label, session_id
    )
    return briefing, questions


def generate_initial_content_full(disease_label: str, session_id: str):
    """Returns (briefing, questions, risk_badge)."""
    context = search_initial_briefing(disease_label)
    chat = _make_chat(briefing_model)

    request = _BRIEFING_REQUEST_TEMPLATE.format(disease_label=disease_label)

    response = chat.invoke(
        {
            "disease_label": disease_label,
            "context": context,
            "input": request,
        },
        config={"configurable": {"session_id": session_id}},
    )

    briefing = re.sub(r"```json|```", "", response.content).strip()
    risk_badge = extract_risk_level(briefing)
    default_qs = _default_initial_questions(disease_label)
    questions = _generate_followups(
        disease_label, _DEFAULT_INITIAL_QUESTION_LABEL, briefing, default_qs
    )
    return briefing, questions, risk_badge


def generate_initial_content_stream(disease_label: str, session_id: str):
    """Streams the briefing token-by-token for a Claude-like typing feel.
    Yields display-ready partial markdown. The final yield is a 3-tuple:
    (full_text, questions, risk_badge) -- questions/risk are None on every
    intermediate yield and only populated once, after the answer is fully
    streamed and a separate, unlinked follow-up-question call has run."""
    context = search_initial_briefing(disease_label)
    chat = _make_chat(briefing_model)

    request = _BRIEFING_REQUEST_TEMPLATE.format(disease_label=disease_label)

    buffer = ""
    for chunk in chat.stream(
        {
            "disease_label": disease_label,
            "context": context,
            "input": request,
        },
        config={"configurable": {"session_id": session_id}},
    ):
        buffer += chunk.content or ""
        display_text = re.sub(r"```json|```", "", buffer).strip()
        yield display_text, None, None

    briefing = re.sub(r"```json|```", "", buffer).strip()
    risk_badge = extract_risk_level(briefing)
    default_qs = _default_initial_questions(disease_label)
    questions = _generate_followups(
        disease_label, _DEFAULT_INITIAL_QUESTION_LABEL, briefing, default_qs
    )
    yield briefing, questions, risk_badge


# ============================================================
# CHATBOT QUERY (non-streaming, kept for compatibility)
# ============================================================


def query_chat_bot(
    user_input: str,
    disease_label: str = "Unknown skin condition",
    session_id: str = "default",
):
    context = (
        search_followup(disease_label, user_input)
        if needs_search(user_input)
        else ""
    )

    chat = _make_chat(chat_model)

    response = chat.invoke(
        {
            "disease_label": disease_label,
            "context": context,
            "input": user_input,
        },
        config={"configurable": {"session_id": session_id}},
    )

    answer = re.sub(r"```json|```", "", response.content).strip()
    default_qs = _default_followup_questions(disease_label)
    questions = _generate_followups(disease_label, user_input, answer, default_qs)
    return answer, questions


def query_chat_bot_stream(
    user_input: str,
    disease_label: str = "Unknown skin condition",
    session_id: str = "default",
):
    """Streams the chat answer. Yields (partial_answer, None) while
    streaming, then a final (full_answer, questions) tuple. `questions`
    is always generated from THIS turn's user_input/answer only -- it
    doesn't matter whether user_input came from the textbox or from a
    clicked suggested-question chip; both paths call this function with
    a plain question string, so behavior is identical either way."""
    context = (
        search_followup(disease_label, user_input)
        if needs_search(user_input)
        else ""
    )

    chat = _make_chat(chat_model)

    buffer = ""
    for chunk in chat.stream(
        {
            "disease_label": disease_label,
            "context": context,
            "input": user_input,
        },
        config={"configurable": {"session_id": session_id}},
    ):
        buffer += chunk.content or ""
        display_text = re.sub(r"```json|```", "", buffer).strip()
        yield display_text, None

    answer = re.sub(r"```json|```", "", buffer).strip()
    default_qs = _default_followup_questions(disease_label)
    questions = _generate_followups(disease_label, user_input, answer, default_qs)
    yield answer, questions