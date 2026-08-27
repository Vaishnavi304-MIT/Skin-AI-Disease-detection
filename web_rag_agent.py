import json
import os
import re

# ============================================================
# SSL FIX (Must be placed before external imports)
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
# GROQ & SEARCH
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024,
    groq_api_key=GROQ_API_KEY,
)

search = DuckDuckGoSearchRun()

# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are Skin AI, a clinical decision support assistant in dermatology.

Predicted condition:
{disease_label}

Use concise, direct, and structured clinical language.

RULES:
- Answer accurately using bullet points and bold section headers.
- Clearly state whether a condition is Benign, Premalignant, or Malignant.
- Use current real-world diagnostic tests and first-line treatments.
- Integrate the provided web context accurately.
- The image prediction is NOT a confirmed diagnosis.

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
# MEMORY
# ============================================================

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def clear_session_history(session_id: str):
    store.pop(session_id, None)


# ============================================================
# WEB SEARCH
# ============================================================


def search_web(disease: str, query_type: str = "general", question: str = "") -> str:
    """Executes targeted searches on DuckDuckGo."""
    if query_type == "tests":
        query = f"{disease} diagnostic tests criteria biopsy laboratory panel dermoscopy dermatology guidelines"
    elif query_type == "initial":
        query = f"{disease} dermatology clinical severity malignancy diagnostic tests first-line treatment guidelines"
    else:
        query = f"{disease} {question} dermatology clinical diagnosis treatment"

    print("=" * 65)
    print("DUCKDUCKGO SEARCH")
    print("Query:", query)

    try:
        result = search.run(query)
        if not result:
            print("No search results returned.")
            return ""
        return result[:2500]
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return ""


# ============================================================
# INITIAL BRIEFING + QUESTIONS
# ============================================================


def generate_initial_content(disease_label: str, session_id: str):
    # Targeted search specifically gathering diagnostic tests and guidelines
    test_context = search_web(disease=disease_label, query_type="tests")
    general_context = search_web(disease=disease_label, query_type="initial")
    combined_context = f"TEST DATA:\n{test_context}\n\nGENERAL CLINICAL DATA:\n{general_context}"

    chain = prompt | model
    chat = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    request = f"""
Provide a structured clinical briefing for: '{disease_label}'.
Use the real-time test information from the web search to supply accurate, current diagnostic tests.

Format the briefing in Markdown with these exact sections:
### ⚠️ Severity & Malignancy
- **Risk Level:** Benign / Pre-malignant / Malignant
- **Clinical Severity:** Expected progression, severity, and warning signs.

### 🧪 Diagnostic Tests Required
- **Current Confirmatory Tests:** List 2 to 4 specific diagnostic/laboratory tests based on current guidelines (e.g., Punch Biopsy, Dermoscopic features, Direct Immunofluorescence, Patch test, Blood panels). Explain briefly why each is required.

### 💊 Recommended Treatments
- **First-line / Topical:** Standard topical or primary therapies.
- **Systemic / Procedural:** Second-line, systemic treatments, or procedural interventions.

Also generate exactly 4 relevant follow-up questions tailored to {disease_label}.

Output MUST be valid JSON:
{{
    "briefing": "markdown string for briefing",
    "questions": [
        "question 1",
        "question 2",
        "question 3",
        "question 4"
    ]
}}
"""

    response = chat.invoke(
        {
            "disease_label": disease_label,
            "context": combined_context,
            "input": request,
        },
        config={"configurable": {"session_id": session_id}},
    )

    text = response.content.strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)
        briefing = data.get("briefing", "").strip()
        questions = data.get("questions", [])

        if not isinstance(questions, list):
            questions = []

        questions = [str(q).strip() for q in questions if str(q).strip()][:4]
        return briefing, questions

    except Exception as e:
        print("JSON parsing failed, falling back:", e)
        default_briefing = f"""### ⚠️ Severity & Malignancy
- **Risk Level:** Clinical confirmation required (Rule out malignancy).
- **Clinical Severity:** Variable severity depending on lesion presentation.

### 🧪 Diagnostic Tests Required
- **Dermoscopy:** Inspection of vascular patterns and pigment distribution.
- **Skin Biopsy:** Punch or shave biopsy for histopathological evaluation.

### 💊 Recommended Treatments
- **First-line:** Topical therapeutic agents.
- **Specialist Care:** Consultation with a dermatologist."""

        return default_briefing, [
            f"What specific biopsy is needed for {disease_label}?",
            f"Is {disease_label} malignant or benign?",
            f"What are the best treatments for {disease_label}?",
            f"What warning signs require immediate care?",
        ]


# ============================================================
# CONVERSATIONAL CHATBOT WITH DYNAMIC QUESTIONS
# ============================================================


def query_chat_bot(
    user_input: str,
    disease_label: str = "Unknown skin condition",
    session_id: str = "default",
):
    """Answers user queries and generates fresh follow-up questions based on the prompt."""
    context = search_web(disease=disease_label, query_type="general", question=user_input)

    chain = prompt | model
    chat = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    request = f"""
User Query: "{user_input}"

1. Provide a direct, structured clinical answer to the user query.
2. Generate 4 new, contextual follow-up questions the user might ask next based on what you just answered.

Output MUST be valid JSON:
{{
    "answer": "markdown string answer",
    "questions": [
        "follow-up question 1",
        "follow-up question 2",
        "follow-up question 3",
        "follow-up question 4"
    ]
}}
"""

    response = chat.invoke(
        {
            "disease_label": disease_label,
            "context": context,
            "input": request,
        },
        config={"configurable": {"session_id": session_id}},
    )

    text = response.content.strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        data = json.loads(text)
        answer = data.get("answer", "").strip()
        questions = data.get("questions", [])

        if not isinstance(questions, list):
            questions = []

        questions = [str(q).strip() for q in questions if str(q).strip()][:4]
        return answer, questions
    except Exception:
        # Fallback if the model outputs regular text
        return text, [
            "What are the side effects of this treatment?",
            "What diagnostic tests should I request from my doctor?",
            "What symptoms indicate a worsening condition?",
            "What are non-medical lifestyle changes that help?",
        ]