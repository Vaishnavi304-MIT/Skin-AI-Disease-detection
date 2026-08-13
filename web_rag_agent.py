import os
import certifi
from ddgs import DDGS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import GROQ_API_KEY

# Fix Windows SSL Certificate loading issues
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

try:
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

model = ChatGroq(model="openai/gpt-oss-120b", temperature=1, groq_api_key=GROQ_API_KEY)

system_prompt = (
    "You are an expert Clinical Dermatology AI Assistant for doctors and medical professionals.\n"
    "The currently classified disease for this session is: {disease_label}.\n\n"
    "CRITICAL CLINICAL DIRECTIVES:\n"
    "When providing briefings or answering questions regarding {disease_label}, ensure you cover:\n"
    "1. **Severity & Risk Assessment**: Specify if benign, premalignant, malignant, acute, or infectious, along with red-flag symptoms.\n"
    "2. **Required & Possible Diagnostic Tests**: Recommend confirmatory tests (e.g., Dermoscopy, Punch/Shave Biopsy, KOH Prep, Wood's Lamp, Skin Swab/PCR, Patch Testing, or Blood Work).\n"
    "3. **Suggested First-Line Treatments**: Outline evidence-based therapies (topical, oral, procedural, or surgical guidelines).\n\n"
    "Integrate findings from the live search context below. If search context is limited, rely on your internal clinical intelligence.\n\n"
    "Web Search Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def clear_session_history(session_id: str):
    """Clears conversation memory for a specific session."""
    if session_id in store:
        del store[session_id]

def fetch_duckduckgo_context(disease_label: str, query: str = "", max_results: int = 4) -> str:
    """Performs live web search for diagnostic tests, severity, and treatments."""
    search_term = f"{disease_label} severity diagnostic tests biopsy treatment guidelines {query}".strip()
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=search_term, max_results=max_results))
            if not results:
                return "No live search results required."
            return "\n".join([f"[{i+1}] {r.get('title', '')}: {r.get('body', '')}" for i, r in enumerate(results)])
    except Exception:
        return "No live search results required."

def query_chat_bot(user_input: str, disease_label: str = "Unspecified Skin Condition", session_id: str = "default_session") -> str:
    search_context = fetch_duckduckgo_context(disease_label=disease_label, query=user_input)
    
    chain = qa_prompt | model
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    response = conversational_chain.invoke(
        {
            "disease_label": disease_label if disease_label else "Unspecified Skin Condition",
            "context": search_context,
            "input": user_input
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    return response.content