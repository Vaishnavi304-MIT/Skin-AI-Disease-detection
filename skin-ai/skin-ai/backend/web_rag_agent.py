import warnings
import difflib
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config import GROQ_API_KEY

warnings.filterwarnings('ignore')

# Map each predicted disease label -> reference URL(s) for that disease only.
# Keys must match (or closely match, via the fuzzy fallback below) the labels
# in model.config.id2label of the DINOv2 classifier.
DISEASE_URL_MAP: Dict[str, List[str]] = {
    "Actinic Keratosis": ["https://www.jaad.org/article/S0190-9622(21)00502-8/fulltext"],
    "Atopic Dermatitis": ["https://www.jaad.org/article/S0190-9622(23)02878-5/fulltext"],
    "Benign Keratosis": ["https://www.ncbi.nlm.nih.gov/books/NBK545285/"],
    "Dermatofibroma": ["https://www.ncbi.nlm.nih.gov/books/NBK470538/"],
    "Melanocytic Nevus": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC9320830/"],
    "Melanoma": ["https://www.ncbi.nlm.nih.gov/books/NBK470409/"],
    "Squamous Cell Carcinoma": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC7319751/"],
    "Tinea/Candidiasis": ["https://www.ncbi.nlm.nih.gov/books/NBK544360/"],
    "Vascular Lesions": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC7007481/"],
}

SUGGESTED_QUESTIONS: Dict[str, List[str]] = {
    "Actinic Keratosis": [
        "What are the first-line treatment options for actinic keratosis?",
        "What is the malignant transformation risk if left untreated?",
        "How do I differentiate this from early SCC clinically?",
    ],
    "Atopic Dermatitis": [
        "What is the recommended step-up treatment ladder for atopic dermatitis?",
        "When should systemic therapy be considered over topical treatment?",
        "What are the key diagnostic criteria for atopic dermatitis?",
    ],
    "Benign Keratosis": [
        "How is a benign keratosis distinguished from a melanoma on exam?",
        "Is treatment necessary for benign keratosis, or is monitoring sufficient?",
        "What are typical dermoscopic features of benign keratosis?",
    ],
    "Dermatofibroma": [
        "What is the classic dermoscopic 'sign' associated with dermatofibroma?",
        "When should a dermatofibroma be biopsied or excised?",
        "How do dermatofibromas typically present and progress?",
    ],
    "Melanocytic Nevus": [
        "What ABCDE features would suggest a nevus needs biopsy?",
        "How often should a benign-appearing melanocytic nevus be monitored?",
        "What distinguishes a dysplastic nevus from a common nevus?",
    ],
    "Melanoma": [
        "What staging workup is recommended after a melanoma diagnosis?",
        "What are the current first-line treatment guidelines for melanoma?",
        "What margins are recommended for excision based on Breslow depth?",
    ],
    "Squamous Cell Carcinoma": [
        "What are the risk factors for metastasis in cutaneous SCC?",
        "What is the recommended surgical margin for low-risk SCC?",
        "When is Mohs surgery indicated over standard excision?",
    ],
    "Tinea/Candidiasis": [
        "How do I differentiate tinea from candidiasis clinically?",
        "What is the first-line antifungal treatment and duration?",
        "When is oral antifungal therapy indicated over topical?",
    ],
    "Vascular Lesions": [
        "What are the treatment options for symptomatic vascular lesions?",
        "How do I distinguish a vascular lesion from a pigmented lesion on dermoscopy?",
        "When is laser therapy indicated for vascular lesions?",
    ],
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Caches so we only scrape/embed a given disease once per process
_retriever_cache: Dict[str, object] = {}
_chain_cache: Dict[str, object] = {}

model = ChatGroq(model="openai/gpt-oss-120b", temperature=1, groq_api_key=GROQ_API_KEY)

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just reformulate it "
    "if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

system_prompt = (
    "You are an expert Clinical Dermatology AI Assistant for doctors.\n"
    "The patient's lesion has been classified as: {disease_name}.\n"
    "Use the following retrieved context to answer the user's question.\n"
    "If you don't know the answer or if the context is insufficient, explicitly state: "
    "'I am not sure based on the provided context.' Do not fabricate answers.\n\n"
    "{context}"
)

store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def _resolve_disease_key(predicted_label: str) -> str:
    """Match the model's raw label to our URL map, tolerating minor naming differences."""
    if predicted_label in DISEASE_URL_MAP:
        return predicted_label
    match = difflib.get_close_matches(predicted_label, DISEASE_URL_MAP.keys(), n=1, cutoff=0.4)
    if match:
        return match[0]
    raise KeyError(
        f"No reference URL mapped for predicted label '{predicted_label}'. "
        f"Known keys: {list(DISEASE_URL_MAP.keys())}"
    )


def _load_docs_safely(urls: List[str]):
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url], requests_kwargs={"timeout": 15})
            loader.session.headers.update(REQUEST_HEADERS)
            docs.extend(loader.load())
            print(f"[OK] Loaded: {url}")
        except Exception as e:
            print(f"[SKIP] Failed to load {url}: {e}")
    return docs


def _build_retriever_for(disease_key: str):
    if disease_key in _retriever_cache:
        return _retriever_cache[disease_key]

    urls = DISEASE_URL_MAP[disease_key]
    print(f"Scraping reference material for: {disease_key}")
    docs = _load_docs_safely(urls)
    if not docs:
        raise RuntimeError(f"Could not load any reference content for '{disease_key}'.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    _retriever_cache[disease_key] = retriever
    return retriever


def _build_chain_for(disease_key: str):
    if disease_key in _chain_cache:
        return _chain_cache[disease_key]

    retriever = _build_retriever_for(disease_key)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.replace("{disease_name}", disease_key)),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    _chain_cache[disease_key] = conversational_rag_chain
    return conversational_rag_chain


def get_suggested_questions(predicted_label: str) -> List[str]:
    try:
        disease_key = _resolve_disease_key(predicted_label)
    except KeyError:
        return []
    return SUGGESTED_QUESTIONS.get(disease_key, [])


def query_chat_bot(user_input: str, predicted_label: str, session_id: str = "default_doctor_session") -> str:
    """
    predicted_label: the raw class name from model.config.id2label for the
    currently classified image.
    """
    disease_key = _resolve_disease_key(predicted_label)
    chain = _build_chain_for(disease_key)
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]
