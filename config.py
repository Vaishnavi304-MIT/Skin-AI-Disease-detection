import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATASET_DIR = "./raw_dataset"
PROCESSED_DATASET_DIR = "./processed_dataset"
AUGMENTED_DATASET_DIR = "./augmented dataset"
MEDICAL_DOCS_DIR = "./medical_guidelines"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 3e-4

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_REPO_ID = os.getenv("HF_REPO_ID")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

os.environ["USER_AGENT"] = "MedicalRAGBot/1.0"
