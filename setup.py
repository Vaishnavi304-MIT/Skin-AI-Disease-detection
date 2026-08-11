
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRA_DIRS = [
    os.path.join(PROJECT_ROOT, "raw_dataset"),
    os.path.join(PROJECT_ROOT, "processed_dataset"),
    os.path.join(PROJECT_ROOT, "augmented_dataset"),
    os.path.join(PROJECT_ROOT, "medical_guidelines"),
]

FILES = {
    os.path.join(PROJECT_ROOT, "__init__.py"): "",

    os.path.join(PROJECT_ROOT, "requirements.txt"): """torch>=2.1.0
torchvision
timm
albumentations
opencv-python
ultralytics
pillow
numpy
scikit-learn
matplotlib
seaborn
onnx
onnx2tf
tensorflow
python-dotenv
langchain-huggingface
langchain-pinecone
langchain-community
langchain-groq
langchain-chroma
pinecone-client
duckduckgo-search
huggingface_hub
pypdf
streamlit
bs4
""",

    os.path.join(PROJECT_ROOT, "config.py"): """import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATASET_DIR = "./raw_dataset"
PROCESSED_DATASET_DIR = "./processed_dataset"
AUGMENTED_DATASET_DIR = "./augmented_dataset"
MEDICAL_DOCS_DIR = "./medical_guidelines"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 3e-4

PINECONE_INDEX_NAME = "rag-knowledge-index"
HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/mobilevit-skin-disease-detection")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

os.environ["USER_AGENT"] = "MedicalRAGBot/1.0"
""",

    os.path.join(PROJECT_ROOT, "utils.py"): """import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dull_razor(image_rgb: np.ndarray, filter_kernel_size: int = 9, inpaint_radius: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_kernel_size, filter_kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cleaned_image = cv2.inpaint(image_rgb, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned_image

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()
""",

    os.path.join(PROJECT_ROOT, "preprocess.py"): """import os
import glob
import cv2
import numpy as np
import albumentations as A
from ultralytics import SAM
from utils import dull_razor
from config import RAW_DATASET_DIR, PROCESSED_DATASET_DIR, AUGMENTED_DATASET_DIR

sam_model = SAM("mobile_sam.pt")

def get_skin_augmentor():
    return A.Compose([
        A.ShiftScaleRotate(scale_limit=(-0.2, 0.3), shift_limit=0.05, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.15, 0.15), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

def segment_lesion_with_sam(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    results = sam_model(image_rgb, points=[[w // 2, h // 2]], labels=[1], verbose=False)
    if results and len(results[0].masks) > 0:
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return image_rgb

def process_dataset(num_augmented_copies=4):
    augmentor = get_skin_augmentor()
    class_folders = [f for f in os.listdir(RAW_DATASET_DIR) if os.path.isdir(os.path.join(RAW_DATASET_DIR, f))]

    for class_name in class_folders:
        raw_class_path = os.path.join(RAW_DATASET_DIR, class_name)
        proc_class_path = os.path.join(PROCESSED_DATASET_DIR, class_name)
        aug_class_path = os.path.join(AUGMENTED_DATASET_DIR, class_name)

        os.makedirs(proc_class_path, exist_ok=True)
        os.makedirs(aug_class_path, exist_ok=True)

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in glob.glob(os.path.join(raw_class_path, ext)):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                clean_rgb = dull_razor(img_rgb)
                segmented_rgb = segment_lesion_with_sam(clean_rgb)

                cv2.imwrite(os.path.join(proc_class_path, f"{base_name}_base.png"), cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))

                for i in range(num_augmented_copies):
                    aug = augmentor(image=segmented_rgb)['image']
                    cv2.imwrite(os.path.join(aug_class_path, f"{base_name}_aug_{i}.png"), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

    print("Preprocessing Complete!")

if __name__ == "__main__":
    process_dataset()
""",

    os.path.join(PROJECT_ROOT, "dataset.py"): """import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224):
        self.img_paths = []
        self.labels = []
        class_names = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        for name in class_names:
            class_folder = os.path.join(root_dir, name)
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for path in glob.glob(os.path.join(class_folder, ext)):
                    self.img_paths.append(path)
                    self.labels.append(self.class_to_idx[name])

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(image=image)['image']
        return tensor_img, torch.tensor(self.labels[idx], dtype=torch.long)
""",

    os.path.join(PROJECT_ROOT, "model.py"): """import torch
import timm

def build_mobilevit(num_classes: int, pretrained: bool = True):
    model = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=num_classes)
    return model
""",

    os.path.join(PROJECT_ROOT, "train.py"): """import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from dataset import SkinDiseaseDataset
from model import build_mobilevit
from utils import FocalLoss
from config import AUGMENTED_DATASET_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE

@torch.no_grad()
def evaluate_and_generate_metrics(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

    print("\\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Skin Disease Detection - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'.\\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SkinDiseaseDataset(root_dir=AUGMENTED_DATASET_DIR, img_size=IMG_SIZE)
    
    class_idx_map = dataset.class_to_idx
    class_names = [name for name, idx in sorted(class_idx_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_mobilevit(num_classes=num_classes, pretrained=True).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {train_loss / len(train_ds):.4f}")

    torch.save(model.state_dict(), "best_mobilevit_skin.pth")
    print("\\nModel saved as 'best_mobilevit_skin.pth'.")

    evaluate_and_generate_metrics(model, val_loader, class_names, device)

if __name__ == "__main__":
    main()
""",

    os.path.join(PROJECT_ROOT, "export_and_publish.py"): """import os
import torch
import timm
from huggingface_hub import HfApi, create_repo
from config import HF_REPO_ID, HF_TOKEN, AUGMENTED_DATASET_DIR

# Auto-detect number of classes from dataset
num_classes = len([f for f in os.listdir(AUGMENTED_DATASET_DIR) if os.path.isdir(os.path.join(AUGMENTED_DATASET_DIR, f))]) or 4

model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("best_mobilevit_skin.pth", map_location="cpu"))
model.eval()

torch.save(model.state_dict(), "mobilevit_skin.pth")

dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "mobilevit_skin.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

os.system(f"onnx2tf -i {onnx_path} -o tflite_dir -oiqt")
tflite_path = "mobilevit_skin_int8.tflite"
if os.path.exists("tflite_dir/mobilevit_skin_quant_int8.tflite"):
    os.rename("tflite_dir/mobilevit_skin_quant_int8.tflite", tflite_path)

if HF_TOKEN:
    api = HfApi(token=HF_TOKEN)
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj="mobilevit_skin.pth", path_in_repo="mobilevit_skin.pth", repo_id=HF_REPO_ID)
    api.upload_file(path_or_fileobj=tflite_path, path_in_repo=tflite_path, repo_id=HF_REPO_ID)
""",

    os.path.join(PROJECT_ROOT, "web_rag_agent.py"): """import os
import warnings
from typing import Dict
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

TARGET_URLS = [
    "https://www.ncbi.nlm.nih.gov/books/NBK430774/",  # Actinic Keratosis
    "https://www.ncbi.nlm.nih.gov/books/NBK448071/",  # Atopic Dermatitis
    "https://www.ncbi.nlm.nih.gov/books/NBK544286/",  # Benign Keratosis
    "https://www.ncbi.nlm.nih.gov/books/NBK470538/",  # Dermatofibroma
    "https://www.ncbi.nlm.nih.gov/books/NBK538232/",  # Melanocytic Nevus
    "https://www.ncbi.nlm.nih.gov/books/NBK470409/",  # Melanoma
    "https://www.ncbi.nlm.nih.gov/books/NBK441939/",  # Squamous Cell Carcinoma
    "https://www.ncbi.nlm.nih.gov/books/NBK448149/",  # Tinea/Candidiasis
    "https://www.ncbi.nlm.nih.gov/books/NBK532882/",  # Vascular Lesions
]

print("Initializing Real-Time Web Scraping Engine...")
loader = WebBaseLoader(web_paths=TARGET_URLS)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY)

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
    "Use the following retrieved context to answer the user's question.\n"
    "If you don't know the answer or if the context is insufficient, explicitly state: "
    "'I am not sure based on the provided context.' Do not fabricate answers.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def query_chat_bot(user_input: str, session_id: str = "default_doctor_session"):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]
""",

    os.path.join(PROJECT_ROOT, "app.py"): """import os
import streamlit as st
import cv2
import numpy as np
import torch
import timm
from PIL import Image
from web_rag_agent import query_chat_bot
from config import AUGMENTED_DATASET_DIR

st.set_page_config(page_title="Skin AI Decision Support", layout="wide")
st.title("👨‍⚕️ Clinical Decision Support System")

# Auto-detect class folders dynamically
if os.path.exists(AUGMENTED_DATASET_DIR):
    classes = sorted([f for f in os.listdir(AUGMENTED_DATASET_DIR) if os.path.isdir(os.path.join(AUGMENTED_DATASET_DIR, f))])
else:
    classes = ["Actinic Keratosis", "Atopic Dermatitis", "Benign Keratosis", "Dermatofibroma", 
               "Melanocytic Nevus", "Melanoma", "Squamous Cell Carcinoma", "Tinea Ringworm", "Vascular Lesion"]

num_classes = len(classes)

@st.cache_resource
def load_vision_model():
    model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load("best_mobilevit_skin.pth", map_location="cpu"))
    model.eval()
    return model

tab1, tab2 = st.tabs(["📷 Lesion Classification", "💬 Real-Time Guideline Chatbot"])

with tab1:
    uploaded_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Lesion", width=300)
        
        img = np.array(image.convert('RGB'))
        img = cv2.resize(img, (224, 224)) / 255.0
        tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        try:
            model = load_vision_model()
            with torch.no_grad():
                outputs = torch.softmax(model(tensor), dim=1)
                conf, pred = torch.max(outputs, 1)
                
            st.success(f"**Predicted Diagnosis:** {classes[pred.item()]} ({conf.item()*100:.1f}% confidence)")
        except Exception as e:
            st.error("Trained model weights ('best_mobilevit_skin.pth') not found. Please run train.py first.")

with tab2:
    st.subheader("Interactive Physician Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask a clinical query regarding skin guidelines..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing web guidelines & memory history..."):
                answer = query_chat_bot(user_prompt, session_id="streamlit_session")
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
"""
}

def setup():
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    for folder in EXTRA_DIRS:
        os.makedirs(folder, exist_ok=True)
        print(f"Created Directory: {folder}")

    for filepath, content in FILES.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created File: {filepath}")

    print("\nProject setup complete! All modules successfully updated.")

if __name__ == "__main__":
    setup()

