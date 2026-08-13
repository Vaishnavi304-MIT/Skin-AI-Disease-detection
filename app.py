import os
import uuid
import certifi

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
try:
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from web_rag_agent import query_chat_bot, clear_session_history

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "shindevaishnavi304/dinov2-finetuned-skin-disease")

print(f"Loading classifier: {HF_MODEL_ID}")
processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
model.eval()


def new_session():
    return str(uuid.uuid4())


def classify_and_start_chat(image: Image.Image, old_session_id: str):
    if image is None:
        return [], "Please upload a dermoscopic skin image first.", None, old_session_id

    if GROQ_API_KEY is None:
        return (
            [],
            "⚠️ GROQ_API_KEY is not set. Please set it in your environment variables.",
            None,
            old_session_id
        )

    # 1. Clear previous session memory & initialize a new session ID
    clear_session_history(old_session_id)
    fresh_session_id = new_session()

    # 2. Perform image classification
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, pred_id = torch.max(probs, dim=1)

    predicted_class = model.config.id2label[pred_id.item()]
    confidence = conf.item() * 100

    result_md = f"**Predicted Diagnosis:** `{predicted_class}`  \n**Confidence:** `{confidence:.1f}%`"

    # 3. Prompt chatbot for Clinical Assessment (Severity, Tests, Treatments)
    briefing_prompt = (
        f"Generate a structured clinical evaluation for {predicted_class}:\n"
        f"1. **Severity & Risk Assessment**: Is it benign, premalignant, malignant, acute, or contagious? State red flags.\n"
        f"2. **Possible & Required Diagnostic Tests**: What tests are needed to confirm the diagnosis (e.g., Dermoscopy, Biopsy, KOH Prep, Swab/PCR)?\n"
        f"3. **Suggested First-Line Treatments**: Outline standard topical, oral, or procedural treatments."
    )
    
    clinical_briefing = query_chat_bot(
        user_input=briefing_prompt,
        disease_label=predicted_class,
        session_id=fresh_session_id
    )

    welcome_msg = (
        f"🔬 **Lesion Analysis Complete**\n\n"
        f"• **Predicted Condition:** `{predicted_class}`\n"
        f"• **Confidence Score:** `{confidence:.1f}%`\n\n"
        f"---\n\n"
        f"{clinical_briefing}\n\n"
        f"---\n"
        f"*Ask any follow-up questions regarding test interpretations or treatment adjustments.*"
    )
    
    new_history = [{"role": "assistant", "content": welcome_msg}]
    return new_history, result_md, predicted_class, fresh_session_id


def respond(message, history, predicted_class, session_id):
    if not message or not message.strip():
        return history, ""

    updated_history = history + [{"role": "user", "content": message}]
    
    try:
        answer = query_chat_bot(
            user_input=message,
            disease_label=predicted_class if predicted_class else "Unspecified Skin Condition",
            session_id=session_id,
        )
    except Exception as e:
        answer = f"⚠️ Error generating response: {e}"

    updated_history = updated_history + [{"role": "assistant", "content": answer}]
    return updated_history, ""


def reset_app(session_id):
    clear_session_history(session_id)
    return None, "", [], None, new_session()


with gr.Blocks(title="Skin AI — Clinical Decision Support", theme=gr.themes.Soft()) as demo:
    session_id_state = gr.State(new_session())
    predicted_class_state = gr.State(None)

    gr.Markdown(
        "# 👨‍⚕️ Skin AI — Clinical Decision Support\n"
        "Upload a dermoscopic skin image to classify the condition. The AI assistant automatically evaluates severity, recommends confirmatory diagnostic tests, and suggests evidence-based treatments."
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Image Classification")
            image_input = gr.Image(type="pil", label="Upload Dermoscopic Image")
            classify_btn = gr.Button("Classify & Start Consultation", variant="primary")
            result_md = gr.Markdown()
            reset_btn = gr.Button("Reset Consultation", variant="secondary")

        with gr.Column(scale=3):
            gr.Markdown("### 2. Clinical AI Assistant")
            chatbot = gr.Chatbot(height=480)
            msg_input = gr.Textbox(
                placeholder="Ask follow-up questions (e.g., 'When is a biopsy strictly required?')...",
                label=None,
                interactive=True
            )
            send_btn = gr.Button("Send Question", variant="primary")

    # Event Handlers
    classify_btn.click(
        fn=classify_and_start_chat,
        inputs=[image_input, session_id_state],
        outputs=[chatbot, result_md, predicted_class_state, session_id_state],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, predicted_class_state, session_id_state],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, predicted_class_state, session_id_state],
        outputs=[chatbot, msg_input],
    )

    reset_btn.click(
        fn=reset_app,
        inputs=[session_id_state],
        outputs=[image_input, result_md, chatbot, predicted_class_state, session_id_state],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)