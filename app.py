import os
import certifi

# Fix potential SSL certificate loading issues
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

try:
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from web_rag_agent import query_chat_bot, get_suggested_questions

st.set_page_config(page_title="Skin AI Decision Support", layout="wide")
st.title("👨‍⚕️ Clinical Decision Support System (DINOv2)")


# Load DINOv2 processor and fine-tuned model directly from Hugging Face
@st.cache_resource
def load_dinov2_model():
    processor = AutoImageProcessor.from_pretrained("shindevaishnavi304/dinov2-finetuned-skin-disease")
    model = AutoModelForImageClassification.from_pretrained(
        "shindevaishnavi304/dinov2-finetuned-skin-disease",
        device_map="auto"
    )
    return processor, model


tab1, tab2 = st.tabs(["📷 Lesion Classification (DINOv2)", "💬 Real-Time Guideline Chatbot"])

with tab1:
    uploaded_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Lesion Image", width=300)

        with st.spinner("Classifying lesion using fine-tuned DINOv2 Transformer..."):
            try:
                processor, model = load_dinov2_model()

                # Preprocess image and infer via DINOv2
                inputs = processor(images=image, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_id = torch.max(probs, dim=1)

                predicted_class = model.config.id2label[pred_id.item()]
                confidence_score = conf.item() * 100

                # Store prediction so tab2 can build the disease-specific chatbot
                if st.session_state.get("predicted_disease") != predicted_class:
                    # New diagnosis -> reset the chat so old messages/suggestions don't linger
                    st.session_state["messages"] = []
                st.session_state["predicted_disease"] = predicted_class

                st.success(f"**Predicted Diagnosis:** {predicted_class} ({confidence_score:.1f}% confidence)")
                st.info("Reference material for this diagnosis is now available in the chatbot tab.")

            except Exception as e:
                st.error(f"Error classifying image: {e}")

with tab2:
    st.subheader("Interactive Physician Chatbot")

    predicted_disease = st.session_state.get("predicted_disease")

    if not predicted_disease:
        st.warning("Classify a lesion image in the first tab before asking clinical questions.")
    else:
        st.caption(f"Chatbot context: **{predicted_disease}**")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show suggested questions only before the conversation has started
        if not st.session_state.messages:
            suggestions = get_suggested_questions(predicted_disease)
            if suggestions:
                st.markdown("**Suggested questions:**")
                for question in suggestions:
                    if st.button(question, key=f"suggest_{question}", use_container_width=True):
                        st.session_state["pending_prompt"] = question

        # Render chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Prefer a suggestion click; otherwise fall back to normal chat input
        user_prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
            "Ask a clinical query regarding disease guidelines..."
        )

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"Analyzing web guidelines for {predicted_disease}..."):
                    try:
                        answer = query_chat_bot(
                            user_prompt,
                            predicted_label=predicted_disease,
                            session_id="streamlit_session",
                        )
                    except KeyError as e:
                        answer = f"⚠️ {e}"
                    except Exception as e:
                        answer = f"⚠️ Error generating response: {e}"
                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})