---
title: Skin AI Clinical Decision Support
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
---


# Skin AI — Clinical Decision Support Dashboard
<img width="1917" height="861" alt="image" src="https://github.com/user-attachments/assets/01c27228-4acf-4224-a3a4-772bec9a952c" />
*Figure 1: Clinical Decision Support Dashboard*
## Introduction

Skin diseases impact the skin's surface, causing visible changes like rashes, inflammation, or itching. While causes vary from genetics to environmental factors, timely and accurate diagnosis remains challenging and costly, often leading to a significant diagnostic gap.

This project addresses these challenges by developing a robust, automated skin disease classification and decision support system. Utilizing advanced machine learning models (EfficientNet and Vision Transformer), it enables licensed clinicians to quickly and accurately classify diseases from dermoscopic skin images.

## Architecture

We have implemented a hybrid system architecture that combines a high-performance image classification pipeline (DINOv2) with an active, search-augmented chatbot memory (DuckDuckGo RAG). This architecture enables immediate, data-driven disease briefings while remembering session context for follow-up clinical questions.

**System Architecture Diagram:**

<img width="2816" height="1536" alt="Gemini_Generated_Image_osqmkjosqmkjosqm" src="https://github.com/user-attachments/assets/261b1ac6-158a-4567-b1b0-6f849c1b9a14" />

*Figure 2: System Architecture*

### Functional Overview

The system operates across three core stages:

#### 1. Real-time Image Classification Pipeline (Vision Transformer)
*   **Image Collection & Preprocessing:** A dermoscopic image is uploaded to the Gradio web UI. It is resized (e.g., 224x224), normalized, and converted to patches for the model.
*   **DINOv2 Fine-tuned Model:** We utilized Meta's DINOv2 self-attention-based transformer (fine-tuned on the DermNet dataset) to capture complex global relationships within the image.
*   **Prediction:** The model outputs the most likely disease label (26 classes) with a high confidence score.

#### 2. Hybrid RAG and Active Conversational Memory (Updates)
*   **Active Memory retained between turns:** To fix memory issues, we implemented a system that automatically maintains conversation context (past questions and answers) within the `web_rag_agent.py`.
*   **Isolated Session Memory:** To prevent context bleeding, every new image classification creates a fresh `session_id`, isolating that patient's data.

#### 3. Decision Support Chatbot & Automatic Analysis
*   **Automatically generated briefings:** When a lesion is classified, the system *automatically* executes a live DuckDuckGo search regarding that condition. Llama 3.1 synthesizes a briefing detailing:
    1.  **Severity & Risk Assessment.**
    2.  **Required & Possible Diagnostic Tests.**
    3.  **Suggested First-Line Treatments.**
*   **Internal Intelligence + RAG Fallback:** The chatbot relies on its medical intelligence by default, fallbacking to live web search only when required, enabling clinicians to ask follow-up questions (e.g., "What are its treatments?").

## Features

- Skin disease classification
- Confidence score
- Severity assessment
- Diagnostic test suggestions
- Treatment information
- Disease-specific suggested questions
- Clickable AI questions
- DuckDuckGo web search for relevant clinical information
- Groq-powered conversational assistant

## link for project
https://huggingface.co/spaces/shindevaishnavi304/Skin-AI-Disease-detection

## Future Scope

1.  **Mobile Deployment:** Transitioning the Gradio web dashboard to a lightweight mobile app for increased diagnostic accessibility.
2.  **Multimodal Diagnostics:** Integrating patient demographic data (age, symptoms, medical history) alongside the image for an even more precise, holistic diagnosis.
3.  **On-Device Inference:** Optimizing the Transformer models for edge deployment to ensure low-latency, private diagnostic processing without relying on cloud APIs.

## Authors

*   **Shinde Vaishnavi** [202402060016]
  <p align="left">
      <a href="https://github.com/Vaishnavi304-MIT"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
      <a href="https://www.linkedin.com/in/vaishnavi-shinde-40190b2b1/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  </p>
*   **Kalaskar Shreya** [202402060006]
*   **Yadav Pushkar** [202402060017]
*   **Aayush Katamkar** [202402060015]

Under the supervision of **Prof. Vinaya S. Tapkir**.

**MIT Academy of Engineering**
School of E&TC Engineering (SEE)
TY-BTech Major Project (SEM-VI), SEM-VII Project - II

## Conclusion

This study highlighted that while EfficientNet performs well on simpler datasets, transformer-based models like Vision Transformer (ViT) show better performance on complex skin datasets with many similar classes. By combining advanced classification with an automatic decision support chatbot, this project demonstrates the powerful potential of transformer-based models in assisting real-world medical tasks.

## Disclaimer

This application is for educational and research purposes.
The prediction is not a confirmed medical diagnosis.
A qualified healthcare professional should confirm any
clinical diagnosis.
