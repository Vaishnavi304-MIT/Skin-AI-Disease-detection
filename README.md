# 🩺 Skin AI — Skin Disease Detection and Clinical Decision Support

> **An AI-powered skin disease detection system that combines DINOv2 image classification with a conversational clinical assistant to provide disease information, diagnostic tests, severity assessment, treatment guidance, and relevant web-based information.**

---

## 🚩 Problem Statement

Skin diseases can be difficult to identify because many conditions have similar visual appearances.

Traditional diagnosis may require:

- Dermatological examination
- Dermoscopy
- Laboratory tests
- Biopsy
- Specialist consultation

This can require time and access to trained healthcare professionals.

Therefore, this project aims to provide an **AI-assisted system** that can analyze a skin image and provide useful clinical information to support further evaluation.

> ⚠️ This system is intended for educational and research purposes. It does **not** provide a confirmed medical diagnosis.

---

# 💡 Solution

The proposed system combines **deep-learning image classification** with a **conversational AI assistant**.

The system works in two main stages:

### Stage 1 — Skin Disease Classification

A fine-tuned **DINOv2** image classification model analyzes the uploaded skin image and predicts the most likely skin disease along with a confidence score.

### Stage 2 — Clinical AI Assistant

After classification, the predicted disease is passed to a conversational AI assistant.

The assistant can provide:

- Severity information
- Malignancy information
- Warning signs
- Relevant diagnostic tests
- Treatment information
- Follow-up information
- Answers to user questions

For relevant clinical questions, the system can also retrieve information using **DuckDuckGo web search** before generating the response.

---

# ✨ Key Features

## 🔬 1. Skin Disease Image Classification

The system accepts a skin/dermoscopic image and uses a fine-tuned DINOv2 model to predict the condition.

It provides:

- Predicted condition
- Model confidence score
- Fast image classification

Example:

```text
Predicted Condition: Bullous Disease
Model Confidence: 87.4%
