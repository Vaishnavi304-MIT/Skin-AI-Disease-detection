import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file in backend/ "
        "(see .env.example) or export it in your environment."
    )

# Comma-separated list of origins allowed to call this API, e.g.
# "http://localhost:3000,https://your-frontend.vercel.app"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "shindevaishnavi304/dinov2-finetuned-skin-disease")
