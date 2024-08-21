from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Autenticar-se usando o token
login(HUGGING_FACE_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
