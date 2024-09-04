import torch
import numpy as np
import faiss
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Configurar dispositivo CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limpar cache da GPU para liberar memória
torch.cuda.empty_cache()

# Evitar erro do OpenMP (atenção para o risco)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carregar o tokenizer e o modelo T5
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-3b", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b").to(device)

# Carregar dados de perguntas e respostas
data = []
with open('financial_market_qa.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

inputs = [item['input'] for item in data]
outputs = [item['output'] for item in data]

# Função para obter embeddings das perguntas usando T5 (Encoder)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
    return encoder_outputs.last_hidden_state.mean(dim=1).cpu()

# Gerar embeddings para todas as perguntas
question_embeddings = np.array([get_embedding(question).numpy().flatten() for question in inputs])

# Configurar Faiss para busca vetorial
d = question_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(question_embeddings)

# Função para recuperar respostas relevantes
def retrieve_relevant_answers(query, k=3):
    query_embedding = get_embedding(query).numpy().flatten().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [outputs[i] for i in indices[0]]

# Função para gerar resposta final usando T5
def generate_response(query):
    relevant_answers = retrieve_relevant_answers(query, k=3)
    context = " ".join(relevant_answers) + " " + query

    # Preparar o input para o modelo T5
    input_text = f"question: {query} context: {context}. Responda com detalhes."
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(device)

    # Geração de resposta com parâmetros ajustados
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,  # Permitir respostas mais longas
        num_beams=5,     # Aumentar o número de beams para uma geração mais refinada
        temperature=0.9, # Aumentar a criatividade na geração
        top_p=0.95,      # Nucleus sampling para variedade e precisão
        early_stopping=False  # Desativar early_stopping para gerar respostas mais longas
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return response

# Testar a geração de resposta
response = generate_response("O que é análise técnica?")
print(response)
