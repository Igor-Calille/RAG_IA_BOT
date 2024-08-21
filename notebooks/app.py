import torch
import numpy as np
import faiss
import json
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import os

# Configurar dispositivo CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU disponível')
    print('Número de GPUs disponíveis:', torch.cuda.device_count())
    print('Nome da GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('GPU não disponível, utilizando CPU')

# Limpar cache da GPU para liberar memória
torch.cuda.empty_cache()

# Configurar variáveis de ambiente
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carregar o modelo BERTimbau e o tokenizer
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)

# Carregar dados de perguntas e respostas
data = []
with open('financial_market_qa.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

inputs = [item['input'] for item in data]
outputs = [item['output'] for item in data]

# Função para obter embeddings das perguntas usando BERTimbau
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu()

# Gerar embeddings para todas as perguntas
question_embeddings = np.array([get_embedding(question).numpy().flatten() for question in inputs])

# Configurar Faiss para busca vetorial
d = question_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(question_embeddings)

# Função para recuperar respostas relevantes
def retrieve_relevant_answers(query, k=5):
    query_embedding = get_embedding(query).numpy().flatten().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [outputs[i] for i in indices[0]]

# Carregar o modelo de linguagem BERTimbau para geração de texto
lm_model = BertForMaskedLM.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)

# Função para gerar resposta final usando BERTimbau (Masked LM)
def generate_response(query):
    relevant_answers = retrieve_relevant_answers(query)
    context = " ".join(relevant_answers) + " " + query
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = lm_model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Testar a geração de resposta
response = generate_response("O que é o índice Bovespa?")
print(response.encode('utf-8').decode('utf-8'))
