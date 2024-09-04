from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Verificação de dispositivo disponível (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o tokenizer e o modelo BERTimbau para QA
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
model = BertForQuestionAnswering.from_pretrained("neuralmind/bert-large-portuguese-cased")

# Mover o modelo para o dispositivo (GPU ou CPU)
model.to(device)

# Definir o contexto e a pergunta
context = """
A economia brasileira cresceu 1,2% no segundo trimestre de 2021, impulsionada pelo aumento no consumo das famílias
e pelos investimentos em infraestrutura. O crescimento foi maior do que o esperado por economistas e trouxe
otimismo para o mercado financeiro.
"""
question = "Qual foi o crescimento da economia brasileira no segundo trimestre de 2021?"

# Tokenização do contexto e da pergunta
inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)

# Fazer a inferência para obter as posições de início e fim da resposta
with torch.no_grad():
    outputs = model(**inputs)

# Extrair as posições de início e fim da resposta
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

# Decodificar a resposta utilizando o tokenizer
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

# Exibir a resposta
print(f"Pergunta: {question}")
print(f"Resposta: {answer}")
