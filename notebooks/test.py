from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Verificação de dispositivo disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Carregar o modelo com técnicas de economia de memória
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    offload_folder="offload"  # Utiliza o disco para offloading se necessário
).to(device)

# Habilitar gradient checkpointing para economizar memória
model.gradient_checkpointing_enable()

# Exemplo de geração de texto
input_text = "O que é o mercado financeiro?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)

# Decodificar e imprimir a saída
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
