from transformers import AutoTokenizer
import transformers
import torch

# model = "codellama/CodeLlama-7b-hf"
model = "codellama/CodeLlama-7b-Instruct-hf"


tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model
    # torch_dtype=torch.float16,
    # device_map="auto",
)

# sequences = pipeline(
#     'import socket\n\ndef ping_exponential_backoff(host: str):',
#     do_sample=True,
#     top_k=10,
#     temperature=0.1,
#     top_p=0.95,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )

messages = [
    {"role": "user", "content": "生命的意义是什么?"},
]
sequences = pipeline(messages, max_length=200)
print(sequences)
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")