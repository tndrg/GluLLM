LLM_CONFIGS = {
    'llama1': {'checkpoint': 'meta-llama/Llama-3.2-1B-Instruct', 'embed_dim': 2048},
    'llama3': {'checkpoint': 'meta-llama//Llama-3.2-3B-Instruct', 'embed_dim': 3072},
    'llama8': {'checkpoint': 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'embed_dim': 4096},
    'mistral': {'checkpoint': 'mistralaiMistral-7B-Instruct-v0.2', 'embed_dim': 4096},
    'gemma': {'checkpoint': 'google/gemma-1.1-2b-it', 'embed_dim': 2048},
    'phi': {'checkpoint': 'microsoft/Phi-3-mini-4k-instruct', 'embed_dim': 3072},
    'openelm': {'checkpoint': 'apple/OpenELM-270M', 'embed_dim': 1280},
    'gpt': {'checkpoint': 'openai-community/gpt2', 'embed_dim': 768},
}
