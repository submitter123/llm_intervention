from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

PYTHIA_LOCAL_PATH = '/YOUR_PATH/.cache/huggingface/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c/'
GPT2_LOCAL_PATH = '/YOUR_PATH/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/'
LLAMA_3_8B_PATH = '/YOUR_PATH/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
GEMMA_2_9B_PATH = '/YOUR_PATH/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819'

def get_hooked_pythia_70m(device):
    """
    Load HookedTransformer pythia-70m from local path
    """
    
    auto_model = AutoModelForCausalLM.from_pretrained(PYTHIA_LOCAL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_LOCAL_PATH, local_files_only=True)
    
    pythia = HookedTransformer.from_pretrained(
        'EleutherAI/pythia-70m-deduped',
        hf_model=auto_model,
        tokenizer=tokenizer,
        device=device,
        local_files_only=True
    )
    pythia.eval()
    return pythia

def get_hooked_gpt2_small(device):
    """
    Load HookedTransformer gpt2-small from local path
    """
    
    auto_model = AutoModelForCausalLM.from_pretrained(GPT2_LOCAL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(GPT2_LOCAL_PATH, local_files_only=True)
        
    gpt2 = HookedTransformer.from_pretrained(
        'gpt2-small',
        hf_model=auto_model,
        tokenizer=tokenizer,
        device=device,
        local_files_only=True
    )
    gpt2.eval()
    return gpt2

def get_gpt2_small(device):
    gpt2 = AutoModelForCausalLM.from_pretrained(GPT2_LOCAL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(GPT2_LOCAL_PATH)
    gpt2.eval()
    return gpt2, tokenizer

def get_pythia_70m(device):
    pythia = AutoModelForCausalLM.from_pretrained(PYTHIA_LOCAL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_LOCAL_PATH)
    pythia.eval()
    return pythia, tokenizer

def get_llama_3_8B():
    llama = AutoModelForCausalLM.from_pretrained(LLAMA_3_8B_PATH, device_map='auto', torch_dtype='auto', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_3_8B_PATH, local_files_only=True)
    llama.eval()
    return llama, tokenizer

def get_gemma_2_9B():
    gemma = AutoModelForCausalLM.from_pretrained(GEMMA_2_9B_PATH, device_map='auto', torch_dtype='auto', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_2_9B_PATH, local_files_only=True)
    gemma.eval()
    return gemma, tokenizer
