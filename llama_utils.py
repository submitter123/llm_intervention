import gc
import os
import json
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from collections import defaultdict
from utils import get_sae_features_by_layer
from enum import Enum
from typing import Callable, Union, List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

def get_overlap_features_full_info(overlap_features_file: str, save_path: str) -> dict:
    with open(overlap_features_file, 'r') as f:
        data = json.load(f)
    
    feature_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for token, features in data['common_token_features'].items():
        for feature in features:
            model = feature['model']
            layer_type = feature['layer_type']
            layer_idx = feature['layer_index']
            feature_groups[model][layer_type][layer_idx].append({
                'token': token,
                'feature': feature
            })
    
    for model in tqdm(feature_groups.keys(), desc="Processing models"):
        for layer_type in tqdm(feature_groups[model].keys(), desc=f"{model} layer"):
            for layer_idx in tqdm(feature_groups[model][layer_type].keys(), desc=f"layer {layer_type}"):
                try:
                    layer_features = get_sae_features_by_layer(model, layer_type, layer_idx)
                    
                    for item in feature_groups[model][layer_type][layer_idx]:
                        feature = item['feature']
                        feature_id = feature['feature_id']
                        
                        if feature_id in layer_features:
                            feature_data = layer_features[feature_id]
                            top_activations = []
                            
                            if feature_data.get('activations'):
                                for activation in feature_data['activations'][:10]:
                                    if activation.get('tokens') and activation.get('values'):
                                        top_activations.append({
                                            'tokens': activation['tokens'],
                                            'values': activation['values']
                                        })
                            
                            feature['activations'] = top_activations
                        else:
                            print(f"warning: feature {feature_id} activation not found in {model} {layer_type}{layer_idx}")
                            feature['activations'] = []
                            
                except Exception as e:
                    print(f"error when processing {model} {layer_type}{layer_idx}: {e}")
                    continue
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

class HookType(Enum):
    ATT = "att"
    MLP = "mlp"
    RES_MID = "res_mid"
    RES_POST = "res_post"
    RES = "res"

# also adapt to gemma
def get_llama_gradient_attack_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    token_str_list: List[str],
    token_pos: int,
    block_idx: int,
    hook_type: Union[str, HookType],
    show_print: bool = True
) -> torch.Tensor:
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        device = model.device

        if isinstance(hook_type, str):
            if hook_type == "res":
                hook_type = "res_post"
            try:
                hook_type = HookType(hook_type)
            except ValueError:
                raise ValueError(f"Invalid hook type: {hook_type}")

        all_token_ids = []
        target_pos = 0
        
        for i, token_str in enumerate(token_str_list):
            curr_ids = tokenizer(token_str, add_special_tokens=False)['input_ids']
            if i < token_pos:
                target_pos += len(curr_ids)
            all_token_ids.extend(curr_ids)
        
        input_ids = torch.tensor([all_token_ids], device=device)
        
        target_token = token_str_list[token_pos]
        target_token_ids = tokenizer(target_token, add_special_tokens=False)['input_ids']
        all_tokens = tokenizer.convert_ids_to_tokens(all_token_ids)
        
        if show_print:
            print(f"target token str: '{target_token}'")
            print(f"target token ID: {target_token_ids}")
            print(f"target token position: {target_pos}")
            print(f"tokens after tokenizer: {all_tokens}")
            print(f"target token: {all_tokens[target_pos]}")

        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.detach().clone()
        inputs_embeds.requires_grad = True

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activation[name.value] = output[0]
                else:
                    activation[name.value] = output
            return hook

        # register and save handles
        handles = []
        if hook_type == HookType.ATT:
            handles.append(
                model.model.layers[block_idx].self_attn.register_forward_hook(
                    get_activation(hook_type)
                )
            )
        elif hook_type == HookType.MLP:
            handles.append(
                model.model.layers[block_idx].mlp.register_forward_hook(
                    get_activation(hook_type)
                )
            )
        elif hook_type == HookType.RES_MID:
            handles.append(
                model.model.layers[block_idx].self_attn.register_forward_hook(
                    get_activation(HookType.ATT)
                )
            )
            handles.append(
                model.model.layers[block_idx].register_forward_hook(
                    lambda m, i, o: activation.update({
                        hook_type.value: activation[HookType.ATT.value] + i[0]
                    })
                )
            )
        elif hook_type == HookType.RES_POST:
            handles.append(
                model.model.layers[block_idx].mlp.register_forward_hook(
                    get_activation(HookType.MLP)
                )
            )
            handles.append(
                model.model.layers[block_idx].register_forward_hook(
                    lambda m, i, o: activation.update({
                        hook_type.value: activation[HookType.MLP.value] + i[0]
                    })
                )
            )

        try:
            # forward prop
            outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            
            # 
            target_output = activation[hook_type.value]
            
            # check type
            if not isinstance(target_output, torch.Tensor):
                raise ValueError(f"wrong hook output type: {type(target_output)}")
            
            # make gradient mask
            grad_output = torch.zeros_like(target_output)
            grad_output[0, target_pos] = 1.0
            
            # back prop
            target_output.backward(gradient=grad_output, retain_graph=True)
            
            # get and normalize gradient
            grad = inputs_embeds.grad[0, target_pos].clone()
            normalized_grad = grad / grad.norm()
            
            return normalized_grad.detach().cpu()

        finally:
            for handle in handles:
                handle.remove()
            
            # clean mediate variables
            del outputs, target_output, grad_output, activation
            if 'grad' in locals():
                del grad
            torch.cuda.empty_cache()
            gc.collect()

    finally:
        torch.cuda.empty_cache()
        gc.collect()

def get_next_token_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    attack_vector: torch.Tensor,
    hook_pos: Tuple[int, str],
    scale: float,
    show_print: bool = False,
    save_path = None
) -> Tuple[List[Tuple], List[Tuple]]:
    """Compare the output change before and after the attack.
    
    Args:
        model: language model
        tokenizer: tokenizer of the model
        text: input text
        attack_vector: vector to be added to the activation space
        hook_pos: (block_idx, hook_type), hook position
        scale: scale of the attack vector
        show_print: whether to print the top tokens
    
    Returns:
        Tuple[List[Tuple], List[Tuple]]: 
            (original top10 tokens:(id, str, prob), 
             top10 tokens after attack:(id, str, prob))
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        device = model.device
        if scale != 0:
            block_idx, hook_type = hook_pos
        
        if show_print:
            print(f"\n{'='*20} Next Token Inference {'='*20}")
            print(f"Input: {text}")
            print(f"Hook pos: Block {block_idx}, Type {hook_type}")
            print(f"Attack scale: {scale}")
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        
        def get_top_tokens(logits: torch.Tensor) -> List[Tuple]:
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_probs, top_ids = torch.topk(probs, k=10)
            results = []
            for token_id, prob in zip(top_ids, top_probs):
                token_str = tokenizer.decode(token_id)
                results.append((token_id.item(), token_str, prob.item()))
            return results
            
        def print_top_tokens(logits: torch.Tensor, prefix: str):
            if not show_print:
                return
            print(f"\n----- {prefix} -----")
            probs = torch.softmax(logits[0, -1], dim=-1)
            top_probs, top_ids = torch.topk(probs, k=10)
            for token_id, prob in zip(top_ids, top_probs):
                token_str = tokenizer.decode(token_id)
                print(f"{token_str:<20} {prob:.4f}")
        
        def attack_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states[0, :] += scale * attack_vector.to(hidden_states.device)
            return output
        
        original_results = None
        with torch.no_grad():
            outputs = model(input_ids)
            original_results = get_top_tokens(outputs.logits)
            if show_print:
                print_top_tokens(outputs.logits, "Original Prediction")
        
        handle = None
        if scale != 0:
            if hook_type == "att":
                handle = model.model.layers[block_idx].self_attn.register_forward_hook(attack_hook)
            elif hook_type == "mlp":
                handle = model.model.layers[block_idx].mlp.register_forward_hook(attack_hook)
            elif hook_type in ["res", "res_post"]:
                handle = model.model.layers[block_idx].register_forward_hook(attack_hook)
            elif hook_type == "res_mid":
                handle = model.model.layers[block_idx].register_forward_hook(
                    lambda m, i, o: attack_hook(m, i, (o[0], o[1])) if isinstance(o, tuple) else attack_hook(m, i, o)
                )
        
        attack_results = None
        try:
            with torch.no_grad():
                outputs = model(input_ids)
                attack_results = get_top_tokens(outputs.logits)
                if show_print:
                    print_top_tokens(outputs.logits, "Post-Attack Prediction")
        finally:
            if handle:
                handle.remove()

        if save_path:
            token_set = {}  # Using dict to maintain token order
            token_strs = []
            orig_probs = []
            attack_probs = []

            # Get all logits
            with torch.no_grad():
                outputs = model(input_ids)
                orig_all_probs = torch.softmax(outputs.logits[0, -1], dim=-1)
        
                if handle:
                    handle = model.model.layers[block_idx].register_forward_hook(attack_hook)
                outputs_attack = model(input_ids)
                attack_all_probs = torch.softmax(outputs_attack.logits[0, -1], dim=-1)
                if handle:
                    handle.remove()

            # Add original top tokens
            for token_id, token_str, _ in original_results:
                if token_str not in token_set:
                    token_set[token_str] = True
                    token_strs.append(token_str)
                    orig_probs.append(float(orig_all_probs[token_id]))
                    attack_probs.append(float(attack_all_probs[token_id]))

            # Add attack top tokens
            for token_id, token_str, _ in attack_results:
                if token_str not in token_set:
                    token_set[token_str] = True
                    token_strs.append(token_str)
                    orig_probs.append(float(orig_all_probs[token_id]))
                    attack_probs.append(float(attack_all_probs[token_id]))

            # Save results with new format
            save_data = {
                "all_tokens": token_strs,
                "all_tokens_probs_before_attack": orig_probs,
                "all_tokens_probs_after_attack": attack_probs,
                "raw_top10_tokens": [t[1] for t in original_results],
                "raw_top10_probs": [t[2] for t in original_results],
                "attacked_top10_tokens": [t[1] for t in attack_results],
                "attacked_top10_probs": [t[2] for t in attack_results]
            }
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
        return original_results, attack_results
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def analyze_output_semantic(
    token_strs: List[str],
    token_probs: Union[torch.Tensor, List[float]],
    condition_fn: Callable[[str], bool]
) -> float:
    """Analyze the output semantic of the model.
    
    Args:
        token_strs: token strings of model's output
        token_probs: tokens' probabilities
        condition_fn: function to check if to take the token into consideration
        
    Returns:
        float: the sum of probabilities of tokens that satisfy the condition
    """
    if isinstance(token_probs, torch.Tensor):
        probs = token_probs.detach().cpu().numpy()
    else:
        probs = token_probs
        
    total_prob = 0.0
    
    for token_str, prob in zip(token_strs, probs):
        if condition_fn(token_str):
            total_prob += float(prob)
            
    return total_prob

def test_feature_attack(
    token: str,
    feature: dict,
    llama_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pythia_model: HookedTransformer,
    token_type_dict: dict,
    num_activations: int = 5,
    prompt: str = ''
) -> dict:
    """Test attack effects for a single feature
    
    Args:
        token: Target token to test
        feature: Feature information dictionary
        llama_model: LLaMA model
        tokenizer: LLaMA tokenizer
        pythia_model: Pythia model (for location check)
        token_type_dict: Token type dictionary
        num_activations: Number of activation texts to test
        prompt: Test prompt text
    
    Returns:
        dict: Best attack results
    """
    print(f"\n=== Testing Feature ===")
    print(f"Token: {token}")
    print(f"Model: {feature['model']}")
    print(f"Feature ID: {feature['feature_id']}")
    print(f"Layer: {feature['layer_type']}{feature['layer_index']}")
    
    scales = [-20, -17, -14, -12, -10, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, 
              -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5]
    scales = scales + [-x for x in scales]
    scales.sort()
    
    best_result = {
        'biggest_sim_improve': 0,
        'best_activation': None,
        'best_hook_pos': None,
        'best_scale': None,
        'best_attack_vec': None,
        'orig_tokens': None,
        'attack_tokens': None
    }
    
    for idx, activation in enumerate(feature['activations'][:num_activations]):
        if not activation.get('tokens'):
            continue
            
        print(f"\n--- Testing Activation Text #{idx+1} ---")
        tokens = activation['tokens']
        max_val = max(activation['values'])
        max_val_id = activation['values'].index(max_val)
        interfered_text = " ".join(tokens)
        print(f"Activation text: {interfered_text}")
        
        # Test each layer
        for layer_idx in tqdm(range(16), desc="Testing layers"):
            # Get attack vector for current layer
            try:
                attack_vec = get_llama_gradient_attack_vector(
                    llama_model, tokenizer,
                    token_str_list=tokens,
                    token_pos=max_val_id,
                    block_idx=layer_idx,
                    hook_type="res",
                    show_print=False
                )
            except Exception as e:
                print(f"Error getting attack vector for layer {layer_idx}: {e}")
                continue
                
            # Test different scales
            for scale in scales:
                try:
                    orig_tokens, attack_tokens = get_next_token_probs(
                        llama_model, tokenizer,
                        text=prompt,
                        attack_vector=attack_vec,
                        hook_pos=(layer_idx, "res"),
                        scale=scale,
                        show_print=False
                    )
                    
                    # Calculate location-related probabilities
                    orig_location_prob = analyze_output_semantic(
                        [t[1] for t in orig_tokens],
                        [t[2] for t in orig_tokens],
                        lambda x: is_location(x, pythia_model, token_type_dict)
                    )
                    
                    attack_location_prob = analyze_output_semantic(
                        [t[1] for t in attack_tokens],
                        [t[2] for t in attack_tokens],
                        lambda x: is_location(x, pythia_model, token_type_dict)
                    )
                    
                    sim_improve = attack_location_prob - orig_location_prob
                    
                    # Update best results if better
                    if sim_improve > best_result['biggest_sim_improve']:
                        best_result.update({
                            'biggest_sim_improve': float(sim_improve),
                            'best_activation': interfered_text,
                            'best_hook_pos': (layer_idx, 'res'),
                            'best_scale': scale,
                            'best_attack_vec': attack_vec,
                            'orig_tokens': orig_tokens,
                            'attack_tokens': attack_tokens
                        })
                        
                except Exception as e:
                    continue
                    
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n=== Best Attack Results ===")
    print(f"Improvement: {best_result['biggest_sim_improve']:.4f}")
    print(f"Best activation text: {best_result['best_activation']}")
    print(f"Best attack layer: {best_result['best_hook_pos']}")
    print(f"Best scale: {best_result['best_scale']}")
    
    print("\n=== Attack Effect Comparison ===")
    print("Original top10 tokens:")
    for token_id, token_str, prob in best_result['orig_tokens']:
        print(f"{token_str:<20} {prob:.4f}")
        
    print("\nAttacked top10 tokens:")
    for token_id, token_str, prob in best_result['attack_tokens']:
        print(f"{token_str:<20} {prob:.4f}")
        
    return best_result

def reproduce_saved_attack(
    result_file: str,
    token: str,
    feature_idx: int,
    llama_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = "I would like to recommend you to spend holidays in",
    save_path = None
) -> None:
    """Reproduce attack result from saved attack results file
    
    Args:
        result_file: Path to the attack results file (e.g. llama_location_attack_results.json)
        token: Target token to reproduce
        feature_idx: Index of the feature for this token (0-based)
        llama_model: LLaMA model
        tokenizer: LLaMA tokenizer
        prompt: Test prompt text
    """
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading file {result_file}: {e}")
        return
        
    if token not in results:
        print(f"Token '{token}' not found in results")
        return
        
    if not results[token]:
        print(f"No successful attacks found for token '{token}'")
        return
        
    if feature_idx >= len(results[token]):
        print(f"Feature index {feature_idx} exceeds available features ({len(results[token])})")
        return
        
    attack_info = results[token][feature_idx]
    
    print("\n=== Attack Information ===")
    print(f"Token: {token}")
    print(f"Source Model: {attack_info['model']}")
    print(f"Feature ID: {attack_info['feature_id']}")
    print(f"Original Layer: {attack_info['layer_type']}{attack_info['layer_index']}")
    print(f"Attack Improvement: {attack_info['biggest_sim_improve']:.4f}")
    print(f"Activation Text: {attack_info['best_interfered_text']}")
    print(f"Attack Layer: {attack_info['best_hook_pos']}")
    print(f"Attack Scale: {attack_info['best_scale']}")
    
    attack_vec = torch.tensor(attack_info['best_attack_vec']).to(llama_model.device)
    
    print("\n=== Reproducing Attack Effect ===")
    get_next_token_probs(
        llama_model,
        tokenizer,
        text=prompt,
        attack_vector=attack_vec,
        hook_pos=attack_info['best_hook_pos'],
        scale=attack_info['best_scale'],
        show_print=True,
        save_path=save_path
    )
    
    torch.cuda.empty_cache()
    gc.collect()

def print_model_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    injection_text: str = None,
    top_k: int = 10,
    save_path: str = None
) -> Tuple[List[Tuple[int, str, float]], List[Tuple[int, str, float]]]:
    """Print model's inference results for given text with optional prompt injection
    
    Args:
        model: language model
        tokenizer: tokenizer
        text: input text
        injection_text: text to inject before the input text
        top_k: number of top tokens to show (default: 10)
        save_path: path to save the token probability changes
        
    Returns:
        Tuple[List[Tuple], List[Tuple]]: 
            (original results:(token_id,str,prob), 
             results after injection:(token_id,str,prob))
    """
    # Original inference
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    
    print(f"\nOriginal input text: {text}")
    
    original_results = []
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_ids = torch.topk(probs, k=top_k)
        
        print(f"\nTop {top_k} tokens:")
        print(f"{'Token':<20} {'Probability':<10}")
        print("-" * 30)
        
        for token_id, prob in zip(top_ids, top_probs):
            token_str = tokenizer.decode(token_id)
            print(f"{token_str:<20} {prob:.4f}")
            original_results.append((token_id.item(), token_str, prob.item()))
    
    injection_results = []
    if injection_text:
        # Injection inference
        full_text = injection_text + text
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        
        print(f"\nInjected input text: {full_text}")
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_ids = torch.topk(probs, k=top_k)
            
            print(f"\nTop {top_k} tokens after injection:")
            print(f"{'Token':<20} {'Probability':<10}")
            print("-" * 30)
            
            for token_id, prob in zip(top_ids, top_probs):
                token_str = tokenizer.decode(token_id)
                print(f"{token_str:<20} {prob:.4f}")
                injection_results.append((token_id.item(), token_str, prob.item()))

    if save_path:
        token_set = {}  # Using dict to maintain token order
        token_strs = []
        orig_probs = []
        inject_probs = []
    
        # Get all logits
        with torch.no_grad():
            outputs = model(tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device))
            orig_all_probs = torch.softmax(outputs.logits[0, -1], dim=-1)
        
            outputs = model(tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device))
            inject_all_probs = torch.softmax(outputs.logits[0, -1], dim=-1)
    
        # Add original top tokens
        for token_id, token_str, _ in original_results:
            if token_str not in token_set:
                token_set[token_str] = True
                token_strs.append(token_str)
                orig_probs.append(float(orig_all_probs[token_id]))
                inject_probs.append(float(inject_all_probs[token_id]))
    
        # Add injection top tokens
        for token_id, token_str, _ in injection_results:
            if token_str not in token_set:
                token_set[token_str] = True
                token_strs.append(token_str)
                orig_probs.append(float(orig_all_probs[token_id]))
                inject_probs.append(float(inject_all_probs[token_id]))
    
        # Save results with new format
        save_data = {
            "all_tokens": token_strs,
            "all_tokens_probs_before_attack": orig_probs,
            "all_tokens_probs_after_attack": inject_probs,
            "raw_top10_tokens": [t[1] for t in original_results],
            "raw_top10_probs": [t[2] for t in original_results],
            "attacked_top10_tokens": [t[1] for t in injection_results],
            "attacked_top10_probs": [t[2] for t in injection_results]
        }
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

def print_model_inference_old(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    top_k: int = 10
) -> List[Tuple[int, str, float]]:
    """Print model's inference results for given text
    
    Args:
        model: language model
        tokenizer: tokenizer
        text: input text
        top_k: number of top tokens to show (default: 10)
        
    Returns:
        List[Tuple[int, str, float]]: list of (token_id, token_str, probability)
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    
    print(f"Input text: {text}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_ids = torch.topk(probs, k=top_k)
        
        print(f"\nTop {top_k} tokens:")
        print(f"{'Token':<20} {'Probability':<10}")
        print("-" * 30)
        
        results = []
        for token_id, prob in zip(top_ids, top_probs):
            token_str = tokenizer.decode(token_id)
            print(f"{token_str:<20} {prob:.4f}")
            results.append((token_id.item(), token_str, prob.item()))
            

def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    attack_vector: torch.Tensor,
    hook_pos: Tuple[int, str],
    scale: float,
    n: int
) -> Tuple[List[str], List[str]]:
    """generate text with a steering vector
    
    Args:
        model: large language model
        tokenizer: tokenizer
        text: input text
        attack_vector: attack vector
        hook_pos: (block_idx, hook_type) tuple, specify attack position
        scale: attack vector scale
        n: number of tokens to generate
    
    Returns:
        Tuple[List[str], List[str]]: original and attack generated tokens
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        device = model.device
        block_idx, hook_type = hook_pos
        
        print(f"\n{'='*20} Inference {'='*20}")
        print(f"Input text: {text}")
        print(f"Tokens to generate: {n}")
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        
        def attack_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            hidden_states[0, :] += scale * attack_vector.to(hidden_states.device)
            return output if isinstance(output, tuple) else hidden_states
        
        print("\n----- Original -----")
        original_tokens = []
        with torch.no_grad():
            current_ids = input_ids.clone()
            for _ in range(n):
                outputs = model(current_ids)
                next_token_id = outputs.logits[0, -1].argmax()
                token = tokenizer.decode(next_token_id)
                original_tokens.append(token)
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                del outputs
                torch.cuda.empty_cache()
        
        print("Original Output:", " ".join(original_tokens))
        del current_ids
        torch.cuda.empty_cache()
        
        print("\n----- Post-Attack -----")
        if hook_type == "att":
            hook = model.model.layers[block_idx].self_attn.register_forward_hook(attack_hook)
        elif hook_type == "mlp":
            hook = model.model.layers[block_idx].mlp.register_forward_hook(attack_hook)
        elif hook_type in ["res", "res_post"]:
            hook = model.model.layers[block_idx].register_forward_hook(attack_hook)
        elif hook_type == "res_mid":
            hook = model.model.layers[block_idx].self_attn.register_forward_hook(attack_hook)
        
        try:
            attack_tokens = []
            with torch.no_grad():
                current_ids = input_ids.clone()
                for _ in range(n):
                    outputs = model(current_ids)
                    next_token_id = outputs.logits[0, -1].argmax()
                    token = tokenizer.decode(next_token_id)
                    attack_tokens.append(token)
                    current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    del outputs
                    torch.cuda.empty_cache()
            
            print("Post-Attack Result:", " ".join(attack_tokens))
            return original_tokens, attack_tokens
        
        finally:
            hook.remove()
            if 'current_ids' in locals():
                del current_ids
            torch.cuda.empty_cache()
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def is_token_type(
    token: str,
    token_type: str,
    model: HookedTransformer,
    token_type_dict: Dict
) -> bool:
    """Check if a token is of a specific type.
    
    Args:
        token: token string to be checked
        token_type: type name ('person', 'location', etc.)
        model: HookedTransformer, used to get token_ids
        token_type_dict: token type dict
        
    Returns:
        bool: whether the token is of the specified type
    """
    # get token_id
    token_ids = model.to_tokens(token, prepend_bos=False)
    if token_ids.numel() != 1:
        return False
        
    token_id_str = str(token_ids.item())
    
    # check token type
    if token_id_str in token_type_dict:
        for _, info in token_type_dict[token_id_str].items():
            if info['type'] == token_type:
                return True
    
    return False

def is_location(
    token: str,
    model: HookedTransformer,
    token_type_dict: Dict
) -> bool:
    """ check whether a token is a location
    
    Args:
        token: token string to be checked
        model: model for tokenization
        token_type_dict: token type dict
        
    Returns:
        bool: check whether the token is a location
    """
    return is_token_type(token, 'location', model, token_type_dict)

def is_person(
    token: str,
    model: HookedTransformer,
    token_type_dict: Dict
) -> bool:
    """Check whether a token is a person name
    
    Args:
        token: the token string to be checked
        model: model for tokenization
        token_type_dict: token type dict
        
    Returns:
        bool: whether the token is a person name
    """
    return is_token_type(token, 'person', model, token_type_dict)

def is_in_list(token: str, token_list: List[str]) -> bool:
    """check whether a token is in a list
    
    Args:
        token: the token string to be checked
        token_list: token list
        
    Returns:
        bool: whether the token is in the list
    """
    return token in token_list