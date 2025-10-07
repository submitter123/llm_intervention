import torch
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_with_raw_model(model, text):
    input_ids = model.to_tokens(text, prepend_bos=False)

    with torch.no_grad():
        last_token_logits = model(input_ids)[0, -1, :]

    probs = torch.softmax(last_token_logits, dim=-1)

    # sort the model's output
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    return sorted_indices, sorted_probs

def run_with_raw_auto_model(model, tokenizer, text, device='cuda'):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        last_token_logits = outputs.logits[0, -1, :]
    
    probs = torch.softmax(last_token_logits, dim=-1)
    
    # sort the model's output
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    return sorted_indices, sorted_probs

def parse_hook_position(mixed_str):
    """
    Get the hook position for a given layer in the model.

    params:
        mixed_str: a string representing the layer (e.g., "1-att")

    return:
        hook position string (e.g., "blocks.1.hook_attn_out")
    """

    hook_names = {
        'att': 'hook_attn_out',
        'mlp': 'hook_mlp_out',
        'res_mid': 'hook_resid_mid',
        'res_post': 'hook_resid_post',
        'res': 'hook_resid_post'
    }

    layer_index = int(mixed_str.split('-')[0])
    layer_type = mixed_str.split('-')[1]
    layer_type = layer_type.removesuffix('-sm').removesuffix('_32k')
    hook_name = hook_names[layer_type]

    return f'blocks.{layer_index}.{hook_name}'

def run_with_hooked_model(model, text, hook_pos, direction, scale):
    def scale_direction(dir, scale):
        def fn_hook(act, hook):
            act[:, :, :] += scale * dir
            return act
        return fn_hook

    # normalize the direction vector
    norm_direction = direction / torch.norm(direction)
    hook_direction = norm_direction
    input_ids = model.to_tokens(text, prepend_bos=False)
    with torch.no_grad():
        last_token_logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[
                (hook_pos, scale_direction(hook_direction, scale))
            ]
        )[0, -1, :]

    probs = torch.softmax(last_token_logits, dim=-1)

    # sort the model's output
    sorted_probs, sorted_tokens = torch.sort(probs, descending=True)

    return sorted_tokens, sorted_probs

def run_with_hooked_auto_model(model, tokenizer, text, hook_pos, direction, scale, device='cuda'):
    """
    Run inference with AutoModelForCausalLM using hooks for intervention
    
    params:
        model: AutoModelForCausalLM model
        tokenizer: tokenizer of the model
        text: input text string
        hook_pos: hook position, ('att', 'res_mid', 'mlp', 'res_post', 'res')
        direction: intervention direction vector (torch.Tensor)
        scale: intervention scale (float)
        device: device to run on (str)
    
    return:
        sorted_tokens: sorted token indices(from high probs -> low)
        sorted_probs: sorted probabilities
    """
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    norm_direction = direction / torch.norm(direction)
    
    def get_hook_module_and_function(model, hook_pos, layer_idx):
        """Get target module path and hook function based on hook_pos and model type"""
        model_type = model.config.model_type
        
        # gpt_neox for pythia-70m-deduped
        if hook_pos == 'att':
            # Attention layer hook
            if model_type == "gpt_neox":  # Pythia
                target_path = f"gpt_neox.layers.{layer_idx}.attention"
            elif model_type == "gpt2":  # GPT2
                target_path = f"transformer.h.{layer_idx}.attn"
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    modified_output = output[0] + scale * norm_direction
                    return (modified_output,) + output[1:]
                else:
                    return output + scale * norm_direction
                    
        elif hook_pos == 'mlp':
            # MLP layer hook
            if model_type == "gpt_neox":  # Pythia
                target_path = f"gpt_neox.layers.{layer_idx}.mlp"
            elif model_type == "gpt2":  # GPT2
                target_path = f"transformer.h.{layer_idx}.mlp"
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    modified_output = output[0] + scale * norm_direction
                    return (modified_output,) + output[1:]
                else:
                    return output + scale * norm_direction
                    
        elif hook_pos == 'res_mid':
            # res_mid hook
            if model_type == "gpt2":
                target_path = f"transformer.h.{layer_idx}.attn"
                
                def hook_fn(module, input, output):
                    # Simulate intervention after attention output + residual connection
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        modified_hidden = hidden_states + scale * norm_direction
                        return (modified_hidden,) + output[1:]
                    return output + scale * norm_direction
            else:
                raise ValueError(f"Model type {model_type} does not support res_mid hook")
                
        elif hook_pos in ['res_post', 'res']:
            # res_post hook - intervene at the end of entire transformer layer
            if model_type == "gpt_neox":  # Pythia
                target_path = f"gpt_neox.layers.{layer_idx}"
            elif model_type == "gpt2":  # GPT2
                target_path = f"transformer.h.{layer_idx}"
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            def hook_fn(module, input, output):
                # Output of entire transformer block
                if isinstance(output, tuple):
                    modified_output = output[0] + scale * norm_direction
                    return (modified_output,) + output[1:]
                else:
                    return output + scale * norm_direction
        else:
            raise ValueError(f"Unsupported hook position: {hook_pos}")
        
        return hook_fn, target_path
    
    # Parse layer index from hook_pos if it's in format like "6-att"
    # If hook_pos is just the layer type, we need to extract layer_idx separately
    if '-' in hook_pos:
        layer_idx = int(hook_pos.split('-')[0])
        hook_type = hook_pos.split('-')[1]
    else:
        # Assume hook_pos is just the type and layer_idx is provided separately
        # For this function, we'll expect the format "layer_idx-hook_type"
        raise ValueError("hook_pos should be in format 'layer_idx-hook_type', e.g., '6-att'")
    
    # Get hook function and target module path
    hook_fn, target_path = get_hook_module_and_function(model, hook_type, layer_idx)
    
    # Get target module
    target_module = model
    for attr in target_path.split('.'):
        target_module = getattr(target_module, attr)
    
    # Register hook
    handle = target_module.register_forward_hook(hook_fn)
    
    try:
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids)
            last_token_logits = outputs.logits[0, -1, :]
    finally:
        # Ensure hook is removed
        handle.remove()
    
    # Calculate probabilities and sort
    probs = torch.softmax(last_token_logits, dim=-1)
    sorted_probs, sorted_tokens = torch.sort(probs, descending=True)
    
    return sorted_tokens, sorted_probs

def get_gradient_intervention_vector_for_hooked_model(
    model,
    token_strs: list,
    token_str_position: int,
    layer_idx: int,
    layer_type: str,
    r_act_ratio: float = 0.5,
    activation_values: list = None
) -> torch.Tensor:
    """
    Get the gradient vector for a specific token position as intervention vector for HookedTransformer.
    Extract gradients from multiple high-activation positions based on provided activation_values.
    
    Args:
        model: HookedTransformer model
        token_strs: List of token strings that make up the text
        token_str_position: Target token string position in the list
        layer_idx: Target layer index
        layer_type: Layer type ('att', 'mlp', 'res_mid', 'res_post', 'res')
        r_act_ratio: Activation ratio threshold (0-1.0), extract gradients from positions 
                    with activation >= r_act_ratio * max_activation
        activation_values: List of activation values corresponding to each token position
        
    Returns:
        torch.Tensor: Combined and normalized gradient vector on CPU
    """
    
    # Join token strings to create the full text
    text = ''.join(token_strs)
    
    # Convert to tokens using the model's tokenizer
    tokens = model.to_tokens(text, prepend_bos=False)
    
    # Build mapping from token_str to actual token positions
    all_token_ids = []
    str_to_token_map = {}
    
    for str_idx, token_str in enumerate(token_strs):
        token_ids = model.to_tokens(token_str, prepend_bos=False)[0].tolist()
        start_pos = len(all_token_ids)
        all_token_ids.extend(token_ids)
        end_pos = len(all_token_ids)
        str_to_token_map[str_idx] = list(range(start_pos, end_pos))
    
    # Get target token positions and validate
    target_token_positions = str_to_token_map[token_str_position]
    primary_token_position = target_token_positions[0]  # Use first token if multiple
    
    token_len = tokens.shape[1]
    if primary_token_position >= token_len:
        raise ValueError(f"Token position {primary_token_position} exceeds actual length {token_len}")
    
    # Determine high-activation positions based on activation_values
    high_activation_positions = []
    
    if activation_values is not None and len(activation_values) > 0:
        # Use provided activation values to determine positions
        max_activation = max(activation_values)
        activation_threshold = r_act_ratio * max_activation
        
        # Map activation_values (which correspond to token_strs) to actual token positions
        for str_idx, activation_val in enumerate(activation_values):
            if activation_val >= activation_threshold:
                # Get all token positions for this token_str
                token_positions = str_to_token_map.get(str_idx, [])
                high_activation_positions.extend(token_positions)

    else:
        # Fallback to primary position if no activation_values provided
        high_activation_positions = [primary_token_position]
    
    # Ensure positions are within bounds and remove duplicates
    high_activation_positions = list(set([pos for pos in high_activation_positions if 0 <= pos < token_len]))
    
    if not high_activation_positions:
        high_activation_positions = [primary_token_position]
        print(f"No valid high-activation positions, falling back to primary position: {primary_token_position}")
    
    # Construct hook name based on layer type
    hook_names = {
        'att': 'hook_attn_out',
        'mlp': 'hook_mlp_out',
        'res_mid': 'hook_resid_mid',
        'res_post': 'hook_resid_post',
        'res': 'hook_resid_post'  # alias for res_post
    }
    
    if layer_type not in hook_names:
        raise ValueError(f"Unsupported layer_type: {layer_type}")
    
    hook_name = f'blocks.{layer_idx}.{hook_names[layer_type]}'
    
    # Extract gradients from all high-activation positions
    combined_gradient = None
    
    for pos in high_activation_positions:
        try:
            # Set up hooks for gradient computation
            def embed_hook(act, hook):
                act_detached = act.detach().clone()
                act_detached.requires_grad = True
                hook.ctx["input_embeds"] = act_detached
                return act_detached
            
            saved_layer_activations = {}
            def save_layer_hook(act, hook):
                saved_layer_activations["layer_output"] = act
                return act
            
            hooks = [
                ("hook_embed", embed_hook),
                (hook_name, save_layer_hook)
            ]
            
            # Forward pass with hooks
            model.run_with_hooks(tokens, fwd_hooks=hooks)
            
            layer_output = saved_layer_activations["layer_output"]
            grad_output = torch.zeros_like(layer_output)
            grad_output[0, pos, :] = 1.0
            
            layer_output.backward(gradient=grad_output, retain_graph=False)
            
            input_embeds = model.hook_dict["hook_embed"].ctx["input_embeds"]
            if input_embeds.grad is None:
                print(f"Warning: No gradient computed for position {pos}")
                continue
            
            position_grad = input_embeds.grad[0, pos, :]
            
            # Check for valid gradient
            if torch.isnan(position_grad).any() or torch.isinf(position_grad).any():
                print(f"Warning: Invalid gradient at position {pos}")
                continue
            
            grad_norm = position_grad.norm()
            if grad_norm < 1e-8:
                print(f"Warning: Very small gradient norm ({grad_norm:.2e}) at position {pos}")
                continue
            
            # Accumulate gradients
            if combined_gradient is None:
                combined_gradient = position_grad.clone()
            else:
                combined_gradient += position_grad
            
        except Exception as e:
            print(f"Warning: Failed to extract gradient from position {pos}: {e}")
            continue
    
    # Normalize the combined gradient
    if combined_gradient is None:
        print("Warning: No valid gradients extracted, returning zero vector")
        # Return zero vector with same dimension as embedding
        return torch.zeros(model.cfg.d_model, device='cpu')
    
    # Normalize the combined gradient
    final_norm = combined_gradient.norm()
    if final_norm < 1e-8:
        print(f"Warning: Combined gradient norm is very small ({final_norm:.2e})")
        normalized_grad = torch.zeros_like(combined_gradient)
    else:
        normalized_grad = combined_gradient / final_norm
    
    return normalized_grad.detach().cpu()

def get_gradient_intervention_vector_for_auto_model(
    model,
    tokenizer,
    token_strs: list,
    token_str_position: int,
    layer_idx: int,
    layer_type: str,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Get the gradient vector for a specific token position as intervention vector for AutoModel
    Uses the first token when token_str is split into multiple tokens
    
    Args:
        model: AutoModelForCausalLM model
        tokenizer: tokenizer for the model
        token_strs: List of token strings that make up the text
        token_str_position: Target token string position in the list
        layer_idx: Target layer index
        layer_type: Layer type ('att', 'mlp', 'res_mid', 'res_post', 'res')
        device: device to run on
        
    Returns:
        torch.Tensor: Normalized gradient vector on CPU
    """
    
    # Step 1: Build mapping from token_str to actual token positions
    all_token_ids = []
    str_to_token_map = {}
    
    for str_idx, token_str in enumerate(token_strs):
        token_ids = tokenizer(token_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        start_pos = len(all_token_ids)
        all_token_ids.extend(token_ids.tolist())
        end_pos = len(all_token_ids)
        str_to_token_map[str_idx] = list(range(start_pos, end_pos))
    
    # Step 2: Get target token positions and validate
    target_token_positions = str_to_token_map[token_str_position]
    actual_token_position = target_token_positions[0]
    
    # Step 3: Create input tensors
    input_ids = torch.tensor([all_token_ids], device=device)
    token_len = input_ids.shape[1]
    
    if actual_token_position >= token_len:
        raise ValueError(f"Token position {actual_token_position} exceeds actual length {token_len}")
    
    # Step 4: Get input embeddings and enable gradients
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(input_ids)
    input_embeds.requires_grad_(True)
    input_embeds.retain_grad()
    
    # Step 5: Set up hook to save activations
    saved_activations = {}
    
    def save_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            if not isinstance(activation, torch.Tensor):
                raise TypeError(f"Expected tensor activation, got {type(activation)}")
            
            saved_activations[name] = activation
            return output
        return hook_fn
    
    # Step 6: Get target module
    model_type = model.config.model_type
    target_module = None
    hook_name = f"layer_{layer_idx}_{layer_type}"
    
    if model_type == "gpt_neox":        
        if layer_type == "att":
            target_module = model.gpt_neox.layers[layer_idx].attention
        elif layer_type == "mlp":
            target_module = model.gpt_neox.layers[layer_idx].mlp
        elif layer_type == "res_mid":
            target_module = model.gpt_neox.layers[layer_idx]  # Hook at layer output before final norm
        elif layer_type in ["res_post", "res"]:
            target_module = model.gpt_neox.layers[layer_idx]
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")
            
    elif model_type == "gpt2":        
        if layer_type == "att":
            target_module = model.transformer.h[layer_idx].attn
        elif layer_type == "mlp":
            target_module = model.transformer.h[layer_idx].mlp
        elif layer_type == "res_mid":
            target_module = model.transformer.h[layer_idx]
        elif layer_type in ["res_post", "res"]:
            target_module = model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Step 7: Register hook and run forward pass
    handle = target_module.register_forward_hook(save_hook(hook_name))
    
    try:
        outputs = model(inputs_embeds=input_embeds)
        last_token_logits = outputs.logits[0, -1, :]
        
        # Step 8: Calculate gradients using backward method (old version style)
        if hook_name not in saved_activations:
            raise RuntimeError(f"Failed to capture activations for {hook_name}")
        
        layer_activations = saved_activations[hook_name]
        
        if not isinstance(layer_activations, torch.Tensor):
            raise TypeError(f"Expected tensor activation, got {type(layer_activations)}")
        
        if len(layer_activations.shape) < 3:
            raise ValueError(f"Expected 3D activation tensor (batch, seq_len, hidden), got shape {layer_activations.shape}")
        
        if actual_token_position >= layer_activations.shape[1]:
            raise ValueError(f"Token position {actual_token_position} exceeds activation sequence length {layer_activations.shape[1]}")
        
        grad_output = torch.zeros_like(layer_activations)
        grad_output[0, actual_token_position, :] = 1.0
        
        # Use backward method like the old version
        layer_activations.backward(gradient=grad_output, retain_graph=True)
        
        # Get gradient from .grad attribute
        if input_embeds.grad is None:
            raise RuntimeError("Failed to compute gradients - input_embeds.grad is None")
        
        token_grad = input_embeds.grad[0, actual_token_position, :]
        
        if torch.isnan(token_grad).any() or torch.isinf(token_grad).any():
            raise ValueError("Gradient contains NaN or Inf values")
        
        grad_norm = token_grad.norm()
        if grad_norm < 1e-8:
            print(f"Warning: gradient norm is very small ({grad_norm:.2e})")
            normalized_grad = torch.zeros_like(token_grad)
        else:
            normalized_grad = token_grad / grad_norm
        
        return normalized_grad.detach().cpu()
        
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        if 'input_embeds' in locals():
            zero_grad = torch.zeros(input_embeds.shape[-1], device='cpu')
            return zero_grad
        else:
            raise e
        
    finally:
        handle.remove()
        # Clean up gradients
        if input_embeds.grad is not None:
            input_embeds.grad.zero_()
        torch.cuda.empty_cache()

def run_sae_intervention_on_small_models(
    model_name: str,
    layer_types: list,
    layer_range: range,
    model,
    tokenizer=None,  # Add tokenizer parameter
    similarity_threshold: float = 0.15,
    n_target_features: int = 3,
    n_interference_features: int = 4,
    n_top_tokens: int = 3,
    n_test_sentences: int = 3,
    scale_range: list = None,
    seed: int = 42,
    device: str = 'cuda',
    use_auto_model: bool = True,  # Add flag to choose model type
    data_loader_func = None
):
    """
    Run SAE intervention experiment on specified model and layers
    
    params:
        model_name: 'pythia' or 'gpt2'
        layer_types: list of layer types to test
        layer_range: range of layer indices to test
        model: model instance (AutoModel or HookedTransformer)
        tokenizer: tokenizer for AutoModel (required if use_auto_model=True)
        similarity_threshold: semantic similarity threshold for interference selection
        n_target_features: number of target features to sample
        n_interference_features: number of interference features per level
        n_top_tokens: number of top activation tokens to use for testing
        n_test_sentences: number of test sentences per token
        scale_range: list of scale values for intervention fine-tune
        seed: random seed
        device: device to run on
        use_auto_model: whether to use AutoModel (True) or HookedTransformer (False)
        
    return:
        dict: experiment results
    """
    import os
    import json
    import numpy as np
    from collections import defaultdict
    from .utils_data import get_all_level_interference_from, get_token_str_test_sentences
    from .utils_metrics import (
        get_weighted_cosine_similarity, 
        get_spearman_correlation, 
        get_kendall_tau_correlation, 
        get_weighted_overlap
    )
    
    if data_loader_func is None:
        data_loader_func = get_all_level_interference_from
    
    if scale_range is None:
        scale_values = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20]
        scale_range = [-x for x in scale_values[::-1]] + scale_values
    
    # Validate parameters for AutoModel
    if use_auto_model and tokenizer is None:
        raise ValueError("tokenizer is required when use_auto_model=True")
    
    # Get token embedding matrix
    if use_auto_model:
        # For AutoModel, get embeddings differently based on model type
        if model.config.model_type == "gpt_neox":  # Pythia
            token_embed_mat = model.gpt_neox.embed_in.weight.detach()
        elif model.config.model_type == "gpt2":  # GPT2
            token_embed_mat = model.transformer.wte.weight.detach()
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")
        vocab_size = model.config.vocab_size
    else:
        # For HookedTransformer
        token_embed_mat = model.embed.W_E.detach()
        vocab_size = model.cfg.d_vocab
    
    # Initialize results
    results = defaultdict(lambda: defaultdict(dict))
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def create_similarity_rankings(token_str):
        """Create similarity rankings for a single token"""
        try:
            if use_auto_model:
                token_id = tokenizer(token_str, return_tensors="pt")["input_ids"][0, 0].item()
            else:
                token_id = model.to_tokens(token_str, prepend_bos=False)[0, 0].item()
        except:
            return list(range(vocab_size))
        
        # Get embedding for target token
        target_embedding = token_embed_mat[token_id]
        
        # Calculate similarities with all tokens
        similarities = torch.cosine_similarity(
            target_embedding.unsqueeze(0), 
            token_embed_mat,
            dim=1
        )
        
        # Sort by similarity (highest to lowest)
        _, sorted_indices = torch.sort(similarities, descending=True)
        
        return sorted_indices.cpu().numpy().tolist()
    
    def get_random_unit_vector(dim):
        """Generate a random unit vector"""
        random_vec = torch.randn(dim, device=device)
        return random_vec / torch.norm(random_vec)
    
    def parse_similarity_range(range_str):
        """Parse similarity range string like '0.4-1.0' into tuple (0.4, 1.0)"""
        try:
            parts = range_str.split('-')
            return (float(parts[0]), float(parts[1]))
        except:
            return None
    
    def categorize_interference_level(range_str):
        """Categorize interference level based on similarity range"""
        range_tuple = parse_similarity_range(range_str)
        if range_tuple is None:
            return range_str  # Return original if can't parse
        
        min_sim, max_sim = range_tuple
        
        # Define categories based on similarity ranges
        if min_sim >= 0.4:
            return 'high'
        elif min_sim >= 0.3:
            return 'hmid'
        elif min_sim >= 0.2:
            return 'mid'
        elif min_sim >= 0.1:
            return 'lmid'
        else:
            return 'low'
    
    for layer_type in tqdm(layer_types):
        for layer_index in tqdm(layer_range):
            try:
                # Get target and interference features
                feature_data = data_loader_func(
                    model_name=model_name,
                    layer_type=layer_type,
                    layer_index=layer_index,
                    threshold=similarity_threshold,
                    n_target_features=n_target_features,
                    n_interference_features=n_interference_features,
                    n_top_tokens=5,
                    r_act_ratio=0.5,
                    seed=seed
                )
                
                if not feature_data:
                    print(f"No feature data found for {layer_type} layer {layer_index}")
                    continue
                
                # Get hook position for this layer
                if use_auto_model:
                    hook_position = f"{layer_index}-{layer_type}"
                else:
                    hook_position = parse_hook_position(f"{layer_index}-{layer_type}")
                
                # Process each target feature
                for target_feature_info in feature_data:
                    target_feature_id = target_feature_info['target_feature_id']
                    target_top_tokens = target_feature_info['target_top_tokens'][:n_top_tokens]
                    target_sae_direction = torch.tensor(
                        target_feature_info['target_feature_sae_direction'], 
                        device=device
                    )
                    
                    # Convert ALL target feature top tokens to IDs for metrics
                    all_target_token_ids = []
                    for t_str in target_top_tokens:
                        try:
                            if use_auto_model:
                                t_id = tokenizer(t_str, return_tensors="pt")["input_ids"][0, 0].item()
                            else:
                                t_id = model.to_tokens(t_str, prepend_bos=False)[0, 0].item()
                            all_target_token_ids.append(t_id)
                        except:
                            continue
                    
                    if not all_target_token_ids:
                        print(f"No valid tokens found for feature {target_feature_id}, skipping")
                        continue
                    
                    # Generate random unit vector for 'rand' level
                    sae_dim = target_sae_direction.shape[0]
                    random_direction = get_random_unit_vector(sae_dim)
                    
                    # Get available interference levels from the actual data
                    available_interference_levels = set(target_feature_info['interference_features'].keys())
                    
                    # Initialize feature results
                    feature_results = {
                        'target_feature_id': target_feature_id,
                        'target_explanation': target_feature_info['target_feature_explanation'],
                        'target_top_tokens': target_top_tokens,
                        'all_target_token_ids': all_target_token_ids,
                        'available_interference_levels': list(available_interference_levels),
                        'tests': {}
                    }
                    
                    # Test each top activation token
                    for token_idx, token_str in enumerate(target_top_tokens):
                        try:
                            # Get test sentences for this token
                            test_sentences = get_token_str_test_sentences(model_name, token_str)
                            selected_sentences = test_sentences[:n_test_sentences]
                            
                            # Create similarity rankings for this specific token
                            sim_rankings = create_similarity_rankings(token_str)
                            
                            token_results = {
                                'token_str': token_str,
                                'sentences': {}
                            }
                            
                            # Test each sentence
                            for sent_idx, sentence in enumerate(selected_sentences):
                                sentence_results = {}
                                
                                # Get baseline (raw model) output
                                if use_auto_model:
                                    raw_token_ids, raw_token_probs = run_with_raw_auto_model(model, tokenizer, sentence, device)
                                else:
                                    raw_token_ids, raw_token_probs = run_with_raw_model(model, sentence)
                                
                                # Calculate baseline metrics using ALL target tokens
                                baseline_weighted_cosine = get_weighted_cosine_similarity(
                                    pred_token_ids=raw_token_ids,
                                    pred_token_probs=raw_token_probs,
                                    target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                    token_embed_mat=token_embed_mat,
                                    compare_nums=vocab_size
                                )
                                
                                baseline_spearman, _ = get_spearman_correlation(
                                    predictions=raw_token_ids.cpu().numpy(),
                                    sim_rank=np.array(sim_rankings)
                                )
                                
                                baseline_kendall, _ = get_kendall_tau_correlation(
                                    predictions=raw_token_ids.cpu().numpy(),
                                    sim_rank=np.array(sim_rankings)
                                )
                                
                                baseline_weighted_overlap = get_weighted_overlap(
                                    pred_token_ids=raw_token_ids,
                                    pred_token_probs=raw_token_probs,
                                    target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                    compare_nums=vocab_size
                                )
                                
                                sentence_results['baseline'] = {
                                    'weighted_cosine_similarity': float(baseline_weighted_cosine),
                                    'spearman_correlation': float(baseline_spearman),
                                    'kendall_correlation': float(baseline_kendall),
                                    'weighted_overlap': float(baseline_weighted_overlap)
                                }
                                
                                # Test self level (target feature itself)
                                levels_to_test = ['self']
                                
                                # Add available interference levels from the data
                                levels_to_test.extend(available_interference_levels)
                                
                                # Add random level
                                levels_to_test.append('rand')
                                
                                for level in levels_to_test:
                                    if level == 'self':
                                        # Handle self level specially
                                        interference_direction = target_sae_direction
                                        interference_info = {
                                            'interference_feature_id': target_feature_id,
                                            'cosine_similarity': 1.0,
                                            'semantic_similarity': 1.0,
                                            'original_level': 'self'
                                        }
                                    elif level == 'rand':
                                        # Handle random level specially
                                        interference_direction = random_direction
                                        interference_info = {
                                            'interference_feature_id': None,
                                            'cosine_similarity': 0.0,
                                            'semantic_similarity': 0.0,
                                            'original_level': 'rand'
                                        }
                                    elif level in target_feature_info['interference_features']:
                                        # Handle actual interference levels from data
                                        level_features = target_feature_info['interference_features'][level]
                                        if not level_features:
                                            continue
                                        
                                        # Use first interference feature for this level
                                        interference_feature = level_features[0]
                                        interference_direction = torch.tensor(
                                            interference_feature['interference_feature_sae_direction'],
                                            device=device
                                        )
                                        interference_info = {
                                            'interference_feature_id': interference_feature['interference_feature_id'],
                                            'cosine_similarity': interference_feature['interference_value'],
                                            'semantic_similarity': interference_feature.get('semantic_similarity', 0.0),
                                            'original_level': level,
                                            'categorized_level': categorize_interference_level(level)
                                        }
                                    else:
                                        continue
                                    
                                    # Test different scales
                                    best_metrics = {
                                        'weighted_cosine_similarity': 0,
                                        'spearman_correlation': 0,
                                        'kendall_correlation': 0,
                                        'weighted_overlap': 0
                                    }
                                    best_scales = {
                                        'weighted_cosine_similarity': 0,
                                        'spearman_correlation': 0,
                                        'kendall_correlation': 0,
                                        'weighted_overlap': 0
                                    }
                                    
                                    for scale in scale_range:
                                        # Run intervention
                                        if use_auto_model:
                                            hooked_token_ids, hooked_token_probs = run_with_hooked_auto_model(
                                                model=model,
                                                tokenizer=tokenizer,
                                                text=sentence,
                                                hook_pos=hook_position,
                                                direction=interference_direction,
                                                scale=scale,
                                                device=device
                                            )
                                        else:
                                            hooked_token_ids, hooked_token_probs = run_with_hooked_model(
                                                model=model,
                                                text=sentence,
                                                hook_pos=hook_position,
                                                direction=interference_direction,
                                                scale=scale
                                            )
                                        
                                        # Calculate metrics using ALL target tokens
                                        current_weighted_cosine = get_weighted_cosine_similarity(
                                            pred_token_ids=hooked_token_ids,
                                            pred_token_probs=hooked_token_probs,
                                            target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                            token_embed_mat=token_embed_mat,
                                            compare_nums=vocab_size
                                        )
                                        
                                        current_spearman, _ = get_spearman_correlation(
                                            predictions=hooked_token_ids.cpu().numpy(),
                                            sim_rank=np.array(sim_rankings)
                                        )
                                        
                                        current_kendall, _ = get_kendall_tau_correlation(
                                            predictions=hooked_token_ids.cpu().numpy(),
                                            sim_rank=np.array(sim_rankings)
                                        )
                                        
                                        current_weighted_overlap = get_weighted_overlap(
                                            pred_token_ids=hooked_token_ids,
                                            pred_token_probs=hooked_token_probs,
                                            target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                            compare_nums=vocab_size
                                        )
                                        
                                        # Update best metrics if current is better
                                        if current_weighted_cosine > best_metrics['weighted_cosine_similarity']:
                                            best_metrics['weighted_cosine_similarity'] = current_weighted_cosine
                                            best_scales['weighted_cosine_similarity'] = scale
                                        
                                        if abs(current_spearman) > abs(best_metrics['spearman_correlation']):
                                            best_metrics['spearman_correlation'] = current_spearman
                                            best_scales['spearman_correlation'] = scale
                                        
                                        if abs(current_kendall) > abs(best_metrics['kendall_correlation']):
                                            best_metrics['kendall_correlation'] = current_kendall
                                            best_scales['kendall_correlation'] = scale
                                        
                                        if current_weighted_overlap > best_metrics['weighted_overlap']:
                                            best_metrics['weighted_overlap'] = current_weighted_overlap
                                            best_scales['weighted_overlap'] = scale
                                    
                                    # Store results for this interference level
                                    sentence_results[level] = {
                                        'interference_info': interference_info,
                                        'best_weighted_cosine_similarity': float(best_metrics['weighted_cosine_similarity']),
                                        'best_weighted_cosine_scale': int(best_scales['weighted_cosine_similarity']),
                                        'best_spearman_correlation': float(best_metrics['spearman_correlation']),
                                        'best_spearman_scale': int(best_scales['spearman_correlation']),
                                        'best_kendall_correlation': float(best_metrics['kendall_correlation']),
                                        'best_kendall_scale': int(best_scales['kendall_correlation']),
                                        'best_weighted_overlap': float(best_metrics['weighted_overlap']),
                                        'best_weighted_overlap_scale': int(best_scales['weighted_overlap'])
                                    }
                                
                                token_results['sentences'][sent_idx] = sentence_results
                            
                            feature_results['tests'][token_idx] = token_results
                            
                        except Exception as e:
                            print(f"    Error processing token '{token_str}': {e}")
                            continue
                    
                    results[layer_type][layer_index][target_feature_id] = feature_results
                    
                    # Clear cache to prevent memory issues
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {layer_type} layer {layer_index}: {e}")
                continue
    
    return dict(results)

def run_gradient_intervention_on_small_models(
    model_name: str,
    layer_types: list,
    layer_range: range,
    model,
    tokenizer,
    similarity_threshold: float = 0.15,
    n_target_features: int = 3,
    n_interference_features: int = 4,
    n_top_tokens: int = 3,
    n_test_sentences: int = 3,
    scale_range: list = None,
    seed: int = 42,
    device: str = 'cuda',
    use_hooked_model: bool = False,
    data_loader_func = None
):
    """
    Run gradient intervention experiment on specified model and layers
    Uses gradient vectors extracted from feature activation contexts
    
    For each target feature's top activation token, test with:
    - Self gradient vector (1 vector)
    - Each interference feature's gradient vectors (up to n_top_tokens vectors per feature)
    
    params:
        model_name: 'pythia' or 'gpt2'
        layer_types: list of layer types to test (only 'att' and 'mlp' supported)
        layer_range: range of layer indices to test
        model: Model instance (AutoModel or HookedTransformer)
        tokenizer: tokenizer for the model (required if use_hooked_model=False)
        similarity_threshold: semantic similarity threshold for interference selection
        n_target_features: number of target features to sample
        n_interference_features: number of interference features per level
        n_top_tokens: number of top activation tokens to use for testing
        n_test_sentences: number of test sentences per token
        scale_range: list of scale values for intervention fine-tune
        seed: random seed
        device: device to run on
        use_hooked_model: whether to use HookedTransformer (True) or AutoModel (False)
        
    return:
        dict: experiment results
    """
    import os
    import json
    import numpy as np
    from collections import defaultdict
    from .utils_data import get_all_level_interference_from, get_token_str_test_sentences
    from .utils_metrics import (
        get_weighted_cosine_similarity, 
        get_spearman_correlation, 
        get_kendall_tau_correlation, 
        get_weighted_overlap
    )
    
    if data_loader_func is None:
        data_loader_func = get_all_level_interference_from
    
    if scale_range is None:
        scale_values = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20]
        scale_range = [-x for x in scale_values[::-1]] + scale_values

    valid_layer_types = [lt for lt in layer_types if lt in ['att', 'mlp']]
    if len(valid_layer_types) != len(layer_types):
        print(f"Warning: Gradient intervention only supports 'att' and 'mlp' layers.")
        print(f"Filtering layer_types from {layer_types} to {valid_layer_types}")
        layer_types = valid_layer_types
    
    if not layer_types:
        print("No valid layer types for gradient intervention. Exiting.")
        return {}
    
    # Validate parameters
    if not use_hooked_model and tokenizer is None:
        raise ValueError("tokenizer is required when use_hooked_model=False")
    
    # Get token embedding matrix
    if use_hooked_model:
        token_embed_mat = model.embed.W_E.detach()
        vocab_size = model.cfg.d_vocab
    else:
        if model.config.model_type == "gpt_neox":  # Pythia
            token_embed_mat = model.gpt_neox.embed_in.weight.detach()
        elif model.config.model_type == "gpt2":  # GPT2
            token_embed_mat = model.transformer.wte.weight.detach()
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")
        vocab_size = model.config.vocab_size
    
    # Initialize results
    results = defaultdict(lambda: defaultdict(dict))
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def create_similarity_rankings(token_str):
        """Create similarity rankings for a single token"""
        try:
            if use_hooked_model:
                token_id = model.to_tokens(token_str, prepend_bos=False)[0, 0].item()
            else:
                token_id = tokenizer(token_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0, 0].item()
        except:
            return list(range(vocab_size))
        
        # Get embedding for target token
        target_embedding = token_embed_mat[token_id]
        
        # Calculate similarities with all tokens
        similarities = torch.cosine_similarity(
            target_embedding.unsqueeze(0), 
            token_embed_mat,
            dim=1
        )
        
        # Sort by similarity (highest to lowest)
        _, sorted_indices = torch.sort(similarities, descending=True)
        
        return sorted_indices.cpu().numpy().tolist()
    
    def get_random_unit_vector(dim):
        """Generate a random unit vector"""
        random_vec = torch.randn(dim, device=device)
        return random_vec / torch.norm(random_vec)
    
    def parse_similarity_range(range_str):
        """Parse similarity range string like '0.4-1.0' into tuple (0.4, 1.0)"""
        try:
            parts = range_str.split('-')
            return (float(parts[0]), float(parts[1]))
        except:
            return None
    
    def categorize_interference_level(range_str):
        """Categorize interference level based on similarity range"""
        range_tuple = parse_similarity_range(range_str)
        if range_tuple is None:
            return range_str
        
        min_sim, max_sim = range_tuple
        
        if min_sim >= 0.4:
            return 'high'
        elif min_sim >= 0.3:
            return 'hmid'
        elif min_sim >= 0.2:
            return 'mid'
        elif min_sim >= 0.1:
            return 'lmid'
        else:
            return 'low'
    
    for layer_type in tqdm(layer_types):
        for layer_index in tqdm(layer_range):
            try:
                # Get target and interference features with gradient information
                feature_data = data_loader_func(
                    model_name=model_name,
                    layer_type=layer_type,
                    layer_index=layer_index,
                    threshold=similarity_threshold,
                    n_target_features=n_target_features,
                    n_interference_features=n_interference_features,
                    n_top_tokens=n_top_tokens,  # Get more tokens for gradient extraction
                    r_act_ratio=0.5,
                    seed=seed,
                    extract_gradient_vectors=True,  # Enable gradient extraction
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    use_hooked_model=use_hooked_model
                )
                
                if not feature_data:
                    print(f"No feature data found for {layer_type} layer {layer_index}")
                    continue
                
                # Get hook position for this layer
                if use_hooked_model:
                    hook_position = parse_hook_position(f"{layer_index}-{layer_type}")
                else:
                    hook_position = f"{layer_index}-{layer_type}"
                
                # Process each target feature
                for target_feature_info in feature_data:
                    target_feature_id = target_feature_info['target_feature_id']
                    target_top_tokens = target_feature_info['target_top_tokens'][:n_top_tokens]
                    target_gradient_info = target_feature_info.get('target_gradient_info', {})
                    
                    # Convert ALL target feature top tokens to IDs for metrics
                    all_target_token_ids = []
                    for t_str in target_top_tokens:
                        try:
                            if use_hooked_model:
                                t_id = model.to_tokens(t_str, prepend_bos=False)[0, 0].item()
                            else:
                                t_id = tokenizer(t_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0, 0].item()
                            all_target_token_ids.append(t_id)
                        except:
                            continue
                    
                    if not all_target_token_ids:
                        print(f"No valid tokens found for feature {target_feature_id}, skipping")
                        continue
                    
                    # Generate random unit vector for 'rand' level
                    gradient_dim = None
                    random_direction = None
                    
                    # Try to get gradient dimension from target gradient info
                    for token_str, grad_info in target_gradient_info.items():
                        if 'gradient_vector' in grad_info:
                            gradient_dim = len(grad_info['gradient_vector'])
                            break
                    
                    if gradient_dim is not None:
                        random_direction = get_random_unit_vector(gradient_dim)
                    
                    # Get available interference levels from the actual data
                    available_interference_levels = set(target_feature_info['interference_features'].keys())
                    
                    # Initialize feature results
                    feature_results = {
                        'target_feature_id': target_feature_id,
                        'target_explanation': target_feature_info['target_feature_explanation'],
                        'target_top_tokens': target_top_tokens,
                        'all_target_token_ids': all_target_token_ids,
                        'available_interference_levels': list(available_interference_levels),
                        'target_gradient_info': target_gradient_info,
                        'tests': {}
                    }
                    
                    # Test each top activation token that has gradient information
                    for token_idx, token_str in enumerate(target_top_tokens):
                        if token_str not in target_gradient_info:
                            print(f"No gradient info for target token '{token_str}', skipping")
                            continue
                        
                        try:
                            # Get test sentences for this token
                            test_sentences = get_token_str_test_sentences(model_name, token_str)
                            selected_sentences = test_sentences[:n_test_sentences]
                            
                            # Create similarity rankings for this specific token
                            sim_rankings = create_similarity_rankings(token_str)
                            
                            # Get target gradient vector (for self level)
                            target_grad_vector = torch.tensor(
                                target_gradient_info[token_str]['gradient_vector'],
                                device=device
                            )
                            
                            token_results = {
                                'token_str': token_str,
                                'target_gradient_context': target_gradient_info[token_str],
                                'sentences': {}
                            }
                            
                            # Test each sentence
                            for sent_idx, sentence in enumerate(selected_sentences):
                                sentence_results = {}
                                
                                # Get baseline (raw model) output
                                if use_hooked_model:
                                    raw_token_ids, raw_token_probs = run_with_raw_model(model, sentence)
                                else:
                                    raw_token_ids, raw_token_probs = run_with_raw_auto_model(model, tokenizer, sentence, device)
                                
                                # Calculate baseline metrics using ALL target tokens
                                baseline_weighted_cosine = get_weighted_cosine_similarity(
                                    pred_token_ids=raw_token_ids,
                                    pred_token_probs=raw_token_probs,
                                    target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                    token_embed_mat=token_embed_mat,
                                    compare_nums=vocab_size
                                )
                                
                                baseline_spearman, _ = get_spearman_correlation(
                                    predictions=raw_token_ids.cpu().numpy(),
                                    sim_rank=np.array(sim_rankings)
                                )
                                
                                baseline_kendall, _ = get_kendall_tau_correlation(
                                    predictions=raw_token_ids.cpu().numpy(),
                                    sim_rank=np.array(sim_rankings)
                                )
                                
                                baseline_weighted_overlap = get_weighted_overlap(
                                    pred_token_ids=raw_token_ids,
                                    pred_token_probs=raw_token_probs,
                                    target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                    compare_nums=vocab_size
                                )
                                
                                sentence_results['baseline'] = {
                                    'weighted_cosine_similarity': float(baseline_weighted_cosine),
                                    'spearman_correlation': float(baseline_spearman),
                                    'kendall_correlation': float(baseline_kendall),
                                    'weighted_overlap': float(baseline_weighted_overlap)
                                }
                                
                                # Test self level (target feature gradient itself)
                                levels_to_test = ['self']
                                
                                # Add available interference levels from the data
                                levels_to_test.extend(available_interference_levels)
                                
                                # Add random level if we have gradient dimension
                                if random_direction is not None:
                                    levels_to_test.append('rand')
                                
                                for level in levels_to_test:
                                    if level == 'self':
                                        # Handle self level: use target feature's own gradient
                                        gradient_vectors_to_test = [target_grad_vector]
                                        level_info = {
                                            'interference_feature_id': target_feature_id,
                                            'cosine_similarity': 1.0,
                                            'semantic_similarity': 1.0,
                                            'original_level': 'self',
                                            'gradient_type': 'target'
                                        }
                                        
                                    elif level == 'rand':
                                        # Handle random level
                                        gradient_vectors_to_test = [random_direction]
                                        level_info = {
                                            'interference_feature_id': None,
                                            'cosine_similarity': 0.0,
                                            'semantic_similarity': 0.0,
                                            'original_level': 'rand',
                                            'gradient_type': 'random'
                                        }
                                        
                                    elif level in target_feature_info['interference_features']:
                                        # Handle actual interference levels from data
                                        level_features = target_feature_info['interference_features'][level]
                                        if not level_features:
                                            continue
                                        
                                        # Collect all gradient vectors from all interference features in this level
                                        gradient_vectors_to_test = []
                                        level_interference_features = []
                                        
                                        for interference_feature in level_features:
                                            if 'interference_gradient_info' in interference_feature:
                                                interference_grad_info = interference_feature['interference_gradient_info']
                                                
                                                # Get up to n_top_tokens gradient vectors from this interference feature
                                                feature_gradients = []
                                                for interference_token_str, grad_info in interference_grad_info.items():
                                                    if 'gradient_vector' in grad_info and len(feature_gradients) < n_top_tokens:
                                                        gradient_vector = torch.tensor(
                                                            grad_info['gradient_vector'],
                                                            device=device
                                                        )
                                                        feature_gradients.append({
                                                            'vector': gradient_vector,
                                                            'token_str': interference_token_str,
                                                            'activation_value': grad_info.get('activation_value', 0.0)
                                                        })
                                                
                                                if feature_gradients:
                                                    gradient_vectors_to_test.extend([fg['vector'] for fg in feature_gradients])
                                                    level_interference_features.append({
                                                        'feature_id': interference_feature['interference_feature_id'],
                                                        'feature_explanation': interference_feature['interference_feature_explanation'],
                                                        'cosine_similarity': interference_feature['interference_value'],
                                                        'semantic_similarity': interference_feature.get('semantic_similarity', 0.0),
                                                        'gradients': feature_gradients
                                                    })
                                        
                                        if not gradient_vectors_to_test:
                                            print(f"No gradient vectors found for interference level {level}, skipping")
                                            continue
                                        
                                        level_info = {
                                            'original_level': level,
                                            'categorized_level': categorize_interference_level(level),
                                            'gradient_type': 'interference',
                                            'interference_features': level_interference_features,
                                            'total_gradients': len(gradient_vectors_to_test)
                                        }
                                        
                                    else:
                                        continue
                                    
                                    # Test all gradient vectors for this level and find the best result
                                    level_best_metrics = {
                                        'weighted_cosine_similarity': baseline_weighted_cosine,
                                        'spearman_correlation': baseline_spearman,
                                        'kendall_correlation': baseline_kendall,
                                        'weighted_overlap': baseline_weighted_overlap
                                    }
                                    level_best_scales = {
                                        'weighted_cosine_similarity': 0,
                                        'spearman_correlation': 0,
                                        'kendall_correlation': 0,
                                        'weighted_overlap': 0
                                    }
                                    level_best_gradient_info = {
                                        'weighted_cosine_similarity': {'gradient_idx': -1, 'gradient_source': 'baseline'},
                                        'spearman_correlation': {'gradient_idx': -1, 'gradient_source': 'baseline'},
                                        'kendall_correlation': {'gradient_idx': -1, 'gradient_source': 'baseline'},
                                        'weighted_overlap': {'gradient_idx': -1, 'gradient_source': 'baseline'}
                                    }
                                    
                                    # Test each gradient vector in this level
                                    for grad_idx, gradient_vector in enumerate(gradient_vectors_to_test):
                                        for scale in scale_range:
                                            # Run gradient intervention
                                            if use_hooked_model:
                                                hooked_token_ids, hooked_token_probs = run_with_hooked_model(
                                                    model=model,
                                                    text=sentence,
                                                    hook_pos=hook_position,
                                                    direction=gradient_vector,
                                                    scale=scale
                                                )
                                            else:
                                                hooked_token_ids, hooked_token_probs = run_with_hooked_auto_model(
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    text=sentence,
                                                    hook_pos=hook_position,
                                                    direction=gradient_vector,
                                                    scale=scale,
                                                    device=device
                                                )
                                            
                                            # Calculate metrics using ALL target tokens
                                            current_weighted_cosine = get_weighted_cosine_similarity(
                                                pred_token_ids=hooked_token_ids,
                                                pred_token_probs=hooked_token_probs,
                                                target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                                token_embed_mat=token_embed_mat,
                                                compare_nums=vocab_size
                                            )
                                            
                                            current_spearman, _ = get_spearman_correlation(
                                                predictions=hooked_token_ids.cpu().numpy(),
                                                sim_rank=np.array(sim_rankings)
                                            )
                                            
                                            current_kendall, _ = get_kendall_tau_correlation(
                                                predictions=hooked_token_ids.cpu().numpy(),
                                                sim_rank=np.array(sim_rankings)
                                            )
                                            
                                            current_weighted_overlap = get_weighted_overlap(
                                                pred_token_ids=hooked_token_ids,
                                                pred_token_probs=hooked_token_probs,
                                                target_token_ids=torch.tensor(all_target_token_ids, device=device),
                                                compare_nums=vocab_size
                                            )
                                            
                                            # Update best metrics if current is better
                                            if current_weighted_cosine > level_best_metrics['weighted_cosine_similarity']:
                                                level_best_metrics['weighted_cosine_similarity'] = current_weighted_cosine
                                                level_best_scales['weighted_cosine_similarity'] = scale
                                                level_best_gradient_info['weighted_cosine_similarity'] = {
                                                    'gradient_idx': grad_idx,
                                                    'gradient_source': level
                                                }
                                            
                                            if abs(current_spearman) > abs(level_best_metrics['spearman_correlation']):
                                                level_best_metrics['spearman_correlation'] = current_spearman
                                                level_best_scales['spearman_correlation'] = scale
                                                level_best_gradient_info['spearman_correlation'] = {
                                                    'gradient_idx': grad_idx,
                                                    'gradient_source': level
                                                }
                                            
                                            if abs(current_kendall) > abs(level_best_metrics['kendall_correlation']):
                                                level_best_metrics['kendall_correlation'] = current_kendall
                                                level_best_scales['kendall_correlation'] = scale
                                                level_best_gradient_info['kendall_correlation'] = {
                                                    'gradient_idx': grad_idx,
                                                    'gradient_source': level
                                                }
                                            
                                            if current_weighted_overlap > level_best_metrics['weighted_overlap']:
                                                level_best_metrics['weighted_overlap'] = current_weighted_overlap
                                                level_best_scales['weighted_overlap'] = scale
                                                level_best_gradient_info['weighted_overlap'] = {
                                                    'gradient_idx': grad_idx,
                                                    'gradient_source': level
                                                }
                                    
                                    # Store results for this interference level
                                    sentence_results[level] = {
                                        'interference_info': level_info,
                                        'best_weighted_cosine_similarity': float(level_best_metrics['weighted_cosine_similarity']),
                                        'best_weighted_cosine_scale': int(level_best_scales['weighted_cosine_similarity']),
                                        'best_weighted_cosine_gradient': level_best_gradient_info['weighted_cosine_similarity'],
                                        'best_spearman_correlation': float(level_best_metrics['spearman_correlation']),
                                        'best_spearman_scale': int(level_best_scales['spearman_correlation']),
                                        'best_spearman_gradient': level_best_gradient_info['spearman_correlation'],
                                        'best_kendall_correlation': float(level_best_metrics['kendall_correlation']),
                                        'best_kendall_scale': int(level_best_scales['kendall_correlation']),
                                        'best_kendall_gradient': level_best_gradient_info['kendall_correlation'],
                                        'best_weighted_overlap': float(level_best_metrics['weighted_overlap']),
                                        'best_weighted_overlap_scale': int(level_best_scales['weighted_overlap']),
                                        'best_weighted_overlap_gradient': level_best_gradient_info['weighted_overlap']
                                    }
                                
                                token_results['sentences'][sent_idx] = sentence_results
                            
                            feature_results['tests'][token_idx] = token_results
                            
                        except Exception as e:
                            print(f"    Error processing token '{token_str}': {e}")
                            continue
                    
                    results[layer_type][layer_index][target_feature_id] = feature_results
                    
                    # Clear cache to prevent memory issues
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {layer_type} layer {layer_index}: {e}")
                continue
    
    return dict(results)