import os
import json
import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy import stats
from typing import List, Tuple
from safetensors import safe_open
from sklearn.metrics.pairwise import cosine_similarity
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer
from .utils_exp import get_gradient_intervention_vector_for_auto_model

def get_sae_features_by_layer(
    model_name: str, layer_type: str, layer_index: str | int, active_only: bool = True
) -> dict:
    """
    get SAE features in a specific layer of the assigned model

    params:
        model_name (str): 'pythia' or 'gpt2'
        layer_type (str): 'att', 'res', 'mlp' for pythia; 'att',
                          'res_mid', 'mlp', 'res_post' for gpt2
        layer_index (int): layer index
        active_only (bool): 
        
    return:
        dict: <feature_index: int, feature_data>
    """

    if model_name == 'pythia':
        data_path = f'YOUR_PATH_OF_DATA'
    elif model_name == 'gpt2':
        data_path = f'YOUR_PATH_OF_DATA'
        
    sae_features = {}
    for raw_json in os.listdir(data_path):
        if raw_json.endswith(".json"):
            with open(os.path.join(data_path, raw_json), 'r') as f:
                features = json.load(f)
                for feature in features:
                    if active_only:
                        if feature['neuron_alignment_indices']:
                            sae_features[int(feature['index'])] = feature
                    else:
                        sae_features[int(feature['index'])] = feature

    return sae_features

def get_sae_decoder_weights(
    model_name: str, layer_type: str, layer_index: str | int
) -> dict:
    """
    get the weights of the decoder of SAE trained in a specific layer.

    params:
        model_name (str): 'pythia', 'gpt2', or 'gemma2'
        layer_type (str): 'att', 'res', 'mlp' for pythia; 'att',
                          'res_mid', 'mlp', 'res_post' for gpt2
        layer_index (int): layer index

    return:
        torch.Tensor: layer weights
    """

    if model_name == 'pythia':
        release_name = f'pythia-70m-deduped-{layer_type}-sm'
        release = get_pretrained_saes_directory()[release_name].__dict__
        for hook_name, _ in release['saes_map'].items():
            if layer_index in hook_name:
                sae, _, _ = SAE.from_pretrained(
                    release=release['release'],
                    sae_id=hook_name
                )

                return sae.W_dec
            
    elif model_name == 'gpt2':
        if layer_type == 'att':
            release_name = f'gpt2-small-attn-out-v5-32k'
        elif layer_type == 'res_mid':
            release_name = f'gpt2-small-resid-mid-v5-32k'
        elif layer_type == 'res_post':
            release_name = f'gpt2-small-resid-post-v5-32k'
        elif layer_type == 'mlp':
            release_name = f'gpt2-small-mlp-out-v5-32k'

        release = get_pretrained_saes_directory()[release_name].__dict__
        for hook_name, _ in release['saes_map'].items():
            if layer_index in hook_name:
                sae, _, _ = SAE.from_pretrained(
                    release=release['release'],
                    sae_id=hook_name
                )

                return sae.W_dec
            
    elif model_name == 'gemma2-2b-16k':
        release_name = f'gemma-scope-2b-pt-{layer_type}-canonical'
        release = get_pretrained_saes_directory()[release_name].__dict__

        hook_name = f'layer_{layer_index}/width_16k/canonical'
        sae, _, _ = SAE.from_pretrained(
            release=release['release'],
            sae_id=hook_name
        )
        return sae.W_dec
    
    elif model_name == 'gemma2-2b-65k':
        release_name = f'gemma-scope-2b-pt-{layer_type}-canonical'
        release = get_pretrained_saes_directory()[release_name].__dict__

        hook_name = f'layer_{layer_index}/width_65k/canonical'
        sae, _, _ = SAE.from_pretrained(
            release=release['release'],
            sae_id=hook_name
        )
        return sae.W_dec
    
def get_sae_decoder_weights_from_local(
    model_name: str, layer_type: str, layer_index: str | int
):
    """
    get the weights of the decoder of SAE trained in a specific layer from local save.
    each weight vector has been normalized

    params:
        model_name: 'pythia', 'gpt2', or 'gemma2'
        layer_type: 'att', 'res', 'mlp' for pythia; 'att',
                          'res_mid', 'mlp', 'res_post' for gpt2
        layer_index: layer index

    return:
        torch.Tensor[feature_nums, d_embedding]: W_dec
    """

    data_path = f'YOUR_PATH_TO_TRAINED_SAE_AT_{layer_type,layer_index}_OF_{model_name}'

    with safe_open(data_path, framework='pt') as f:
        W_dec = f.get_tensor('W_dec')

    return W_dec

def get_feature_explanations_by_layer(
    model_name: str, layer_type: str, layer_index: str | int
) -> dict:
    """
    get explanations of features generated by GPT-4o-mini in a specific layer

    params:
        model_name: 'pythia' or 'gpt2'
        layer_type: 'att', 'res', 'mlp' for pythia; 'att', 'res_mid', 'mlp', 'res_post' for gpt2
        layer_index: layer index

    return:
        dict: <feature_index: int, feature_explanation: str>
    """
    data_path = 'YOUR_PATH_OF_DATA'

    feature_explanations = {}
    for raw_jsonl in os.listdir(data_path):
        if raw_jsonl.endswith(".jsonl"):
            with open(os.path.join(data_path, raw_jsonl), 'r') as f:
                for line in f:
                    feature = json.loads(line)
                    feature_explanations[int(feature['index'])] = feature['description']

    return feature_explanations

def get_feature_explanation_embeddings_by_layer(
    model_name: str, layer_type: str, layer_index: str | int
) -> torch.Tensor:
    """
    get embeddings of feature explanations in GPT-4o-mini

    params:
        model_name: 'pythia' or 'gpt2'
        layer_type: 'att', 'res', 'mlp' for pythia; 'att', 'res_mid', 'mlp', 'res_post' for gpt2
        layer_index: layer index

    return:
        dict: <feature_index: int, feature_embedding: torch.Tensor>
    """
    
    exp_dict = {}

    # get explanations' raw data path
    data_path = 'YOUR_PATH_OF_DATA'

    # get explanation embeddings
    for exp_clip in os.listdir(data_path):
        if exp_clip.endswith('.jsonl'):
            with open(f'{data_path}/{exp_clip}') as f:
                for line in f:
                    exp = json.loads(line)
                    feature_index = int(exp['index'])
                    if feature_index not in exp_dict:
                        exp_dict[feature_index] = []
                    exp_dict[feature_index].append(
                        [float(x.strip()) for x in exp['embedding'].strip('[]').split(',')]
                    )

    for feature, embs in exp_dict.items():
        if len(embs) > 1:
            exp_dict[feature] = torch.tensor(np.mean(embs, axis=0), dtype=torch.float32)
        else:
            exp_dict[feature] = torch.tensor(embs[0], dtype=torch.float32)
    return exp_dict

def get_pythia_att_5_explanation_embeddings():
    with open('YOUR_PATH_OF_DATA.json', 'r') as f:
        deepseek_embeddings_data = json.load(f)
        
    deepseek_embeddings = {}
    for feature_id, feature_data in deepseek_embeddings_data['feature_embeddings'].items():
        deepseek_embeddings[int(feature_id)] = feature_data['embedding']
        
    return deepseek_embeddings

def get_token_id_test_sentences(model_name, token_id):
    base_path = '../corpus/'

    if model_name == 'pythia':
        file_path = base_path+'pythia_vocabulary_sentences.json'
    elif model_name == 'gpt2':
        file_path = base_path+'gpt2_vocabulary_sentences.json'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # load as a json variable
    with open(file_path, 'r') as f:
        token_sentences = json.load(f)

    return list(token_sentences[str(token_id)].values())[0]

def get_token_str_test_sentences(model_name, token_str):
    base_path = '/PATH_TO_THIS_PROJECT/llm_intervention/corpus/'

    if model_name == 'pythia':
        file_path = base_path+'pythia_vocabulary_sentences.json'
    elif model_name == 'gpt2':
        file_path = base_path+'gpt2_vocabulary_sentences.json'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # load as a json variable
    with open(file_path, 'r') as f:
        token_sentences = json.load(f)

    for token_id, token_info in token_sentences.items():
        for key, value in token_info.items():
            if key == token_str:
                return value
    
    raise ValueError(f"Token string '{token_str}' not found in {model_name} vocabulary")

def get_all_level_interference_from(
    model_name: str,
    layer_type: str,
    layer_index: int,
    threshold: float,
    n_target_features: int = 3,
    n_interference_features: int = 4,
    n_top_tokens: int = 10,
    r_act_ratio: float = 0.0,
    seed: int = 42,
    extract_gradient_vectors: bool = False,
    model=None,
    tokenizer=None,
    device: str = 'cuda',
    use_hooked_model: bool = False
) -> List[dict]:
    """
    Sample target features from pre-filtered interference files;
    for each target feature, sample interference features 
    from all interference ranges with top activation tokens comparison
    
    params:
        model_name (str): 'pythia' or 'gpt2'
        layer_type (str): 'att', 'res', 'mlp' for pythia; 'att', 'res_mid', 'mlp', 'res_post' for gpt2
        layer_index (int): layer index
        threshold (float): semantic similarity threshold (0.4, 0.3, 0.2, 0.15, 0.1)
        n_target_features (int): number of target features to sample
        n_interference_features (int): number of interference features to sample
            for each target feature
        n_top_tokens (int): number of top activation tokens to compare
        r_act_ratio (float): activation ratio threshold for top tokens (0-1)
        seed (int): random seed
        extract_gradient_vectors (bool): whether to extract gradient intervention vectors
        model: Model instance (AutoModel or HookedTransformer, required if extract_gradient_vectors=True)
        tokenizer: tokenizer instance (required if extract_gradient_vectors=True and use_hooked_model=False)
        device (str): device to run on
        use_hooked_model (bool): whether to use HookedTransformer (True) or AutoModel (False)
        
    return:
        List[dict]: dict of all the information about sampled features
    """
    
    # Validate gradient extraction parameters
    if extract_gradient_vectors:
        if model is None:
            raise ValueError("model is required when extract_gradient_vectors=True")
        
        if layer_type not in ['att', 'mlp']:
            print(f"Warning: Gradient extraction is only supported for 'att' and 'mlp' layers, skipping for layer_type='{layer_type}'")
            extract_gradient_vectors = False
        
        if not use_hooked_model and tokenizer is None:
            raise ValueError("tokenizer is required when extract_gradient_vectors=True and use_hooked_model=False")
        
        # Import the appropriate gradient extraction function
        if use_hooked_model:
            from .utils_exp import get_gradient_intervention_vector_for_hooked_model
        else:
            from .utils_exp import get_gradient_intervention_vector_for_auto_model
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Construct file path for pre-filtered interference data
    # NOTE: This is the preprocessed data files generated by 
    # /preprocessing/interference_overview.ipynb
    base_path = '/YOUR_PATH_PREPROCESSED_DATA/med_data/all_level_interferences/'
    filename = f"{model_name}_{layer_type}_{layer_index}_{threshold:.2f}.json"
    file_path = os.path.join(base_path, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pre-filtered interference file not found: {file_path}")
    
    # Load pre-filtered interference data
    with open(file_path, 'r', encoding='utf-8') as f:
        interference_data = json.load(f)
    
    if not interference_data:
        raise ValueError(f"No interference features found in {filename}")
    
    # Get feature IDs that have complete interference ranges
    available_feature_ids = list(interference_data.keys())
    
    if len(available_feature_ids) < n_target_features:
        raise ValueError(f"Requested {n_target_features} target features, but only {len(available_feature_ids)} available in {filename}")
    
    # Randomly sample target feature IDs
    target_feature_ids = random.sample(available_feature_ids, n_target_features)
    
    # Get additional data needed for complete feature information
    decoder_weights = get_sae_decoder_weights_from_local(model_name, layer_type, layer_index)
    explanation_embeddings_dict = get_feature_explanation_embeddings_by_layer(model_name, layer_type, layer_index)
    feature_explanations = get_feature_explanations_by_layer(model_name, layer_type, layer_index)
    
    # Get all SAE features for top token comparison
    all_features = get_sae_features_by_layer(model_name, layer_type, layer_index, active_only=True)
    
    def extract_highest_activation_context(feature_data, token_str):
        """
        Extract the context (token_strs) and position where the target token appears
        with the HIGHEST activation value from the feature's activation data
        
        Returns:
            Tuple[List[str], int, List[float]] or None: (token_strs, target_token_position, activation_values)
            for the highest activation context, or None if not found
        """
        best_context = None
        highest_activation = float('-inf')
        
        if 'activations' not in feature_data:
            return best_context
        
        for activation_record in feature_data['activations']:
            if 'tokens' not in activation_record or 'values' not in activation_record:
                continue
                
            tokens = activation_record['tokens']
            values = activation_record['values']
            
            # Ensure tokens and values have same length
            if len(tokens) != len(values):
                continue
            
            # Find all positions where the target token appears
            for pos, token in enumerate(tokens):
                if token == token_str:
                    activation_value = values[pos]
                    if activation_value > highest_activation:
                        highest_activation = activation_value
                        # Return complete activation values list
                        best_context = (tokens, pos, values)
        
        return best_context
    
    def extract_gradient_vectors_for_all_tokens(feature_data, top_tokens, feature_id, layer_idx):
        """
        Extract gradient vectors for ALL top tokens of a feature
        
        Args:
            feature_data: feature activation data
            top_tokens: list of top token strings
            feature_id: feature identifier for error reporting
            layer_idx: layer index for gradient extraction
            
        Returns:
            dict: mapping from token_str to gradient info, or empty dict if extraction fails
        """
        if not extract_gradient_vectors:
            return {}
        
        gradient_info_dict = {}
        
        for token_str in top_tokens:
            try:
                # Extract the highest activation context for this token
                best_context = extract_highest_activation_context(feature_data, token_str)
                
                if best_context is None:
                    print(f"Warning: No context found for token '{token_str}' in feature {feature_id}")
                    continue
                
                token_strs, target_token_position, activation_values = best_context
                
                try:
                    if use_hooked_model:
                        # Use HookedTransformer gradient extraction with activation_values
                        gradient_vector = get_gradient_intervention_vector_for_hooked_model(
                            model=model,
                            token_strs=token_strs,
                            token_str_position=target_token_position,
                            layer_idx=layer_idx,
                            layer_type=layer_type,
                            r_act_ratio=0.8,
                            activation_values=activation_values
                        )
                    else:
                        # Use AutoModel gradient extraction
                        gradient_vector = get_gradient_intervention_vector_for_auto_model(
                            model=model,
                            tokenizer=tokenizer,
                            token_strs=token_strs,
                            token_str_position=target_token_position,
                            layer_idx=layer_idx,
                            layer_type=layer_type,
                            device=device
                        )
                    
                    # Convert to CPU list for JSON serialization
                    gradient_vector_cpu = gradient_vector.detach().cpu().tolist()
                    
                    gradient_info_dict[token_str] = {
                        'token_strs': token_strs,
                        'target_token_position': target_token_position,
                        'activation_values': activation_values,
                        'gradient_vector': gradient_vector_cpu
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to extract gradient for token '{token_str}' at position {target_token_position} in feature {feature_id}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Failed to process token '{token_str}' for gradient extraction: {e}")
                continue
        
        return gradient_info_dict
    
    results = []
    
    for target_feature_id_str in target_feature_ids:
        target_feature_id = int(target_feature_id_str)
        
        # Get target feature's interference data from pre-filtered file
        target_data = interference_data[target_feature_id_str]
        target_explanation = target_data['explanation']
        available_interference_ranges = target_data['interference_features']
        
        # Get target feature's decoder weights and explanation embedding
        target_decoder = decoder_weights[target_feature_id]
        target_explanation_emb = explanation_embeddings_dict[target_feature_id]
        
        # Get target feature's top activation tokens
        if target_feature_id not in all_features:
            continue
            
        target_feature_data = all_features[target_feature_id]
        target_top_tokens = get_unique_top_act_tokens(
            target_feature_data,
            n=n_top_tokens,
            r=r_act_ratio
        )
        target_top_tokens_set = set(target_top_tokens)
        
        # Extract gradient vectors for ALL top tokens of target feature if requested
        target_gradient_info = extract_gradient_vectors_for_all_tokens(
            target_feature_data, target_top_tokens, target_feature_id, layer_index
        )
        
        # Initialize target feature result structure
        target_result = {
            'target_feature_id': target_feature_id,
            'target_feature_explanation': target_explanation,
            'target_feature_sae_direction': target_decoder.tolist(),
            'target_top_tokens': target_top_tokens,
        }
        
        # Add gradient information if extracted
        if target_gradient_info:
            target_result['target_gradient_info'] = target_gradient_info
            # Also add summary statistics
            target_result['target_gradient_summary'] = {
                'total_tokens': len(target_top_tokens),
                'successful_extractions': len(target_gradient_info),
                'failed_extractions': len(target_top_tokens) - len(target_gradient_info),
                'success_rate': len(target_gradient_info) / len(target_top_tokens) if target_top_tokens else 0.0
            }
        
        target_result['interference_features'] = {}
        
        # For each interference range, sample features from pre-filtered candidates
        for range_key, candidate_feature_ids in available_interference_ranges.items():
            if not candidate_feature_ids:
                # No candidates in this range
                target_result['interference_features'][range_key] = []
                continue
            
            # Parse range bounds for validation
            range_min, range_max = map(float, range_key.split('-'))
            
            # Randomly shuffle candidate list based on seed
            random_start_idx = random.randint(0, len(candidate_feature_ids) - 1)
            
            # Create a circular iteration starting from random index
            ordered_candidates = (
                candidate_feature_ids[random_start_idx:] + 
                candidate_feature_ids[:random_start_idx]
            )
            
            # Select interference features with top token comparison
            selected_interference_features = []
            
            for candidate_id in ordered_candidates:
                if len(selected_interference_features) >= n_interference_features:
                    break
                    
                # Check if candidate feature exists in SAE features
                if candidate_id not in all_features:
                    continue
                
                # Get candidate's top activation tokens
                candidate_feature_data = all_features[candidate_id]
                candidate_top_tokens = get_unique_top_act_tokens(
                    candidate_feature_data,
                    n=n_top_tokens,
                    r=r_act_ratio
                )
                candidate_top_tokens_set = set(candidate_top_tokens)
                
                # Check for overlap with target feature's top tokens
                # if target_top_tokens_set & candidate_top_tokens_set:
                #     # Has overlap, skip this candidate
                #     continue
                
                # Get candidate's decoder and embedding for precise validation
                candidate_decoder = decoder_weights[candidate_id]
                candidate_explanation_emb = explanation_embeddings_dict[candidate_id]
                
                # Compute precise decoder similarity (interference value)
                decoder_similarity = F.cosine_similarity(
                    target_decoder.unsqueeze(0), 
                    candidate_decoder.unsqueeze(0)
                ).item()
                
                # Compute precise semantic similarity
                semantic_similarity = F.cosine_similarity(
                    target_explanation_emb.unsqueeze(0),
                    candidate_explanation_emb.unsqueeze(0)
                ).item()
                
                # Validate that the candidate meets the requirements
                # 1. Interference value should be in the specified range
                if not (range_min <= decoder_similarity <= range_max):
                    continue
                
                # 2. Semantic similarity should be below threshold
                if semantic_similarity >= threshold:
                    continue
                
                # Extract gradient vectors for ALL top tokens of interference feature if requested
                interference_gradient_info = extract_gradient_vectors_for_all_tokens(
                    candidate_feature_data, candidate_top_tokens, candidate_id, layer_index
                )
                
                # Candidate passes all validation, add to selected features
                interference_feature_data = {
                    'interference_feature_id': candidate_id,
                    'interference_feature_explanation': feature_explanations[candidate_id],
                    'interference_feature_sae_direction': candidate_decoder.tolist(),
                    'interference_top_tokens': candidate_top_tokens,
                    'interference_value': decoder_similarity,
                    'semantic_similarity': semantic_similarity
                }
                
                # Add gradient information if extracted
                if interference_gradient_info:
                    interference_feature_data['interference_gradient_info'] = interference_gradient_info
                    # Also add summary statistics
                    interference_feature_data['interference_gradient_summary'] = {
                        'total_tokens': len(candidate_top_tokens),
                        'successful_extractions': len(interference_gradient_info),
                        'failed_extractions': len(candidate_top_tokens) - len(interference_gradient_info),
                        'success_rate': len(interference_gradient_info) / len(candidate_top_tokens) if candidate_top_tokens else 0.0
                    }
                
                selected_interference_features.append(interference_feature_data)
            
            # Store selected features for this range
            target_result['interference_features'][range_key] = selected_interference_features
        
        results.append(target_result)
    
    return results

def get_unique_top_act_tokens(
    feature_data: dict, n: int = 10, r: float = 0.0
) -> List[str]:
    """
    Get top n unique tokens with highest activation values from feature data
    If a token appears multiple times, only keep the occurrence with highest activation
    Continue to find different tokens until reaching n unique tokens
    Additionally filter tokens by relative activation threshold
    
    params:
        feature_data (dict): feature data containing 'activations' field
        n (int): number of top unique tokens to return
        r (float): relative threshold (0-1), tokens must have activation >= r * max_activation
        
    return:
        List[str]: list of top n unique tokens with highest activation values that meet threshold
    """
    
    if 'activations' not in feature_data:
        raise ValueError("Feature data must contain 'activations' field")
    
    if not 0 <= r <= 1:
        raise ValueError(f"Parameter r must be between 0 and 1, got {r}")
    
    # Dictionary to store the maximum activation value for each unique token
    token_max_activation = {}
    
    for activation_record in feature_data['activations']:
        if 'tokens' not in activation_record or 'values' not in activation_record:
            continue
            
        tokens = activation_record['tokens']
        values = activation_record['values']
        
        # Ensure tokens and values have same length
        if len(tokens) != len(values):
            continue
            
        # Update maximum activation for each token
        for token, value in zip(tokens, values):
            if token not in token_max_activation or value > token_max_activation[token]:
                token_max_activation[token] = value
    
    if not token_max_activation:
        return []
    
    # Sort unique tokens by their maximum activation value in descending order
    sorted_tokens = sorted(token_max_activation.items(), key=lambda x: x[1], reverse=True)
    
    # Get the maximum activation value for threshold calculation
    max_activation = sorted_tokens[0][1]
    activation_threshold = r * max_activation
    
    # First, get top n tokens
    top_n_tokens = sorted_tokens[:n]
    
    # Then, filter by activation threshold
    filtered_tokens = [
        token for token, activation in top_n_tokens 
        if activation >= activation_threshold
    ]
    
    return filtered_tokens

def load_type_dicts(model_name: str):
    """
    Load token type mapping dictionaries for the specified model
    
    Args:
        model_name (str): 'pythia' or 'gpt2'
        
    Returns:
        tuple: (token_id_to_type_dict, token_str_to_type_dict) for the specified model
    """
    import json
    import os
    
    base_path = os.path.join(os.path.dirname(__file__), '../dataset')
    
    if model_name == 'pythia':
        id_file = os.path.join(base_path, 'pythia_token_id_to_type.json')
        str_file = os.path.join(base_path, 'pythia_token_str_to_type.json')
    elif model_name == 'gpt2':
        id_file = os.path.join(base_path, 'gpt2_token_id_to_type.json')
        str_file = os.path.join(base_path, 'gpt2_token_str_to_type.json')
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Must be 'pythia' or 'gpt2'")
    
    # Load token_id to type mapping
    try:
        with open(id_file, 'r', encoding='utf-8') as f:
            token_id_to_type = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {id_file} not found, using empty dict")
        token_id_to_type = {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing {id_file}: {e}, using empty dict")
        token_id_to_type = {}
    
    # Load token_str to type mapping
    try:
        with open(str_file, 'r', encoding='utf-8') as f:
            token_str_to_type = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {str_file} not found, using empty dict")
        token_str_to_type = {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing {str_file}: {e}, using empty dict")
        token_str_to_type = {}
    
    return token_id_to_type, token_str_to_type

def load_large_model_token_type_dict(
    json_path = '../dataset/large_model_token_type.json'
) -> dict:
    """
    load large_model_token_type.json as dict

    Args:
        json_path (str): file path

    Returns:
        dict: {token_str: token_type, ...}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def get_large_model_token_type(
    token_str: str, large_model_token_type_dict: dict
) -> str:
    return large_model_token_type_dict.get(token_str, 'unknown')

def get_token_id_type(
    model_name: str, token_id: int, token_id_to_type_dict: dict = None
) -> str:
    """
    Get token type by token_id
    
    Args:
        model_name (str): 'pythia' or 'gpt2'
        token_id (int): The token ID
        token_id_to_type_dict (dict, optional): Token ID to type mapping dict. If provided, use directly
        
    Returns:
        str: Token type, 'unknown' if not found
    """
    if token_id_to_type_dict is None:
        token_id_to_type_dict, _ = load_type_dicts(model_name)
    
    # Convert token_id to string key (JSON keys are strings)
    token_id_str = str(token_id)
    
    return token_id_to_type_dict.get(token_id_str, 'unknown')

def get_token_str_type(
    model_name: str, token_str: str, token_str_to_type_dict: dict = None
) -> str:
    """
    Get token type by token_str
    
    Args:
        model_name (str): 'pythia' or 'gpt2'
        token_str (str): The token string representation
        token_str_to_type_dict (dict, optional): Token string to type mapping dict. If provided, use directly
        
    Returns:
        str: Token type, 'unknown' if not found
    """
    if token_str_to_type_dict is None:
        _, token_str_to_type_dict = load_type_dicts(model_name)
    
    return token_str_to_type_dict.get(token_str, 'unknown')

def get_target_type_and_interference_feature_tokens(
    model_name: str,
    token_type: str,
    r_act_ratio: float = 0.8,
    interference_threshold: float = 0.2,
    semantic_threshold: float = 0.3
) -> dict:
    """
    Find target type features and their interference features across all layers
    """
    import torch
    import torch.nn.functional as F
    import gc
    from .utils_model import get_hooked_pythia_70m, get_hooked_gpt2_small
    from tqdm.notebook import tqdm
    
    def extract_high_activation_tokens(feature_data, r_act_ratio):
        """Extract token strings sorted by activation value in descending order"""
        if not feature_data.get('activations'):
            return []
        
        # Find maximum activation across all contexts
        max_activation = 0
        for activation in feature_data['activations']:
            if activation.get('values'):
                current_max = max(activation['values'])
                max_activation = max(max_activation, current_max)
        
        if max_activation == 0:
            return []
        
        threshold_value = r_act_ratio * max_activation
        token_value_pairs = []
        
        for activation in feature_data['activations']:
            if not activation.get('values') or not activation.get('tokens'):
                continue
            
            values = activation['values']
            tokens = activation['tokens']
            
            # Collect tokens and their activation values
            for value, token in zip(values, tokens):
                if value >= threshold_value:
                    token_value_pairs.append((token, value))
        
        # Sort by activation value in descending order
        token_value_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates while preserving the highest activation value order
        seen = set()
        unique_tokens = []
        for token, value in token_value_pairs:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    # Load model for token conversion
    if model_name == 'pythia':
        model = get_hooked_pythia_70m('cpu')
        layer_types = ['att', 'mlp', 'res']
        n_layers = 6
    elif model_name == 'gpt2':
        model = get_hooked_gpt2_small('cpu')
        layer_types = ['att', 'mlp', 'res_mid', 'res_post']
        n_layers = 12
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load token type dictionaries
    token_id_to_type_dict, token_str_to_type_dict = load_type_dicts(model_name)
    
    results = {}
    
    print(f"Analyzing {model_name} for token type '{token_type}'...")
    print(f"Parameters: r_act_ratio={r_act_ratio}, interference_threshold={interference_threshold}, semantic_threshold={semantic_threshold}")
    
    for layer_type in tqdm(layer_types, desc="Layer types", position=0):
        results[layer_type] = {}
        
        for layer_idx in tqdm(range(n_layers), 
                             desc=f"{layer_type} layers", 
                             position=1, 
                             leave=False):
            try:
                # Get features for current layer - keys are int
                all_features = get_sae_features_by_layer(model_name, layer_type, layer_idx)
                
                if not all_features:
                    tqdm.write(f"Warning: No features found for {model_name} {layer_type} layer {layer_idx}")
                    continue
                
                # Step 1: Identify target type features
                target_features = []
                
                for feature_id, feature_data in all_features.items():
                    if not feature_data.get('activations'):
                        continue
                    
                    try:
                        # Use get_unique_top_act_tokens to get high activation tokens
                        high_activation_tokens = get_unique_top_act_tokens(
                            feature_data, 
                            n=100,  # Get enough tokens to check
                            r=r_act_ratio
                        )
                        
                        if not high_activation_tokens:
                            continue
                        
                        # Check if any high activation token has target type
                        has_target_type = False
                        for token in high_activation_tokens:
                            token_type_result = get_token_str_type(
                                model_name, token, token_str_to_type_dict
                            )
                            if token_type_result == token_type:
                                has_target_type = True
                                break
                        
                        if has_target_type:
                            # Extract high activation token strings sorted by activation value
                            high_activation_token_strs = extract_high_activation_tokens(feature_data, r_act_ratio)
                            
                            target_features.append({
                                'feature_id': feature_id,  # Keep as int
                                'high_activation_tokens': high_activation_token_strs  # Sorted token strings
                            })
                            
                    except Exception as e:
                        tqdm.write(f"Error processing feature {feature_id}: {e}")
                        continue
                
                if not target_features:
                    tqdm.write(f"No target type features found for {model_name} {layer_type} layer {layer_idx}")
                    continue
                
                # Step 2: Find interference features
                interference_features = []
                
                try:
                    # Get decoder weights and semantic embeddings
                    decoder_weights = get_sae_decoder_weights_from_local(model_name, layer_type, layer_idx)
                    explanation_embeddings_dict = get_feature_explanation_embeddings_by_layer(
                        model_name, layer_type, layer_idx
                    )
                    
                    # Get common feature indices
                    decoder_feature_ids = set(range(decoder_weights.shape[0]))
                    embedding_feature_ids = set(explanation_embeddings_dict.keys())
                    common_feature_ids = sorted(list(
                        decoder_feature_ids.intersection(embedding_feature_ids)
                    ))
                    
                    if len(common_feature_ids) == 0:
                        tqdm.write(f"Warning: No common features for interference analysis in {model_name} {layer_type} layer {layer_idx}")
                    else:
                        # Build interference matrix (decoder weights cosine similarity)
                        interference_matrix = decoder_weights[common_feature_ids].float()
                        interference_matrix_norm = F.normalize(interference_matrix, p=2, dim=1)
                        interference_cosine_matrix = torch.mm(
                            interference_matrix_norm, interference_matrix_norm.T
                        )
                        
                        # Clear intermediate matrices
                        del interference_matrix, interference_matrix_norm
                        gc.collect()
                        
                        # Build semantic similarity matrix
                        semantic_vectors = []
                        for feature_id in common_feature_ids:
                            if feature_id in explanation_embeddings_dict:
                                semantic_vectors.append(explanation_embeddings_dict[feature_id])
                            else:
                                tqdm.write(f"Warning: Missing embedding for feature {feature_id}")
                                semantic_vectors.append(torch.zeros_like(
                                    next(iter(explanation_embeddings_dict.values()))
                                ))
                        
                        semantic_matrix = torch.stack(semantic_vectors).float()
                        semantic_matrix_norm = F.normalize(semantic_matrix, p=2, dim=1)
                        semantic_cosine_matrix = torch.mm(
                            semantic_matrix_norm, semantic_matrix_norm.T
                        )
                        
                        # Clear intermediate matrices
                        del semantic_vectors, semantic_matrix, semantic_matrix_norm
                        gc.collect()
                        
                        # Create mapping from feature_id to matrix index
                        feature_id_to_idx = {fid: idx for idx, fid in enumerate(common_feature_ids)}
                        
                        # Pre-compute valid interference pairs mask
                        high_interference_mask = interference_cosine_matrix > interference_threshold
                        low_semantic_mask = semantic_cosine_matrix < semantic_threshold
                        valid_pairs_mask = high_interference_mask & low_semantic_mask
                        
                        # Remove diagonal elements (self-interference)
                        n_features = len(common_feature_ids)
                        diagonal_mask = torch.eye(n_features, dtype=torch.bool)
                        valid_pairs_mask = valid_pairs_mask & (~diagonal_mask)
                        
                        # Clear intermediate masks
                        del high_interference_mask, low_semantic_mask, diagonal_mask
                        gc.collect()
                        
                        # Find interference features for each target feature
                        for target_feature in target_features:
                            target_id = target_feature['feature_id']
                            
                            if target_id not in feature_id_to_idx:
                                continue
                            
                            target_idx = feature_id_to_idx[target_id]
                            
                            # Get valid candidate indices for this target feature (row-wise)
                            target_row_mask = valid_pairs_mask[target_idx]
                            candidate_indices = torch.where(target_row_mask)[0]
                            
                            for candidate_idx in candidate_indices:
                                candidate_id = common_feature_ids[candidate_idx]
                                
                                # Get interference value
                                interference_value = interference_cosine_matrix[target_idx, candidate_idx].item()
                                
                                # Check if candidate feature doesn't contain target type in high activations
                                candidate_feature_data = all_features.get(candidate_id)
                                
                                if not candidate_feature_data:
                                    continue
                                
                                try:
                                    # Use get_unique_top_act_tokens for candidate feature
                                    candidate_high_tokens = get_unique_top_act_tokens(
                                        candidate_feature_data,
                                        n=100,  # Get enough tokens to check
                                        r=r_act_ratio
                                    )
                                    
                                    # Check if candidate has target type in high activation tokens
                                    candidate_has_target_type = False
                                    for token in candidate_high_tokens:
                                        token_type_result = get_token_str_type(
                                            model_name, token, token_str_to_type_dict
                                        )
                                        if token_type_result == token_type:
                                            candidate_has_target_type = True
                                            break
                                    
                                    # If candidate doesn't have target type, it's an interference feature
                                    if not candidate_has_target_type:
                                        # Extract high activation token strings sorted by activation value
                                        candidate_high_activation_tokens = extract_high_activation_tokens(
                                            candidate_feature_data, r_act_ratio
                                        )
                                        
                                        # Check if this interference feature is already recorded
                                        existing_feature = None
                                        for if_feat in interference_features:
                                            if if_feat['feature_id'] == candidate_id:
                                                existing_feature = if_feat
                                                break
                                        
                                        if existing_feature:
                                            # Add interference relationship
                                            existing_feature['interferences'].append({
                                                'target_feature_id': target_id,
                                                'interference_value': float(interference_value)
                                            })
                                        else:
                                            # Create new interference feature record
                                            interference_features.append({
                                                'feature_id': candidate_id,
                                                'high_activation_tokens': candidate_high_activation_tokens,
                                                'interferences': [{
                                                    'target_feature_id': target_id,
                                                    'interference_value': float(interference_value)
                                                }]
                                            })
                                            
                                except Exception as e:
                                    tqdm.write(f"Error processing candidate feature {candidate_id}: {e}")
                                    continue
                        
                        # Clear all matrices and masks after processing this layer
                        del interference_cosine_matrix, semantic_cosine_matrix, valid_pairs_mask
                        del feature_id_to_idx, common_feature_ids
                        gc.collect()
                
                except Exception as e:
                    tqdm.write(f"Error in interference analysis for {model_name} {layer_type} layer {layer_idx}: {e}")
                    # Clear any remaining variables in case of error
                    locals_to_clear = [
                        'decoder_weights', 'explanation_embeddings_dict', 
                        'interference_matrix', 'interference_matrix_norm', 'interference_cosine_matrix',
                        'semantic_vectors', 'semantic_matrix', 'semantic_matrix_norm', 'semantic_cosine_matrix',
                        'high_interference_mask', 'low_semantic_mask', 'valid_pairs_mask', 'diagonal_mask'
                    ]
                    for var_name in locals_to_clear:
                        if var_name in locals():
                            del locals()[var_name]
                    gc.collect()
                
                # Store results for current layer
                results[layer_type][layer_idx] = {
                    'target_features': target_features,
                    'interference_features': interference_features,
                }
                
                # Clear feature data after processing each layer
                del all_features
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update progress info
                tqdm.write(f"Layer {layer_type}{layer_idx}: {len(target_features)} target features, "
                          f"{len(interference_features)} interference features")
                
            except Exception as e:
                tqdm.write(f"Error processing {model_name} {layer_type} layer {layer_idx}: {e}")
                # Clear any variables in case of layer processing error
                if 'all_features' in locals():
                    del all_features
                gc.collect()
                continue
    
    # Final cleanup
    if 'model' in locals():
        del model
    if 'token_id_to_type_dict' in locals():
        del token_id_to_type_dict
    if 'token_str_to_type_dict' in locals():
        del token_str_to_type_dict
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results