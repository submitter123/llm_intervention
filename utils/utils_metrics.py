import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
def get_spearman_correlation(predictions, sim_rank):
    """
    compute the spearman correlation between two input arrays

    params:
        predictions: a tensor array of sorted model output 
            from high probs to low probs
        sim_rank: a tensor array of sorted token dicts 
            with respect to their embedding cosine similarity to a given token

    return:
        value of spearman correlation, and p_value
    """
    # Convert to numpy if inputs are tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(sim_rank):
        sim_rank = sim_rank.cpu().numpy()
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(predictions, sim_rank)
    
    return correlation, p_value

def get_kendall_tau_correlation(predictions, sim_rank):
    """
    compute the kendall tau correlation between two input arrays

    params:
        predictions: a tensor array of sorted model output 
            from high probs to low probs
        sim_rank: a tensor array of sorted token dicts 
            with respect to their embedding cosine similarity to a given token

    return:
        value of kendall tau correlation, and p_value
    """
    # Convert to numpy if inputs are tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(sim_rank):
        sim_rank = sim_rank.cpu().numpy()
    
    # Calculate Kendall's tau correlation
    correlation, p_value = kendalltau(predictions, sim_rank)
    
    return correlation, p_value

def get_weighted_cosine_similarity(
    pred_token_ids, pred_token_probs, 
    target_token_ids, token_embed_mat, compare_nums
):
    """
    compute the weighted cosine similarity between model predictions
    and a target token set, based on the model's token embedding

    params:
        pred_token_ids: sorted token ids of model's output
        pred_token_probs: sorted token probabilities of model's output
        target_token_ids: sorted token ids of target set
        token_embed_mat: token embedding matrix
        compare_nums: number of top tokens to compare
        
    return:
        weighted similarity score (scalar)
    """

    # Get the top-k predicted token ids and their probabilities
    top_pred_ids = pred_token_ids[:compare_nums]
    top_pred_probs = pred_token_probs[:compare_nums]

    # Get the embeddings for the top predicted tokens and target tokens
    top_pred_embeds = token_embed_mat[top_pred_ids]  # [compare_nums, embed_dim]
    target_embeds = token_embed_mat[target_token_ids]  # [num_targets, embed_dim]
    
    # Normalize embeddings for cosine similarity
    top_pred_embeds_norm = F.normalize(top_pred_embeds, p=2, dim=1)  # [compare_nums, embed_dim]
    target_embeds_norm = F.normalize(target_embeds, p=2, dim=1)  # [num_targets, embed_dim]

    # Compute cosine similarities between all pred tokens and all target tokens
    # similarities[i, j] = cosine_sim(pred_token_i, target_token_j)
    similarities = torch.mm(top_pred_embeds_norm, target_embeds_norm.T)  # [compare_nums, num_targets]
    
    # For each predicted token, find the maximum similarity with any target token
    max_similarities = similarities.max(dim=1)[0]  # [compare_nums]
    
    # Compute weighted similarity: sum of (max_similarity * probability)
    weighted_sim = (max_similarities * top_pred_probs).sum()

    return weighted_sim

def get_weighted_overlap(
    pred_token_ids, pred_token_probs, target_token_ids, compare_nums
):
    """
    compute the weighted overlap between model predictions and a target token set

    params:
        pred_token_ids: sorted token ids of model's output
        pred_token_probs: sorted token probabilities of model's output
        target_token_ids: sorted token ids of target set
        compare_nums: number of top tokens to compare
        
    return:
        weighted overlap score (scalar)
    """
    
    # Ensure compare_nums doesn't exceed the length of predictions
    actual_compare_nums = min(compare_nums, len(pred_token_ids))
    
    # Get the top-k predicted token ids and their probabilities
    top_pred_ids = pred_token_ids[:actual_compare_nums]
    top_pred_probs = pred_token_probs[:actual_compare_nums]
    
    # Use torch isin to check which predicted tokens are in target set
    # isin returns a boolean tensor indicating membership
    overlap_mask = torch.isin(top_pred_ids, target_token_ids)
    
    # Calculate weighted overlap by summing probabilities where overlap_mask is True
    weighted_overlap = (top_pred_probs * overlap_mask).sum()
    
    return weighted_overlap.item()