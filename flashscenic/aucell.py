import torch
import numpy as np

def get_aucell(exp_array, adj_array,
                          k=50, auc_threshold=0.05,
                          device='cuda', batch_size=32,
                          seed=None):
    """
    Fully vectorized pySCENIC-equivalent AUCell calculation.

    Uses the actual number of target genes per TF/regulon (not fixed k) to match
    pySCENIC behavior. When adj_array has weighted entries, uses weights for AUC.

    Args:
        exp_array (np.ndarray): Expression matrix (n_cells x n_genes)
        adj_array (np.ndarray): Adjacency matrix (n_tfs x n_genes), can be binary or weighted
        k (int): Max target genes per TF (pads shorter regulons). Default is 50.
        auc_threshold (float): Fraction of genome for AUC calculation. Default is 0.05.
        device (str): Device, 'cpu' or 'cuda'. Default is 'cuda'.
        batch_size (int): Batch size for processing cells. Default is 32.
        seed (int): Random seed for tie-breaking. Default is None.

    Returns:
        np.ndarray: AUCell scores matrix of shape (n_cells, n_TFs)
    """
    n_cells, n_genes = exp_array.shape
    n_tfs = adj_array.shape[0]

    # Calculate rank cutoff (matches pySCENIC's derive_rank_cutoff)
    rank_cutoff = max(1, int(round(auc_threshold * n_genes)) - 1)

    adj_tensor = torch.tensor(adj_array, device=device, dtype=torch.float32)

    # Count actual target genes per TF (non-zero entries)
    n_targets_per_tf = (adj_tensor > 0).sum(dim=1)  # (n_tfs,)

    # Clamp k to actual number of targets available
    k_actual = min(k, n_genes)

    # Get top k target genes and their weights for each TF
    topk_weights, topk_adj_idx = adj_tensor.topk(k_actual, dim=1)  # (n_tfs, k_actual)

    # Create a mask for real target genes (weight > 0) vs padding
    target_mask = topk_weights > 0  # (n_tfs, k_actual)

    # Compute per-TF weight sums and max_auc for proper normalization
    # pySCENIC: max_auc = (rank_cutoff + 1) * sum(weights)
    weight_sums = topk_weights.sum(dim=1)  # (n_tfs,)
    # For binary adj (all weights 1.0), this equals n_targets_per_tf
    max_aucs = (rank_cutoff + 1) * weight_sums  # (n_tfs,)

    exp_tensor = torch.tensor(exp_array, device=device, dtype=torch.float32)
    all_aucs = []

    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_exp = exp_tensor[i:min(i+batch_size, n_cells), :]
            batch_size_actual = batch_exp.shape[0]

            # Add noise for tie-breaking
            if seed is not None:
                torch.manual_seed(seed + i)
            noise = torch.rand_like(batch_exp) * 1e-10
            batch_exp_noisy = batch_exp + noise

            # Get rankings (descending order, 0-indexed)
            order = torch.argsort(-batch_exp_noisy, dim=1)
            rankings = torch.argsort(order, dim=1)  # (batch, n_genes)

            # Gather target gene rankings for all TFs at once
            topk_adj_idx_expanded = topk_adj_idx.unsqueeze(0).expand(batch_size_actual, -1, -1)

            target_rankings = torch.gather(
                rankings.unsqueeze(1).expand(-1, n_tfs, -1),  # (batch, n_tfs, n_genes)
                dim=2,
                index=topk_adj_idx_expanded
            )  # (batch, n_tfs, k_actual)

            # Compute AUC for all TFs at once
            batch_aucs = _compute_auc(
                target_rankings, topk_weights, target_mask,
                rank_cutoff, max_aucs
            )
            all_aucs.append(batch_aucs.cpu().numpy())

    return np.concatenate(all_aucs, axis=0)


def _compute_auc(target_rankings, weights, target_mask, rank_cutoff, max_aucs):
    """
    Vectorized weighted AUC computation matching pySCENIC's weighted_auc1d.

    Args:
        target_rankings: (batch_size, n_tfs, k) tensor of gene rankings
        weights: (n_tfs, k) tensor of gene weights
        target_mask: (n_tfs, k) bool tensor, True for real target genes
        rank_cutoff: int
        max_aucs: (n_tfs,) tensor of per-TF max AUC values

    Returns:
        (batch_size, n_tfs) tensor of AUC values
    """
    batch_size, n_tfs, k_val = target_rankings.shape
    device = target_rankings.device

    # Expand weights and target_mask to batch dimension
    weights_expanded = weights.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_tfs, k)
    target_mask_expanded = target_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_tfs, k)

    # Valid genes: real target genes with rank < rank_cutoff
    # (matches pySCENIC: filter_idx = ranking < rank_cutoff, applied only to regulon genes)
    rank_valid = target_rankings < rank_cutoff  # (batch, n_tfs, k)
    valid_mask = rank_valid & target_mask_expanded  # only real target genes within cutoff

    # Replace invalid entries with rank_cutoff (will sort to end, contribute 0)
    target_rankings_float = target_rankings.float()
    target_rankings_masked = torch.where(valid_mask, target_rankings_float,
                                         torch.full_like(target_rankings_float, float(rank_cutoff)))

    # Replace invalid weights with 0
    weights_masked = torch.where(valid_mask, weights_expanded,
                                 torch.zeros_like(weights_expanded))

    # Sort by ranking (invalid entries go to end)
    sorted_indices = torch.argsort(target_rankings_masked, dim=2)
    sorted_rankings = torch.gather(target_rankings_masked, 2, sorted_indices)  # (batch, n_tfs, k)
    sorted_weights = torch.gather(weights_masked, 2, sorted_indices)  # (batch, n_tfs, k)

    # Cumulative weights (matches pySCENIC: y = weights[sort_idx].cumsum())
    cumsum_weights = sorted_weights.cumsum(dim=2)  # (batch, n_tfs, k)

    # Append rank_cutoff to sorted_rankings for diff computation
    cutoff_tensor = torch.full((batch_size, n_tfs, 1), rank_cutoff, device=device, dtype=torch.float32)
    sorted_with_cutoff = torch.cat([sorted_rankings, cutoff_tensor], dim=2)  # (batch, n_tfs, k+1)

    # Compute rank differences: rank[i+1] - rank[i]
    # (matches pySCENIC: np.diff(x))
    rank_diffs = sorted_with_cutoff[:, :, 1:] - sorted_with_cutoff[:, :, :-1]  # (batch, n_tfs, k)

    # AUC = sum(diff(x) * cumsum(y)) / max_auc
    # (matches pySCENIC: np.sum(np.diff(x) * y) / max_auc)
    auc_contrib = rank_diffs * cumsum_weights

    # Only count contributions from valid positions
    valid_sorted = torch.gather(valid_mask.float(), 2, sorted_indices)
    auc_contrib = auc_contrib * valid_sorted

    # Sum and normalize per-TF
    raw_aucs = auc_contrib.sum(dim=2)  # (batch, n_tfs)
    max_aucs_expanded = max_aucs.unsqueeze(0).expand(batch_size, -1)  # (batch, n_tfs)

    # Normalize, avoiding division by zero for TFs with no targets
    aucs = torch.where(max_aucs_expanded > 0,
                       raw_aucs / max_aucs_expanded,
                       torch.zeros_like(raw_aucs))

    return aucs
