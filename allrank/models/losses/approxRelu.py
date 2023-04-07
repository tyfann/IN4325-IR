import torch
import numpy as np

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS

class GumbelSampler(object):
    """Random sampler for sampling gumbel distributed logits."""

    def __init__(self, name=None, sample_size=8, temperature=1.0, seed=None):
        """Constructor."""
        self._name = name
        self._sample_size = sample_size
        self._temperature = temperature
        self._seed = seed

    def sample_gumbel(self, shape, eps=1e-20, seed=None):
        if seed:
            torch.manual_seed(seed)
        u = torch.rand(shape).cuda()
        return -torch.log(-torch.log(u + eps) + eps)

    def sample(self, labels, logits, weights=None):
        """Samples scores from Concrete(logits).
        Args:
          labels: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size]
            same as `logits`, representing graded relevance. Or in the diversity
            tasks, a `Tensor` (or `RaggedTensor`) with shape [batch_size, list_size,
            subtopic_size]. Each value represents relevance to a subtopic, 1 for
            relevent subtopic, 0 for irrelevant, and -1 for paddings. When the
            actual subtopic number of a query is smaller than the `subtopic_size`,
            `labels` will be padded to `subtopic_size` with -1.
          logits: A `Tensor` or `RaggedTensor` with shape [batch_size, list_size].
            Each value is the ranking score of the corresponding item.
          weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
            weights, or a `Tensor` or `RaggedTensor` with shape [batch_size,
            list_size] for item-wise weights. If None, the weight of a list in the
            mini-batch is set to the sum of the labels of the items in that list.
        Returns:
          A tuple of expanded labels, logits, and weights where the first dimension
          is now batch_size * sample_size. Logit Tensors are sampled from
          Concrete(logits) while labels and weights are simply tiled so the
          resulting
          Tensor has the updated dimensions.
        """
        batch_size = labels.size()[0]
        list_size = labels.size()[1]

        # Expand labels.
        expanded_labels = torch.unsqueeze(labels, 1)
        expanded_labels = torch.tile(expanded_labels, (1,self._sample_size,1))
        expanded_labels = torch.reshape(expanded_labels, (batch_size*self._sample_size, list_size))

        # Sample logits from Concrete(logits).
        sampled_logits = torch.unsqueeze(logits, 1)
        sampled_logits = torch.tile(sampled_logits, (1,self._sample_size,1))
        sampled_logits += self.sample_gumbel(
            [batch_size, self._sample_size, list_size], seed=self._seed)
        sampled_logits = torch.reshape(sampled_logits, (batch_size*self._sample_size, list_size))

        is_label_valid = (expanded_labels >= 0)
        if is_label_valid.dim() > 2:
            is_label_valid = torch.any(is_label_valid, dim=-1)
        sampled_logits = torch.where(
          is_label_valid, sampled_logits / self._temperature,
          np.log(1e-20) * torch.ones_like(sampled_logits))
        sampled_logits = torch.log(torch.nn.functional.softmax(sampled_logits,1) + 1e-20)

        expanded_weights = weights
        if expanded_weights is not None:
            true_fn = lambda: torch.unsqueeze(torch.unsqueeze(expanded_weights, 1), 1)
            false_fn = lambda: torch.unsqueeze(expanded_weights, 1)
            expanded_weights = torch.where(
                condition=torch.equal(torch.dim(expanded_weights), 1),
                input=true_fn,
                other=false_fn)
            expanded_weights = torch.tile(expanded_weights, (1,self._sample_size,1))
            expanded_weights = torch.reshape(expanded_weights, (batch_size*self._sample_size, -1))

        return expanded_labels, sampled_logits, expanded_weights

def approxRELULoss(y_pred, y_true, alpha = 1.0, temp=0.1, sample_size=8, gumbel_temp=1.0):
    gumbel = GumbelSampler(sample_size=sample_size, temperature=gumbel_temp)
    # batch, t= output.shape
    # dim = 1
    # output = output.reshape(-1,dim)
    # y = y.reshape(-1,dim)
    gbl_labels, gbl_logits, gbl_weights = gumbel.sample(y_true, y_pred)
    return approxNDCGLoss(gbl_logits, gbl_labels, alpha=alpha)



def approxNDCGLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)
