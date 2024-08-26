import torch
import torch.nn as nn
import math

from data.shapenet_pc_loader import ShapenetPC


class CustomWeightedDenominatorCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomWeightedDenominatorCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets, weights):
        """
        Compute the custom weighted multiclass cross entropy loss with weights in the denominator.

        Parameters:
        - predictions: A tensor of shape (n_samples, n_classes) with the raw class scores (logits).
        - targets: A tensor of shape (n_samples) with the target class indices.
        - weights: A tensor of shape (n_samples, n_classes) with the weights for each class for each sample.

        Returns:
        - loss: The computed loss value.
        """

        # Applying weights to the exponentials in the denominator
        weighted_exp_sum = torch.sum(weights * torch.exp(predictions), dim=1, keepdim=True)

        # Compute log softmax with weighted denominator
        log_softmax_weighted = predictions - torch.log(weighted_exp_sum + 1e-9)  # Adding epsilon to avoid log(0)

        # Gather the log softmax probabilities for the correct classes
        target_log_probs = torch.gather(log_softmax_weighted, 1, targets.unsqueeze(1))

        # Compute the negative log likelihood loss
        loss = -torch.mean(target_log_probs)

        return loss


def adjust_learning_rate(optimizer, epoch, step, total_steps, warmup_epochs, total_epochs, initial_lr, final_lr):
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr + (final_lr - initial_lr) * (step + epoch * total_steps) / (warmup_epochs * total_steps)
    else:
        # Cosine annealing
        progress = ((epoch - warmup_epochs) * total_steps + step) / ((total_epochs - warmup_epochs) * total_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        decayed_lr = (final_lr - initial_lr) * cosine_decay + initial_lr
        lr = decayed_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_datasets(similarity, point_cloud_root, embeddings_root, labels_root, labels_text_root):
    if similarity == 'I2I':
        if point_cloud_root is None or embeddings_root is None or labels_root is None:
            raise ValueError('Point cloud root, embeddings root and labels root must be provided for I2I similarity')
        dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_root, split='train', remove_extra_cat=True)
        val_dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_root, split='val', remove_extra_cat=True)
        return dataset, val_dataset, None, None
    elif similarity == '(I2L)2':
        if point_cloud_root is None or embeddings_root is None or labels_text_root is None:
            raise ValueError('Point cloud root, embeddings root and labels text root must be provided for (I2L)2 similarity')
        dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_text_root, split='train', annotation_type='text_sims')
        val_dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_text_root, split='val',
                                 annotation_type='text_sims')
        return None, None, dataset, val_dataset
    elif similarity == 'AVG':
        if point_cloud_root is None or embeddings_root is None or labels_root is None or labels_text_root is None:
            raise ValueError('Point cloud root, embeddings root, labels root and labels text root must be provided for AVG similarity')
        dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_root, split='train', remove_extra_cat=True)
        dataset_texts = ShapenetPC(point_cloud_root, embeddings_root, labels_text_root, split='train',
                                   annotation_type='text_sims')
        val_dataset = ShapenetPC(point_cloud_root, embeddings_root, labels_root, split='val', remove_extra_cat=True)
        val_dataset_texts = ShapenetPC(point_cloud_root, embeddings_root, labels_text_root, split='val',
                                       annotation_type='text_sims')
        return dataset, dataset_texts, val_dataset, val_dataset_texts


def get_weights(similarity, dataset, dataset_texts, cat, idx, split, negative_add=0.25):
    if similarity == 'I2I' or similarity == 'AVG':
        weights = dataset.get_weights(cat, idx, split, negative_add=negative_add).cuda()
    if similarity == '(I2L)2' or similarity == 'AVG':
        weights_text = dataset_texts.get_weights(cat, idx, split, negative_add=negative_add).cuda()
    if similarity == 'AVG':
        weights = (weights + weights_text) / 2

    return weights if similarity == 'I2I' or similarity == 'AVG' else weights_text