# mlcore/losses.py
import torch
import torch.nn.functional as F

def info_nce_two_view(emb_a, emb_b, temp=0.07):
    """
    emb_a, emb_b: (B, C) normalized embeddings from two augmentations of same items.
    """
    sim = emb_a @ emb_b.t() / temp
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_ab = F.cross_entropy(sim, labels)
    loss_ba = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_ab + loss_ba)
