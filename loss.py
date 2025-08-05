import torch
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.5):
    B = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix
    sim = torch.matmul(z, z.T)  # [2B, 2B]
    sim = sim / temperature

    # Positive pairs: (i, i + B) and (i + B, i)
    pos_mask = torch.eye(B, device=z.device)
    pos = torch.cat([pos_mask.roll(shifts=i, dims=0) for i in range(B)], dim=1)
    pos = pos[:2*B, :2*B]

    # Remove self-similarity
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = torch.matmul(z, z.T).clamp(min=-1.0, max=1.0) / temperature

    # For each positive pair: log-softmax over 2B-1 negatives
    pos_sim = torch.sum(z_i * z_j, dim=-1) / temperature  # [B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2B]

    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()
