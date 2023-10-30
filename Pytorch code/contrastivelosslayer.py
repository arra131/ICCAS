import torch
import torch.nn.functional as F   #for 'normalize' method

def nt_xent_loss(out, out_aug, batch_size, hidden_norm = False, temperature = 1.0):
    
    if hidden_norm:   #to ensure unit norm
        out = F.normalize(out, dim=-1, p=2)   #L2(p) normalisation along last dimension(-1)
        out_aug = F.normalize(out_aug, dim=-1, p=2)

    INF = 1e9  #similar to +ve infinity
    device = out.device #device on which tensor out is currently located - to ensure all tensors are operating on same device 

    labels = torch.eye(batch_size * 2).to(device)   #square identity matrix to create +ve and -ve pairs for contrastive loss
    masks = torch.eye(batch_size).to(device)   #square identity matrix to mask similarity score - to separate +ve and -ve pairs
    masksINF = masks * INF   #masked version of masks where positive elements are replaced with INF

    logits_aa = torch.matmul(out, out.t()) / temperature   #similarity scores (logits) between embeddings from the same set, out
    logits_bb = torch.matmul(out_aug, out_aug.t()) / temperature   #similarity scores (logits) between embeddings from the same set, out_aug

    logits_aa = logits_aa - masksINF
    logits_bb = logits_bb - masksINF

    logits_ab = torch.matmul(out, out_aug.t()) / temperature   #similarity scores between embeddings from different sets out and out_aug
    logits_ba = torch.matmul(out_aug, out.t()) / temperature   #similarity scores between embeddings from different sets out_aug and out

    logits = torch.cat([torch.cat([logits_ab, logits_aa], dim=1), torch.cat([logits_ba, logits_bb], dim=1)], dim=0)   #to create a single tensor
    loss = F.cross_entropy(logits, torch.arange(batch_size * 2).to(device))

    return loss
