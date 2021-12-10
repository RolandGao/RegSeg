import torch
import torch.nn as nn

class BootstrappedCE(nn.Module):
    def __init__(self, min_K, loss_th, ignore_index):
        super().__init__()
        self.K = min_K
        self.threshold = loss_th
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, labels):
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)

        mask=(pixel_losses>self.threshold)
        if torch.sum(mask).item()>self.K:
            pixel_losses=pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, self.K)
        return pixel_losses.mean()
if __name__=='__main__':
    pass
