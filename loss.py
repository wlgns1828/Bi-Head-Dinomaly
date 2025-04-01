import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logit, target):
        """
        logit: (B, 1, H, W) - 모델의 로짓 출력
        target: (B, 1, H, W) - 정답 (0 또는 1)
        """
        # BCE Loss 계산 (PyTorch는 logit을 그대로 받음)
        BCE_loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')

        # 확률값 변환 (pt) -> 정답 class에 대한 확률
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(logit) if target == 1, else 1 - sigmoid(logit)

        # alpha를 정답(target)에 따라 다르게 적용
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Focal Loss 적용
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        # 손실 계산 방식 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLoss_smoothl1(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss_smoothl1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logit, target):
        """
        logit: (B, 1, H, W) - 모델의 로짓 출력
        target: (B, 1, H, W) - 정답 (0 또는 1)
        """

        smooth_l1_loss = F.smooth_l1_loss(logit, target, reduction='none')

        pt = torch.exp(-smooth_l1_loss)  


        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)


        focal_loss = alpha_t * (1 - pt) ** self.gamma * smooth_l1_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


