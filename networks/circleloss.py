import torch


class CircleLoss(torch.nn.Module):
    def __init__(self, args):
        super(CircleLoss, self).__init__()
        self.B = args.B
        self.m = args.m
        self.gamma = args.gamma
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.op = 1 + self.m
        self.on = -self.m
        self.delta_p = 1 - self.m
        self.delta_n = self.m

    def forward(self, mat, pos_mask, neg_mask):
        pos_mask_idx = torch.where(pos_mask==1)
        neg_mask_idx = torch.where(neg_mask==1)
        anchor_positive = mat[pos_mask_idx].reshape(self.B, -1)
        anchor_negative = mat[neg_mask_idx].reshape(self.B, -1)
        eq_sp = torch.zeros_like(anchor_positive)
        eq_sn = torch.zeros_like(anchor_negative)
        N = eq_sp.shape[1]
        M = eq_sn.shape[1]
        loss = torch.zeros(self.B)

        eq_sp = (
            -self.gamma
            * torch.relu(self.op - anchor_positive.detach())
            * (anchor_positive - self.delta_p)
        )
        eq_sn = (
            self.gamma
            * torch.relu(anchor_negative.detach() - self.on)
            * (anchor_negative - self.delta_n)
        )
        for i in range(self.B):
            neg_reshape = eq_sn[i].repeat(N,1)
            pos_reshape = eq_sp[i].repeat(M,1).t()
            expsum = torch.sum(torch.exp(neg_reshape + pos_reshape))
            loss[i] = torch.log(1+expsum)

        return loss
