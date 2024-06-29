import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict, Counter



class DD_weight(nn.Module):
    weight_dd = []

    def __init__(self, weight_init, beta):
        super(DD_weight, self).__init__()
        self.weight = weight_init
        self.beta = beta
        DD_weight.weight_dd = self.weight
        self.weight_dict = defaultdict(Counter)
        for i, weight in enumerate(weight_init):
            self.weight_dict[i] = weight

    def update(self, loss):
        for i, weight in enumerate(loss):
            if weight != 0:

                self.weight_dict[i] = self.beta * self.weight_dict[i] + weight
                loss[i] = self.weight_dict[i]
            else:
                loss[i] = self.weight_dict[i] * (1 + self.beta)
        return loss

    def forward(self, pred, target):
        with torch.no_grad():
            pred = F.log_softmax(pred, dim=-1)
            loss = torch.zeros(self.weight.shape).type_as(pred)
            # old_weight = self.weight - torch.nn.functional.normalize(torch.ones(loss.shape[0]).type_as(self.weight), p=2, dim=-1)
            for i in range(self.weight.shape[0]):
                index = torch.where(target == i)
                if len(index[0]) != 0:
                    # loss[i] = -pred[index][:, i].sum() / len(index[0]) #soft
                    loss[i] = int((torch.argmax(pred[index], dim=-1) != i).sum()) / int(len(index[0]))  # hard loss

            weight = self.update(loss)
        return weight


class DDLoss(nn.Module):
    def __init__(self, beta, weight_init):
        super(DDLoss, self).__init__()
        self.weight_cal = DD_weight(weight_init, beta)

    def forward(self, logits, target, *args):
        weight = self.weight_cal(logits, target.long())
        self.criterion_loss = nn.CrossEntropyLoss(weight=weight, reduction='sum')

        loss = self.criterion_rel_loss(logits, target.long())
        loss = (loss / len(logits))
        return loss





