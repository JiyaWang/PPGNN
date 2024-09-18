import torch
import torch.nn as nn

class GraphLoss(nn.Module):
    def __init__(self):
        super(GraphLoss,self).__init__()
        self.avg_accuracy = None

    def reset_avg_accuracy(self):
        self.avg_accuracy = None

    def forward(self,train_pred,train_lab,logprobs):
        #logprobs:b*n_train*k*1
        #train_pred:b*n_train*c
        #train_lab:b*n_train
        corr_pred = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().detach()

        if self.avg_accuracy is None:
            self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

        point_w = (
            self.avg_accuracy - corr_pred
        )  # *(1*corr_pred + self.k*(1-corr_pred))
        graph_loss = point_w * logprobs.exp().mean([-1, -2])

        self.avg_accuracy = (self.avg_accuracy.to(corr_pred.device) * 0.9 + 0.1 * corr_pred)

        return graph_loss.mean()

