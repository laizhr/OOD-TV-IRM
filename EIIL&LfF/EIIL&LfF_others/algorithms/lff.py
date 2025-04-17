import torch
from torch import optim, nn


class LfF:
    def __init__(self, flags, dp):
        self.aux_mlp = dp.fetch_mlp().cuda()
        if flags.opt == "adam":
            self.aux_optimizer = optim.Adam(self.aux_mlp.parameters(), lr=flags.lr)
        elif flags.opt == "sgd":
            self.aux_optimizer = optim.SGD(
                self.aux_mlp.parameters(), momentum=0.9, lr=flags.lr
            )
        else:
            raise Exception
        self.flags = flags

    def __call__(self, batch_data, step, mlp=None, scale=None, mean_nll=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        aux_pred = self.aux_mlp(train_x)  # nan
        if aux_pred.shape[1] < 2:
            softmax = nn.Sigmoid().cuda()
            aux_pred_softmax = softmax(aux_pred)
            py = torch.mul(
                aux_pred_softmax**train_y, (1 - aux_pred_softmax) ** (1 - train_y)
            )
            loss_function = torch.nn.MSELoss(reduction='none')
        else:
            softmax = nn.Softmax(dim=1).cuda()
            aux_pred_softmax = softmax(aux_pred)
            train_y_one_hot = torch.stack(
                [train_y == i for i in range(6)], dim=1
            ).float()  # one-hot
            py = torch.diag(torch.mm(aux_pred_softmax, train_y_one_hot.squeeze().t()))
            train_y = train_y.long()
            loss_function = torch.nn.MSELoss(reduction='none')

        aux_train_loss = ((1 - torch.pow(py, 0.7)) / 0.7).mean()
        self.aux_optimizer.zero_grad()
        aux_train_loss.backward()  # GCE LOSS
        self.aux_optimizer.step()

        pred = mlp(train_x)
        if self.flags.dataset=="landcover":
            aux_ce_loss = loss_function(aux_pred, torch.nn.functional.one_hot(train_y,6).float())
            ce_loss = loss_function(pred, torch.nn.functional.one_hot(train_y,6).float())
        else:
            aux_ce_loss = loss_function(aux_pred, train_y)
            ce_loss = loss_function(pred, train_y)
       
        weight = aux_ce_loss / (
            aux_ce_loss + ce_loss
        )  # aux_ce_loss: softmax of bias model; ce_loss: softmax of debiased model
        loss = (weight.detach() * ce_loss).mean()  # W * CE LOSS
        train_nll = loss
        # train_penalty = torch.zeros_like(train_nll).cuda()
        train_penalty = torch.tensor(0.1).cuda()
        return train_nll, train_penalty
