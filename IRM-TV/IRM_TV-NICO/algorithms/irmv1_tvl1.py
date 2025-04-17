# from .model import InferEnv
import torch
import pandas as pd

class IrmV1TVL1:
    
    def __init__(self, flags, dp):
        self.flags = flags
        # self.infer_env = infer_env

    
    def __call__(self, batch_data, step,  mlp=None, scale=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = torch.nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        else:
            train_nll = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
        ground_envs = train_g
        env1_loss = (train_nll * ground_envs).mean()
        env2_loss = (train_nll * (1 - ground_envs)).mean()
        grad1 = torch.autograd.grad(
            env1_loss,
            [scale],
            create_graph=True)[0]
        grad2 = torch.autograd.grad(
            env2_loss,
            [scale],
            create_graph=True)[0]
        # TV-L1
        train_penalty = (grad1.abs() + grad2.abs()) ** 2
        train_nll = train_nll.mean()

        return train_nll, train_penalty