from .model import InferEnv
import torch
import pandas as pd
class InferIrmV1TVL1:
    
    def __init__(self, flags, dp):
        if flags.dataset == "logit_z":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "celebaz_feature":
            infer_env = InferEnv(flags, z_dim=7).cuda()
        elif flags.dataset == "house_price":
            infer_env = InferEnv(flags, z_dim=1).cuda()
        elif flags.dataset == "logit_2z":
            infer_env = InferEnv(flags, z_dim=flags.z_dim).cuda()
        self.flags = flags
        self.infer_env = infer_env
        self.optimizer_infer_env = torch.optim.Adam(infer_env.parameters(), lr=0.001) 
    
    def get_penalty_weight(self,mlp,mlp2):
        parameter = torch.tensor([], requires_grad=True).cuda()
        for v in mlp.parameters():
            if v.requires_grad:
                t = v.view(1, -1)
                parameter = torch.cat((parameter, v.view(1, -1)), dim=1)
        penalty_weight = mlp2(parameter).mean()
        return penalty_weight
    
    def __call__(self, batch_data, step,  mlp=None,mlp2=None, scale=None, **kwargs):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = torch.nn.MSELoss(reduction='none')
            train_nll = loss_fun(train_logits, train_y)
        else:
            train_nll = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, train_y, reduction="none")
        infered_envs = self.infer_env(normed_z)
        env1_loss = (train_nll * infered_envs).mean()
        env2_loss = (train_nll * (1 - infered_envs)).mean()
        grad1 = torch.autograd.grad(
            env1_loss,
            [scale],
            create_graph=True)[0]
        grad2 = torch.autograd.grad(
            env2_loss,
            [scale],
            create_graph=True)[0]
        ## edited by z lai
        #train_penalty = grad1 ** 2 + grad2 ** 2
        #train_penalty = infered_envs.mean() * grad1.abs() + (1 - infered_envs).mean() * grad2.abs()
        train_penalty = (grad1.abs() + grad2.abs()) ** 2
        #print(infered_envs.size())
        #print(train_penalty.size())
        #train_penalty = infered_envs * grad1.abs() + (1 - infered_envs) * grad2.abs()
        train_nll = train_nll.mean()
        
        penalty_weight=self.get_penalty_weight(mlp,mlp2)
        
        if step < self.flags.penalty_anneal_iters:
            # gradient ascend on infer_env net
            # self.optimizer_infer_env.zero_grad()
            (-train_penalty).backward(retain_graph=True)
            for param_mlp in self.infer_env.parameters():
                if torch.mean( param_mlp.grad)!=0:
                    param_mlp.data = param_mlp.data + 0.001 * param_mlp.grad / param_mlp.grad.norm() / (step+1)
                else:
                    break

        return train_nll, train_penalty,penalty_weight