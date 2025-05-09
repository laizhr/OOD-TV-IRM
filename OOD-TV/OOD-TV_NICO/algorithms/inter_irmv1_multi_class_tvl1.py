from .model import InferEnvMultiClass
import torch
from torch import optim, nn, autograd
from torchsummary import summary


class Infer_Irmv1_Multi_Class_TVL1:

    def __init__(self, flags, dp):
        if flags.dataset == "logit_z":
            infer_env = InferEnvMultiClass(
                flags, z_dim=1, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "celebaz_feature":
            infer_env = InferEnvMultiClass(
                flags, z_dim=7, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "house_price":
            infer_env = InferEnvMultiClass(
                flags, z_dim=1, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "landcover":
            infer_env = InferEnvMultiClass(
                flags, z_dim=flags.aux_num, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "adult":
            infer_env = InferEnvMultiClass(
                flags, z_dim=flags.aux_num, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "mnist":
            infer_env = InferEnvMultiClass(
                flags, z_dim=flags.aux_num, class_num=flags.z_class_num
            ).cuda()
        elif flags.dataset == "nico":
            infer_env = InferEnvMultiClass(
                flags, z_dim=flags.aux_num, class_num=flags.z_class_num
            ).cuda()
        else:
            raise Exception
        self.optimizer_infer_env = optim.Adam(infer_env.parameters(), lr=0.001)
        self.flags = flags
        self.infer_env = infer_env

    def get_penalty_weight(self, mlp, mlp2):
        parameter = torch.tensor([], requires_grad=True).cuda()
        for v in mlp.parameters():
            if v.requires_grad:
                t = v.view(1, -1)
                parameter = torch.cat((parameter, v.view(1, -1)), dim=1)
        penalty_weight = mlp2(parameter).mean()
        return penalty_weight

    def __call__(
        self,
        norm,
        batch_data,
        step,
        mlp=None,
        mlp2=None,
        scale=None,
        mean_nll=None,
        **kwargs
    ):
        train_x, train_y, train_z, train_g, train_c, train_invnoise = batch_data
        normed_z = (train_z.float() - train_z.float().mean()) / train_z.float().std()
        train_logits = scale * mlp(train_x)
        if self.flags.dataset == "house_price":
            loss_fun = nn.MSELoss(reduction="none")
            train_nll = loss_fun(train_logits, train_y)
        elif self.flags.dataset == "landcover":  # Not full batch
            train_nll = mean_nll(train_logits, train_y, reduction="none")
        elif self.flags.dataset == "mnist":
            train_nll = nn.CrossEntropyLoss()(train_logits, train_g)
        elif self.flags.dataset == "nico":
            train_nll = nn.CrossEntropyLoss()(train_logits, train_g)
        else:
            train_nll = nn.functional.binary_cross_entropy_with_logits(
                train_logits, train_y, reduction="none"
            )

        infered_envs = self.infer_env(normed_z)
        # print(summary(self.infer_env, input_size=(1,3)))
        train_penalty = 0
        if (
            self.flags.dataset == "landcover"
        ):  # also for any dataset that is trained with mini-batch manner.
            train_nll = torch.unsqueeze(train_nll, 1)
            multi_loss = train_nll * infered_envs
            for i in range(multi_loss.shape[1]):
                grad1 = autograd.grad(
                    multi_loss[:, i][0::2].mean(),  # multi_loss.shape=[bs,2]
                    [scale],
                    create_graph=True,
                )[0]
                grad2 = autograd.grad(
                    multi_loss[:, i][1::2].mean(), [scale], create_graph=True
                )[0]
                # train_penalty += (grad1 * grad2).mean()
                train_penalty += (grad1 * grad2).mean().abs().sqrt()
            train_penalty = train_penalty**2
        else:
            multi_loss = (train_nll * infered_envs).mean(axis=0)
            for i in range(multi_loss.shape[0]):
                grad = autograd.grad(multi_loss[i], [scale], create_graph=True)[0]
                # train_penalty += grad ** 2
                if norm == "l1":
                    # print("runas:",norm)
                    train_penalty += grad.norm()
                else:
                    # print("runas:",norm)
                    train_penalty += grad.norm() ** 2
                    
            # train_penalty = train_penalty ** 2

        train_nll = train_nll.mean()

        penalty_weight = self.get_penalty_weight(mlp, mlp2)

        if step < self.flags.penalty_anneal_iters:
            # gradient ascend on infer_env net
            self.optimizer_infer_env.zero_grad()
            (-train_penalty).backward(retain_graph=True)
            self.optimizer_infer_env.step()
            # for param_mlp in self.infer_env.parameters():
            #     if torch.mean( param_mlp.grad)!=0:
            #         param_mlp.data = param_mlp.data + 0.001 * param_mlp.grad / param_mlp.grad.norm() / (step+1)
            #     else:
            #         break
        return train_nll, train_penalty, penalty_weight
