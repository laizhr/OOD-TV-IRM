import argparse
import datetime
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pdb
import math
import numpy as np
import torch
from torchvision import datasets
from data import CowCamels
from data import AntiReg
import os
import sys
from torch import nn, optim, autograd
import random
from landcover import init_dataset, init_dataloader, initialize
from model import MLP2Layer
import torch
from utils import mean_accuracy_reg
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from innout.models.cnn1d import Net
import torch.nn.functional as F


class CustomMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="training"):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 假设你的数据存储在root_dir下的training和testing文件夹中
        split_dir = os.path.join(root_dir, split)
        for label in os.listdir(split_dir):

            label_dir = os.path.join(split_dir, label)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                self.images.append(image_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)  # 转换为灰度图像
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class MNIST(object):
    def __init__(self, flags):
        super(MNIST, self).__init__()
        config = dict()
        # config['train_transforms'] = [{'classname': 'innout.datasets.transforms.LambdaTransform', 'args': {'function_path': 'innout.datasets.transforms.to_tensor'}}, {'classname': 'innout.datasets.transforms.LambdaTransform', 'args': {'function_path': 'innout.datasets.transforms.tensor_to_float'}}]
        # config['test_transforms'] = [{'classname': 'innout.datasets.transforms.LambdaTransform', 'args': {'function_path': 'innout.datasets.transforms.to_tensor'}}, {'classname': 'innout.datasets.transforms.LambdaTransform', 'args': {'function_path': 'innout.datasets.transforms.tensor_to_float'}}]
        # config['dataset'] = {'classname': 'innout.datasets.mnist.ColoredMnist', 'args': {'root': '/u/nlp/data/landcover/timeseries_by_box_v2', 'cache_path': 'datasets/landcover.pkl', 'include_NDVI': True, 'include_ERA5': True, 'standardize': True, 'shuffle_domains': True, 'seed': 1, 'use_cache': True, 'use_unlabeled_id': False, 'use_unlabeled_ood': False, 'unlabeled_prop': 0.9, 'pretrain': False, 'multitask': False}, 'train_args': {'split': 'nonafrica-train'}, 'eval_train_args': {'split': 'nonafrica-train'}, 'val_args': {'split': 'nonafrica-val'}, 'test_args': {'split': 'nonafrica-test'}, 'test2_args': {'split': 'africa'}}
        # config['model'] = {'classname': 'innout.models.cnn1d.CNN1D', 'args': {'in_channels': 8, 'output_size': 6}}
        # config['use_cuda'] = True
        # config['batch_size'] = flags.batch_size
        # self.data_num_train = flags.batch_size
        # config['eval_batch_size'] = flags.batch_size

        # # create dataset
        # train_dataset = init_dataset(config, 'train')
        # train_eval_dataset = init_dataset(config, 'train')
        # val_dataset = init_dataset(config, 'val', train_dataset)
        # test_dataset = init_dataset(config, 'test2', train_dataset)
        # print("landcover dataset loaded")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = CustomMNISTDataset(
            root_dir=".\\datasets\\Mnist", transform=transform
        )
        test_dataset = CustomMNISTDataset(
            root_dir=".\\datasets\\Mnist", transform=transform, split="testing"
        )
        train_size = int(0.8 * len(train_dataset))
        validate_size = len(train_dataset) - train_size
        train_dataset, validate_dataset = random_split(
            train_dataset, [train_size, validate_size]
        )

        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=64, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=validate_dataset, batch_size=64, shuffle=True
        )
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

        self.mlp = Net()

    def fetch_train(self):
        bs = next(iter(self.train_loader))
        train_x = bs[0]  # 假设数据已经是张量格式
        # train_y = bs[]
        train_z = torch.mean(train_x, dim=(2, 3))
        train_g = F.one_hot(bs[1], num_classes=10)
        train_g = train_g.float()
        # train_c = torch.unsqueeze(random_zero_one(train_y.shape[0]), 1)
        return (
            train_x.cuda(),
            # train_y.cuda(),
            None,
            train_z.cuda(),  # 移除lat_lon和climate
            train_g.cuda(),
            # train_c.cuda(),
            None,
            None,
        )

    def fetch_test(self):
        bs = next(iter(self.test_loader))
        test_x = bs[0]
        test_z = torch.mean(test_x, dim=(2, 3))
        test_g = F.one_hot(bs[1], num_classes=10).float()
        return (
            test_x.cuda(),
            # test_y.cuda(),
            None,
            test_z,  # 移除lat_lon和climate
            test_g.cuda(),
            # test_c.cuda(),
            None,
            None,
        )

    def fetch_val(self):
        bs = next(iter(self.val_loader))
        val_x = bs[0]
        val_z = torch.mean(val_z, dim=(2, 3))
        val_g = F.one_hot(bs[1], num_classes=10).float()
        return (
            val_x.cuda(),
            # val_y.cuda(),
            None,
            val_z,  # 移除lat_lon和climate
            val_g.cuda(),
            # val_c.cuda(),
            None,
            None,
        )

    def fetch_mlp(self):
        return self.mlp


def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()


def torch_xor(a, b):
    return (a - b).abs()


def concat_envs_z(con_envs):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat(
        [ig * torch.ones_like(env["labels"]) for ig, env in enumerate(con_envs)]
    )
    z_list = []
    accum = 0
    for i in range(len(con_envs)):
        env = con_envs[i]
        len_env = len(env["labels"])
        z_list.append(accum + torch.arange(len_env))
        accum += len_env
    con_z = torch.cat(z_list).view(-1, 1)
    con_c = torch.cat([env["color"] for env in con_envs])
    con_inv = torch.cat([env["noise"] for env in con_envs])
    return (
        con_x.cuda(),
        con_y.cuda(),
        con_z.cuda(),
        con_g.cuda(),
        con_c.cuda(),
        con_inv.cuda(),
    )


def concat_envs_2z(con_envs):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat(
        [ig * torch.ones_like(env["labels"]) for ig, env in enumerate(con_envs)]
    )
    z_list = []
    accum1 = 0
    accum2 = 0
    for i in range(len(con_envs)):
        env = con_envs[i]
        len_env = len(env["labels"])
        assert len_env == 2500
        row = i // 2
        col = i % 2
        z1 = col * 50 + torch.arange(50).repeat(50, 1).view(-1)
        z2 = row * 50 + torch.arange(50).view(-1, 1).repeat(1, 50).view(-1)
        z = torch.stack([z1, z2], axis=1)
        z_list.append(z)
    con_z = torch.cat(z_list)
    con_c = torch.cat([env["color"] for env in con_envs])
    con_inv = torch.cat([env["noise"] for env in con_envs])
    return (
        con_x.cuda(),
        con_y.cuda(),
        con_z.cuda(),
        con_g.cuda(),
        con_c.cuda(),
        con_inv.cuda(),
    )


def eval_acc_class(logits, labels, colors):
    acc = mean_accuracy_class(logits, labels)
    minacc = mean_accuracy_class(logits[colors != 1], labels[colors != 1])
    majacc = mean_accuracy_class(logits[colors == 1], labels[colors == 1])
    return acc, minacc, majacc


class MetaAcc(object):
    def __init__(self, env, acc_measure, acc_type="train"):
        self.env = env
        self.meta_list = []
        self.acc_measure = acc_measure
        self.acc_type = acc_type

    def clear(self):
        self.meta_list = []

    @property
    def acc_fields(self):
        return [self.acc_type + "_acc"] + [
            self.acc_type + "_e%s" % x for x in range(self.env)
        ]

    def process_batch(
        self,
        labels,
        logits,
    ):  # calculates data_num, acc, acc for e0, acc for e1
        batch_dict = {}
        data_num = len(labels)
        batch_dict.update({"data_num": data_num})

        acc = self.acc_measure(logits, labels)
        batch_dict.update({"acc": acc})

        # for e in range(self.env):
        #     env_locs = (g == e).resize(g.shape[0])  # shape=[bs]
        #     env_labels, env_logits, env_g = (
        #         labels[env_locs],
        #         logits[env_locs],
        #         g[env_locs],
        #     )  # labels.shape=[bs,1]; logits.shape=[bs, class_num]; g.shape=[bs, 1];
        #     env_data_num = len(env_labels)
        #     if env_data_num == 0:
        #         env_acc = 0
        #     else:
        #         env_acc = self.acc_measure(env_logits, env_labels)

        batch_dict.update({"acc_e%s" % 1: acc, "data_num_e%s" % 1: 1})

        self.meta_list.append(batch_dict)

    @property
    def meta_data_num(self):
        data_num_dict = {}
        data_num_dict.update({"data_num": sum([x["data_num"] for x in self.meta_list])})
        for e in range(self.env):
            data_num_dict.update(
                {
                    "data_num_e%s"
                    % e: sum([x["data_num_e%s" % e] for x in self.meta_list])
                }
            )
        return data_num_dict

    @property
    def meta_acc(self):
        # added by w wang
        if self.acc_measure.__name__ == mean_accuracy_reg.__name__:
            acc_worst = -1
        else:
            acc_worst = 1.0
        meta_data_num = self.meta_data_num
        acc_value = (
            1.0
            / meta_data_num["data_num"]
            * sum([x["acc"] * x["data_num"] for x in self.meta_list])
        )
        full_dict = {}
        full_dict.update(
            {
                self.acc_type
                + "_acc": 1.0
                / meta_data_num["data_num"]
                * sum([x["acc"] * x["data_num"] for x in self.meta_list])
            }
        )

        try:
            [full_dict[fd] for fd in self.acc_fields]
        except:
            raise Exception
        # added by w wang
        
        acc_env_worst = self.acc_type + "_acc_worst"
        if acc_worst >= acc_value:
            acc_worst = acc_value
        full_dict.update({acc_env_worst: acc_worst})
        return full_dict


def mean_accuracy_class(logits, y):
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def eval_acc_multi_class(logits, labels, colors):
    acc = mean_accuracy_multi_class(logits, labels)
    minacc = mean_accuracy_multi_class(
        logits[colors.view(-1) != 1], labels[colors.view(-1) != 1]
    )
    majacc = mean_accuracy_multi_class(
        logits[colors.view(-1) == 1], labels[colors.view(-1) == 1]
    )
    return acc, minacc, majacc


def mean_accuracy_multi_class(output, target):
    probs = torch.softmax(output, dim=1)
    winners = probs.argmax(dim=1)
    corrects = winners == target.view(-1)
    accuracy = corrects.sum().float() / float(corrects.size(0))
    return accuracy


def eval_acc_reg(logits, labels, colors):
    acc = mean_nll_reg(logits, labels)
    minacc = torch.tensor(0.0)
    majacc = torch.tensor(0.0)
    return acc, minacc, majacc


def make_one_logit_z(num, sp_ratio, noise_ratio, dim_inv, dim_spu):
    cc = CowCamels(
        dim_inv=dim_inv,
        dim_spu=dim_spu,
        n_envs=1,
        p=[sp_ratio],
        nr=[noise_ratio],
        s=[0.5],
    )
    inputs, outputs, colors, inv_noise = cc.sample(n=num, env="E0")
    return {
        "images": inputs,
        "labels": outputs,
        "color": colors,  # sp noise
        "noise": inv_noise,  # inv noise
    }


def make_logit_envs_z(flags):
    envs = []
    cons_ratio_float = [float(x) for x in flags.cons_ratio.split("_")]
    envs_num_train = flags.envs_num_train
    for i in range(envs_num_train):
        envs.append(
            make_one_logit_z(
                flags.data_num_train // envs_num_train,
                cons_ratio_float[i],
                flags.noise_ratio,
                flags.dim_inv,
                flags.dim_spu,
            )
        )
    envs_num_test = flags.envs_num_test
    for i in range(envs_num_test):
        envs.append(
            make_one_logit_z(
                flags.data_num_test // envs_num_test,
                cons_ratio_float[i + envs_num_train],
                flags.noise_ratio,
                flags.dim_inv,
                flags.dim_spu,
            )
        )
    return envs


def mean_nll_class(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_nll_multi_class(logits, y):
    nll = nn.CrossEntropyLoss()
    return nll(logits, y.view(-1).long())


def mean_nll_reg(logits, y):
    l2loss = nn.MSELoss()
    return l2loss(logits, y)


def mean_accuracy_reg(logits, y, colors=None):
    return mean_nll_reg(logits, y)


def pretty_print_ly(values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode="fixed")
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode="fixed")
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass


class LOGITZ(object):
    def __init__(self, flags):
        super(LOGITZ, self).__init__()
        self.flags = flags
        self.envs = make_logit_envs_z(flags)
        self.preprocess_data()
        self.feature_dim = self.train_x.shape[1]
        self.data_num_train = self.train_x.shape[0]

    def preprocess_data(self):
        (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            self.train_invnoise,
        ) = concat_envs_z(self.envs[: self.flags.envs_num_train])
        (
            self.test_x,
            self.test_y,
            self.test_z,
            self.test_g,
            self.test_c,
            self.test_invnoise,
        ) = concat_envs_z(self.envs[self.flags.envs_num_train :])

    def fetch_train(self):
        return (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            self.train_invnoise,
        )

    def fetch_test(self):
        return (
            self.test_x,
            self.test_y,
            self.test_z,
            self.test_g,
            self.test_c,
            self.test_invnoise,
        )


class LOGIT2Z(object):
    def __init__(self, flags):
        super(LOGIT2Z, self).__init__()
        self.flags = flags
        self.envs = make_logit_envs_z(flags)
        self.preprocess_data()
        self.feature_dim = self.train_x.shape[1]
        self.data_num_train = self.train_x.shape[0]
        self.z_dim = self.train_z.shape[1]

    def preprocess_data(self):
        (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            self.train_invnoise,
        ) = concat_envs_2z(self.envs[: self.flags.envs_num_train])
        (
            self.test_x,
            self.test_y,
            self.test_z,
            self.test_g,
            self.test_c,
            self.test_invnoise,
        ) = concat_envs_2z(self.envs[self.flags.envs_num_train :])

    def fetch_train(self):
        return (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            self.train_invnoise,
        )

    def fetch_test(self):
        return (
            self.test_x,
            self.test_y,
            self.test_z,
            self.test_g,
            self.test_c,
            self.test_invnoise,
        )


class CELEBAZ(LYDataProvider):
    def __init__(self, flags):
        super(CELEBAZ, self).__init__()
        self.flags = flags
        self.preprocess_data()

    def preprocess_data(self):
        from celeba_z import get_data_loader_sp

        (
            self.spd,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.train_data,
            self.val_data,
            self.test_data,
        ) = get_data_loader_sp(
            root_dir="/home/jzhangey/datasets/Spurious/data/celeba",
            target_name="Smiling",
            confounder_names="Male",
            auxilary_names=[
                "Young",
                "Blond_Hair",
                "Eyeglasses",
                "High_Cheekbones",
                "Big_Nose",
                "Bags_Under_Eyes",
                "Chubby",
            ],
            batch_size=100,
            train_num=50000,
            test_num=10000,
            cons_ratios=[0.99, 0.9, 0.1],
        )

    def fetch_train(self):
        try:
            batch_data = self.train_loader_iter.__next__()
        except:
            self.train_loader_iter = iter(self.train_loader)
            batch_data = self.train_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, z, g, sp = batch_data
        return x.float().cuda(), y.float().cuda(), z.float().cuda(), g, sp

    def fetch_test(self):
        try:
            batch_data = self.test_loader_iter.__next__()
        except:
            self.test_loader_iter = iter(self.test_loader)
            batch_data = self.test_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, z, g, sp = batch_data
        return x.float().cuda(), y.float().cuda(), z.float().cuda(), g, sp

    def test_batchs(self):
        return math.ceil(self.test_data.x_array.shape[0] / self.flags.batch_size)

    def train_batchs(self):
        return math.ceil(self.train_data.x_array.shape[0] / self.flags.batch_size)


class CELEBAZ_FEATURE(object):
    def __init__(self, flags):
        super(CELEBAZ_FEATURE, self).__init__()
        self.flags = flags
        self.preprocess_data()
        self.feature_dim = self.train_x.shape[1]
        self.data_num_train = self.train_x.shape[0]

    def preprocess_data(self):
        import pandas as pd

        train_num = 40000
        test_num = 20000

        def process_file(file_name):
            df = pd.read_csv(file_name)
            flds = ["x_", "y_", "z_", "g_", "sp_"]
            out_list = []
            for fld in flds:
                _names = [x for x in df.columns.tolist() if fld in x]
                if fld == "z_":
                    _names = _names[: self.flags.aux_num]
                out_list.append(torch.Tensor(df[_names].values).cuda())
            return tuple(out_list)

        train_file = (
            f"datasets/CelebA/train_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv"
        )
        # , self.train_invnoise
        self.train_x, self.train_y, self.train_z, self.train_g, self.train_c = (
            process_file(train_file)
        )
        test_file = f"datasets/CelebA/test_40000_0.999_0.8_20000_0.01_0.2_0.8_0.999.csv"
        print(train_file)
        self.test_x, self.test_y, self.test_z, self.test_g, self.test_c = process_file(
            test_file
        )

    def fetch_train(self):
        return (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            None,
        )

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_z, self.test_g, self.test_c, None

    def fetch_mlp(self):
        return MLP2Layer(self.flags, self.feature_dim, self.flags.hidden_dim).cuda()


class HousePrice(object):
    def __init__(self, flags):
        super(HousePrice, self).__init__()
        self.flags = flags
        self.preprocess_data()
        self.feature_dim = self.train_x.shape[1]
        self.data_num_train = self.train_x.shape[0]

    def preprocess_data(self):
        mypath = "datasets/house_data_precessed.csv"
        full_df = pd.read_csv(mypath)
        full_df["yr_built_norm"] = (
            full_df["yr_built"] - full_df["yr_built"].mean()
        ) / full_df["yr_built"].std()
        full_df["yr_renovated_norm"] = (
            full_df["yr_renovated"] - full_df["yr_renovated"].mean()
        ) / full_df["yr_renovated"].std()
        x_fields = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "grade",
            "sqft_above",
            "sqft_basement",
            "lat",
            "long",
            "sqft_living15",
            "sqft_lot15",
        ]
        full_df["g"] = (full_df["yr_built"] - 1900) // 10
        y_fields = ["price"]
        z_fields = ["yr_built_norm"]  # ,'yr_renovated'
        g_fields = ["g"]
        train_g = [0, 1, 2, 3, 4, 5, 6]
        test_g = [7, 8, 9, 10, 11]
        train_df = full_df[full_df.g.isin(train_g)]
        test_df = full_df[full_df.g.isin(test_g)]
        self.train_x = torch.Tensor(train_df[x_fields].values).cuda()
        self.train_y = torch.Tensor(train_df[y_fields].values).cuda()
        self.train_z = torch.Tensor(train_df[z_fields].values).cuda()
        self.train_g = torch.Tensor(train_df[g_fields].values).cuda()
        self.train_c = torch.randint(0, 2, self.train_g.shape)
        self.test_x = torch.Tensor(test_df[x_fields].values).cuda()
        self.test_y = torch.Tensor(test_df[y_fields].values).cuda()
        self.test_z = torch.Tensor(test_df[z_fields].values).cuda()
        self.test_g = torch.Tensor(test_df[g_fields].values).cuda()
        self.test_g = self.test_g - self.test_g.min()
        self.test_c = torch.randint(0, 2, self.test_g.shape)
        self.feature_dim = len(x_fields)
        self.envs_num_train = self.train_g.max().int() + 1
        self.envs_num_test = (self.test_g.max() - self.test_g.min()).int() + 1

    def fetch_train(self):
        return (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            None,
        )

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_z, self.test_g, self.test_c, None


def random_zero_one(length):
    zeros = [0] * int(length / 2)
    ones = [1] * (length - int(length / 2))
    result = zeros + ones
    random.shuffle(result)
    return torch.Tensor(result)


def all_zero(length):
    zeros = [0] * int(length)
    return torch.Tensor(zeros)


class LANDCOVER(object):
    def __init__(self, flags):
        super(LANDCOVER, self).__init__()
        config = dict()
        config["train_transforms"] = [
            {
                "classname": "innout.datasets.transforms.LambdaTransform",
                "args": {"function_path": "innout.datasets.transforms.to_tensor"},
            },
            {
                "classname": "innout.datasets.transforms.LambdaTransform",
                "args": {"function_path": "innout.datasets.transforms.tensor_to_float"},
            },
        ]
        config["test_transforms"] = [
            {
                "classname": "innout.datasets.transforms.LambdaTransform",
                "args": {"function_path": "innout.datasets.transforms.to_tensor"},
            },
            {
                "classname": "innout.datasets.transforms.LambdaTransform",
                "args": {"function_path": "innout.datasets.transforms.tensor_to_float"},
            },
        ]
        config["dataset"] = {
            "classname": "innout.datasets.landcover.Landcover",
            "args": {
                "root": "/u/nlp/data/landcover/timeseries_by_box_v2",
                "cache_path": "datasets/landcover.pkl",
                "include_NDVI": True,
                "include_ERA5": True,
                "standardize": True,
                "shuffle_domains": True,
                "seed": 1,
                "use_cache": True,
                "use_unlabeled_id": False,
                "use_unlabeled_ood": False,
                "unlabeled_prop": 0.9,
                "pretrain": False,
                "multitask": False,
            },
            "train_args": {"split": "nonafrica-train"},
            "eval_train_args": {"split": "nonafrica-train"},
            "val_args": {"split": "nonafrica-val"},
            "test_args": {"split": "nonafrica-test"},
            "test2_args": {"split": "africa"},
        }
        config["model"] = {
            "classname": "innout.models.cnn1d.CNN1D",
            "args": {"in_channels": 8, "output_size": 6},
        }
        config["use_cuda"] = True
        config["batch_size"] = flags.batch_size
        self.data_num_train = flags.batch_size
        config["eval_batch_size"] = flags.batch_size

        # create dataset
        train_dataset = init_dataset(config, "train")
        train_eval_dataset = init_dataset(config, "train")
        val_dataset = init_dataset(config, "val", train_dataset)
        test_dataset = init_dataset(config, "test2", train_dataset)
        print("landcover dataset loaded")

        #  create dataloader.
        self.train_loader = init_dataloader(config, train_dataset, "train")
        self.val_loader = init_dataloader(config, val_dataset, "val")
        self.test_loader = init_dataloader(config, test_dataset, "test2")

        self.mlp = initialize(config["model"])

    def fetch_train(self):
        bs = next(iter(self.train_loader))
        train_xz = bs["data"]
        train_x = train_xz[:, :8, :]
        train_y = bs["target"]
        lat_lon = bs["domain_label"]["lat_lon"]
        climate = torch.mean(train_xz[:, 8:, :].double(), dim=2)
        train_z = lat_lon  # aux info. You can choose from lat_lon and climate

        train_g = torch.unsqueeze(random_zero_one(train_y.shape[0]), 1)
        train_c = torch.unsqueeze(random_zero_one(train_y.shape[0]), 1)
        return (
            train_x.cuda(),
            train_y.cuda(),
            train_z.cuda(),
            train_g.cuda(),
            train_c.cuda(),
            None,
        )

    def fetch_test(self):
        bs = next(iter(self.test_loader))
        test_xz = bs["data"]
        test_x = test_xz[:, :8, :]
        test_y = torch.unsqueeze(bs["target"], -1)
        lat_lon = bs["domain_label"]["lat_lon"]
        climate = torch.mean(test_xz[:, 8:, :].double(), dim=2)
        test_z = lat_lon  # aux info

        test_g = torch.unsqueeze(random_zero_one(test_y.shape[0]), 1)
        test_c = torch.unsqueeze(random_zero_one(test_y.shape[0]), 1)
        return (
            test_x.cuda(),
            test_y.cuda(),
            test_z.cuda(),
            test_g.cuda(),
            test_c.cuda(),
            None,
        )

    def fetch_val(self):
        bs = next(iter(self.val_loader))
        val_xz = bs["data"]
        val_x = val_xz[:, :8, :]
        val_y = torch.unsqueeze(bs["target"], -1)
        lat_lon = bs["domain_label"]["lat_lon"]
        climate = torch.mean(val_xz[:, 8:, :].double(), dim=2)
        val_z = lat_lon  # aux info

        val_g = torch.unsqueeze(random_zero_one(val_y.shape[0]), 1)
        val_c = torch.unsqueeze(random_zero_one(val_y.shape[0]), 1)
        return (
            val_x.cuda(),
            val_y.cuda(),
            val_z.cuda(),
            val_g.cuda(),
            val_c.cuda(),
            None,
        )

    def fetch_mlp(self):
        return self.mlp


# added by w wang
class ADULT(object):
    def __init__(self, flags):
        super(ADULT, self).__init__()
        self.flags = flags
        self.preprocess_data()
        self.feature_dim = self.train_x.shape[1]
        self.data_num_train = self.train_x.shape[0]

    def preprocess_data(self):
        import pandas as pd

        def process_file(file_name):
            df = pd.read_csv(file_name)
            flds = ["x_", "y_", "z_", "g_"]
            out_list = []
            for fld in flds:
                _names = [x for x in df.columns.tolist() if fld in x]
                out_list.append(torch.Tensor(df[_names].values).cuda())
            return tuple(out_list)

        train_file = f"datasets/Adult/train_post_adult.csv"
        train_test_x, train_test_y, train_test_z, train_test_g = process_file(
            train_file
        )

        # split data in train envs
        train_test_data_num = train_test_x.shape[0]
        rand_index = np.random.permutation(train_test_data_num)
        train_index = rand_index[: int(train_test_data_num * 0.6666)]
        for_test_index = rand_index[int(train_test_data_num * 0.6666) :]

        self.train_x, self.train_y = (
            train_test_x[train_index, :],
            train_test_y[train_index, :],
        )
        self.train_z, self.train_g = (
            train_test_z[train_index, :],
            train_test_g[train_index, :],
        )
        self.train_c = None

        test_file = f"datasets/Adult/test_post_adult.csv"
        test_x_, test_y_, test_z_, test_g_ = process_file(test_file)

        self.test_x = torch.cat((train_test_x[for_test_index, :], test_x_))
        self.test_y = torch.cat((train_test_y[for_test_index, :], test_y_))
        self.test_z = torch.cat((train_test_z[for_test_index, :], test_z_))
        self.test_g = torch.cat((train_test_g[for_test_index, :], test_g_))
        self.test_c = None

    def fetch_train(self):
        return (
            self.train_x,
            self.train_y,
            self.train_z,
            self.train_g,
            self.train_c,
            None,
        )

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_z, self.test_g, self.test_c, None

    def fetch_mlp(self):
        return MLP2Layer(self.flags, self.feature_dim, self.flags.hidden_dim).cuda()


# if __name__ == "__main__":
#     data = ADULT(None)
#     train_x, train_y, train_z, train_g, train_c, _ = data.fetch_train()
#     test_x, test_y, test_z, test_g, test_c, _ = data.fetch_test()
#     print(train_x.shape, test_x.shape)
