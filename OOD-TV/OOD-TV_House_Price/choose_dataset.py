from utils_z import CELEBAZ_FEATURE, HousePrice, LOGITZ, LOGIT2Z, LANDCOVER, ADULT
from utils import (
    LOGIT_LYDP,
    mean_nll_multi_class,
    eval_acc_multi_class,
    mean_accuracy_multi_class,
)
from utils import (
    eval_acc_class,
    eval_acc_reg,
    mean_nll_class,
    mean_accuracy_class,
    mean_nll_reg,
    mean_accuracy_reg,
)
from torchsummary import summary
from innout.models.cnn1d import CNN1DNoNegative
from model import MLP2Layer, MLP2LayerParameter
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_celebaz_feature(flags):
    dp = CELEBAZ_FEATURE(flags)
    feature_dim = dp.feature_dim
    hidden_dim = flags.hidden_dim
    mlp = MLP2Layer(flags, feature_dim, hidden_dim).cuda()
    # mlp2 = nn.Sequential(
    #     nn.Linear(in_features=feature_dim + 1, out_features=1).cuda(), nn.Softplus(True)
    # )
    # mlp2=MLP2LayerNoNegative()
    inputSize = feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1
    mlp2 = MLP2LayerParameter(flags, inputSize, hidden_dim).cuda()
    inputSize = feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1
    print(summary(mlp2, input_size=(1, inputSize)))

    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return (
        dp,
        mlp,
        mlp2,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_adult(flags):
    dp = ADULT(flags)
    feature_dim = dp.feature_dim
    hidden_dim = flags.hidden_dim
    mlp = MLP2Layer(flags, feature_dim, hidden_dim).cuda()
    mlp2 = MLP2LayerParameter(
        flags, feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1, hidden_dim
    ).cuda()
    # print(
    #     summary(
    #         mlp2,
    #         input_size=(1, feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1),
    #     )
    # )
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return (
        dp,
        mlp,
        mlp2,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_house_price(flags):
    dp = HousePrice(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    feature_dim = dp.feature_dim
    hidden_dim = flags.hidden_dim
    mlp = MLP2Layer(flags, feature_dim, hidden_dim).cuda()
    # mlp2 = MLP2LayerNoNegative(flags, feature_dim+1, hidden_dim).cuda()
    mlp2 = MLP2LayerParameter(
        flags, feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1, hidden_dim
    ).cuda()
    print(
        summary(
            mlp2,
            input_size=(1, feature_dim * hidden_dim + hidden_dim + hidden_dim * 1 + 1),
        )
    )
    mean_nll = mean_nll_reg
    mean_accuracy = mean_accuracy_reg
    eval_acc = eval_acc_reg
    return (
        dp,
        mlp,
        mlp2,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_logit_z(flags):
    dp = LOGITZ(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
    input_dim = flags.dim_inv + flags.dim_spu
    hidden_dim = 1
    # mlp2 = nn.Sequential(
    #     nn.Linear(in_features=flags.dim_inv + flags.dim_spu + 1, out_features=1).cuda(),
    #     nn.Softplus(True),
    # )
    # mlp2=MLP2LayerNoNegative(flags, flags.dim_inv + flags.dim_spu + 1,hidden_dim=32).cuda()
    mlp2 = MLP2LayerParameter(flags, input_dim + 1, hidden_dim).cuda()
    print(summary(mlp2, input_size=(1, input_dim + 1)))
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return (
        dp,
        mlp,
        mlp2,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_landcover(flags):
    dp = LANDCOVER(flags)
    mlp = dp.fetch_mlp().cuda()
    n_parameters = count_parameters(mlp)
    mlp2 = MLP2LayerParameter(flags, feature_dim=n_parameters, hidden_dim=32).cuda()
    print(summary(mlp2, input_size=(1, n_parameters)))
    test_batch_num = len(dp.test_loader)
    train_batch_num = len(dp.train_loader)
    val_batch_num = len(dp.val_loader)
    test_batch_fetcher = dp.fetch_test
    mean_nll = mean_nll_multi_class  # CrossEntropyLoss
    mean_accuracy = mean_accuracy_multi_class  # test acc
    eval_acc = eval_acc_multi_class  # train acc
    return (
        dp,
        mlp,
        mlp2,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_logit(flags):
    dp = LOGIT_LYDP(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_spu + flags.dim_inv, out_features=1).cuda()

    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return (
        dp,
        mlp,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_logit_2z(flags):
    dp = LOGIT2Z(flags)
    test_batch_num = 1
    train_batch_num = 1
    val_batch_num = 0
    test_batch_fetcher = dp.fetch_test
    mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
    mean_nll = mean_nll_class
    mean_accuracy = mean_accuracy_class
    eval_acc = eval_acc_class
    return (
        dp,
        mlp,
        test_batch_num,
        train_batch_num,
        val_batch_num,
        test_batch_fetcher,
        mean_nll,
        mean_accuracy,
        eval_acc,
    )


def init_dataset(flags):
    dataset_specific_action = {
        "celebaz_feature": init_celebaz_feature,
        "house_price": init_house_price,
        "logit": init_logit,
        "logit_z": init_logit_z,
        "logit_2z": init_logit_2z,
        "landcover": init_landcover,
        "adult": init_adult,
    }
    return dataset_specific_action[flags.dataset](flags)
