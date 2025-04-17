# Here defines the parameters for each algorithm and dataset.
# These parameters are not the same as the parameters used in the paper
parameterMap = {
    "ZIN-TV-L1": {
        "mnist": {
            "aux_num": 3,
            "batch_size": 1024,
            "seed": 112,
            "classes_num": 3,
            "dataset": "mnist",
            "opt": "adam",
            "l2_regularizer_weight": 0.001,
            "n_restarts": 1,
            "num_classes": 6,
            "z_class_num": 1,
            "noise_ratio": 0.2,
            "cons_train": "0.999_0.8",
            "cons_test": "0.01_0.2_0.8_0.999",
            "penalty_weight": 2,
            "dim_inv": 5,
            "dim_sp": 5,
            "irm_type": "infer_irmv1_multi_class_tvl1",
            "lr": 0.01,
            "steps": 1,
            "penalty_anneal_iters": 2,
        },
    },
}
