# Here defines the parameters for each algorithm and dataset.
# These parameters are not the same as the parameters used in the paper
parameterMap = {
    "EIIL": {
        "adult": {
            "l2_regularizer_weight": 0.001,
            "irm_type": "eiil",
            "dataset": "adult",
            "n_restarts": 1,
            "seed": 1,
            "lr": 0.004,
            "aux_num": 6,
            "z_class_num": 4,
            "envs_num_train": 2,
            "envs_num_test": 4,
            "penalty_anneal_iters": 8000,
            "steps":1,
            "penalty_weight": 2000,
        }
    },
    "LfF": {
        "adult": {
            "l2_regularizer_weight": 0.001,
            "irm_type": "lff",
            "dataset": "adult",
            "n_restarts": 1,
            "seed": 1,
            "lr": 0.004,
            "aux_num": 6,
            "z_class_num": 4,
            "envs_num_train": 2,
            "envs_num_test": 4,
            "penalty_anneal_iters": 8000,
            "steps":1,
            "penalty_weight": 2000,
        }
    },
}
