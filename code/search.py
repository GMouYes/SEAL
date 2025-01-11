import argparse
import os
from util import *
from warnings import simplefilter
from ray import tune
from trainer import train_func
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
simplefilter(action='ignore', category=UserWarning)

def tune_model(args):
    scaling_config = ScalingConfig(
        num_workers=2, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
        storage_path=args["outputPath"]+'/'+args["expName"]
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    algo = ConcurrencyLimiter(algo, max_concurrent=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": args},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            search_alg=algo,
            num_samples=args["trails"],
        ),
    )
    return tuner.fit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--expName', type=str, default='debug')
    parser.add_argument('--log_dir', type=str, default='../lightning_logs/')

    args = parser.parse_args()
    cfgs = read_config(args.config_path)
    args = {**vars(args), **cfgs}

    if not os.path.isdir(args["log_dir"]+args["expName"]):
        os.mkdir(args["log_dir"]+args["expName"])

    search_params = {k:eval(v) for k,v in args["search"].items()}
    del args["search"]
    args = {**args, **search_params}
    results = tune_model(args)
    return




if __name__ == '__main__':
    main()