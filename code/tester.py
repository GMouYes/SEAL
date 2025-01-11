import argparse
import os
from util import *
from warnings import simplefilter
from ray import tune,train
from trainer import *
from ray.train import Result, Checkpoint
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
simplefilter(action='ignore', category=UserWarning)

# /home/gmou/ray_results/TorchTrainer_2024-03-19_23-20-11
def infer_model(args):
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # restored_tuner = tune.Tuner.restore(args["outputPath"]+args["expName"], trainable=ray_trainer)
    # restored_tuner = tune.Tuner.restore('/home/gmou/ray_results/TorchTrainer_2024-03-19_23-20-11', trainable=ray_trainer)
    # result_grid = restored_tuner.get_results()
    # best_result: Result = result_grid.get_best_result()

    # with best_result.checkpoint.as_directory() as checkpoint_dir:
    #     lightning_checkpoint_path = f"{checkpoint_dir}/checkpoint.ckpt"
    lightning_checkpoint_path = '../output/extra_MLP_BERT_ray/TorchTrainer_2024-03-30_23-46-25/TorchTrainer_841a8d97_8_activities=12,batch_size=4096,bert_dim=768,clip_grad=1000,config_path=config_MLP_BERT_extra_ray_config_yml_2024-04-01_01-37-21/checkpoint_000387/checkpoint.ckpt'

    testData = loadData(args, 'test')
    model = myModel.load_from_checkpoint(lightning_checkpoint_path)
    model.args["picked_checkpoint"] = '/'.join(lightning_checkpoint_path.split('/')[:-1])

    trainer = L.Trainer()
    trainer.test(
        model,
        dataloaders=testData,
    )
    
    print("Infer finished!! ===========")
    mcc = torch.load(model.args["picked_checkpoint"]+"/mcc.pt").cpu().detach().numpy()
    f1 = torch.load(model.args["picked_checkpoint"]+"/f1.pt").cpu().detach().numpy()

    print("MCC!! ===========")
    print("infer Phone Placement: {:.4f}".format(np.nanmean(mcc[:args["phonePlacements"]])))
    print("infer Activities: {:.4f}".format(np.nanmean(mcc[args["phonePlacements"]:])))
    print(mcc)
    print("")

    print("F1!! ===========")
    print("infer Phone Placement: {:.4f}".format(np.nanmean(f1[:args["phonePlacements"]])))
    print("infer Activities: {:.4f}".format(np.nanmean(f1[args["phonePlacements"]:])))
    print(f1)
    print("")


    return



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
    results = infer_model(args)
    return




if __name__ == '__main__':
    main()