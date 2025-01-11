import argparse
import torch
import torchmetrics
from model import DecisionMaking
from data import loadData, getText
from warnings import simplefilter
import lightning as L
import os
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
simplefilter(action='ignore', category=UserWarning)

class myModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DecisionMaking(args)
        self.text = getText()
        self.save_hyperparameters()
        self.preds, self.labels, self.weights =[], [], []
        self.mcc = torchmetrics.classification.BinaryMatthewsCorrCoef()
        self.f1 = torchmetrics.classification.MulticlassF1Score(average='macro',num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x, self.text.to(self.device))

        loss = self.model.loss(y_hat, y, w)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x, self.text.to(self.device))

        loss = self.model.loss(y_hat, y, w)
        self.log("val_loss", loss, prog_bar=True)
        return

    def configure_optimizers(self):
        self.optimizer = torch.optim.RAdam(self.parameters(), lr=10**self.args["lr"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args["gamma"])
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.model(x, self.text.to(self.device))

        loss = self.model.loss(y_hat, y, w)
        self.preds.append(y_hat)
        self.labels.append(y)
        self.weights.append(w)
        self.log("test_loss", loss, prog_bar=True)
        return 

    def reset_metric(self):
        self.mcc.reset()
        self.f1.reset()

    def on_test_end(self):
        output_path = self.args["picked_checkpoint"]
        probs = torch.cat(self.preds, axis=0)
        labels = torch.cat(self.labels, axis=0)
        weights = torch.cat(self.weights, axis=0)
        

        torch.save(probs,output_path+"/pred.pt")
        torch.save(labels,output_path+"/test_labels.pt")
        torch.save(weights,output_path+"/test_mask.pt")
        labels = labels.long()
        probs = torch.sigmoid(probs)
        probs = (probs > 0.5).long()

        mcc, f1 = [], []
        for i in range(self.args['phonePlacements']+self.args["activities"]):

            target_label = labels[:,self.args['users']+i]
            target_weights = weights[:, self.args['users']+i]>0
            target_pred= probs[:,i]
            mcc.append(self.mcc(target_pred[target_weights], target_label[target_weights]))
            f1.append(self.f1(target_pred[target_weights], target_label[target_weights]))
            self.reset_metric()

        mcc = torch.stack(mcc)
        f1 = torch.stack(f1)

        torch.save(mcc,output_path+"/mcc.pt")
        torch.save(f1,output_path+"/f1.pt")


def train_func(config):
    seed_everything(config["seed"], workers=True)
    trainData, validData = [loadData(config, dataType) for dataType in ['train', 'valid']]
    logger = TensorBoardLogger(save_dir=config["log_dir"], name=config["expName"])
    # effective batch size = batch_szie * gpus * nodes
    trainer = L.Trainer(
        max_epochs=int(config["epoch"]),
        accelerator="auto",
        devices='auto',
        deterministic=False,
        strategy=RayDDPStrategy(),  # "dp", "fsdp", "ddp"
        callbacks=[RayTrainReportCallback()],
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        gradient_clip_val=config["clip_grad"],
        logger=logger,
        enable_progress_bar=True,
        profiler="simple",
        plugins=[RayLightningEnvironment()],
        # default_root_dir=config["checkpoint_folder"]
    )
    trainer = prepare_trainer(trainer)

    trainer.fit(
        model=myModel(config),
        train_dataloaders=trainData,
        val_dataloaders=validData,
    )
    # checkpoint_callback.best_model_path
    # new_model = MyLightningModule.load_from_checkpoint(checkpoint_path="example.ckpt")
    return


