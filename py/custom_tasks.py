from terratorch.tasks import SemanticSegmentationTask
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import lightning.pytorch as pl

class DiffLRSemanticSegmentationTask(SemanticSegmentationTask):

    # Overriding this method of TerratorchTask, parent class of SemanticSegmentationTask.
    # Pytorch Lightning uses it to build the optimizer. Sould return a dict specifying:
    # - the optimizer
    # - the lr scheduler (optional)
    # - eventually, their hparams
    def configure_optimizers(self):
        opt_hparams = self.hparams.get("optimizer_hparams", {})
        scheduler = self.hparams.get("scheduler", None)
        scheduler_hparams = self.hparams.get("scheduler_hparams", {})
        params = [
            {"params": self.model.encoder.parameters(), "lr": opt_hparams.get("encoder_lr", 1e-6)},
            {"params": self.model.decoder.parameters(), "lr": opt_hparams.get("decoder_lr", 1e-4)},
            {"params": self.model.head.parameters(), "lr": opt_hparams.get("head_lr", 1e-4)}
        ]
        if hasattr(self.model, 'neck'):
            params.append(
                {"params": self.model.neck.parameters(), "lr": opt_hparams.get("decoder_lr", 1e-4)}
            )
        optimizer = torch.optim.AdamW(params, weight_decay=opt_hparams.get("weight_decay", 1e-3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_hparams.get("mode", "min"),
            factor=scheduler_hparams.get("factor", 0.5),
            patience=scheduler_hparams.get("patience", 5),
            min_lr=scheduler_hparams.get("min_lr", 1e-8)
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }