from pathlib import Path
from datetime import datetime
import torch

class Checkpointer():
    """Classe per gestire il salvataggio e caricamento di checkpoint durante il training."""
    def __init__(self,
                 project_dir,
                 experiment_name=None,
                 save_best=True,
                 save_latest=True,
                 save_every_n_epochs=None,
                 monitor='val_loss',
                 mode='min',  # 'min' or 'max'
                 ):
        self.project_dir = Path(project_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_best = save_best
        self.save_latest = save_latest
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.latest_epoch = 0
        # Crea directory per i checkpoints
        self.checkpoint_dir = self.project_dir / "checkpoints" / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, optimizer, lr_scheduler, epoch, metrics, checkpoint_name=None):
        """Salva un checkpoint del modello"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        # Salva il checkpoint
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)