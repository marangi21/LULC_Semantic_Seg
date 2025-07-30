from pathlib import Path
from datetime import datetime
import torch
from models import toyModel

class Checkpointer():
    """Classe per gestire il salvataggio e caricamento di checkpoint durante il training."""
    def __init__(self,
                 project_dir,
                 experiment_name=None,
                 save_best=True,
                 save_last=True,
                 save_every_n_epochs=None,
                 monitor='val_loss',
                 mode='min',  # 'min' or 'max'
                 ):
        self.project_dir = Path(project_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.latest_epoch = 0

        if experiment_name is None:
            self.experiment_name = self._make_exp_name()
        # Crea directory per i checkpoints
        self.checkpoint_dir = self.project_dir / "checkpoints" / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, optimizer, lr_scheduler, epoch, metrics, is_best=False):
        """Salva un checkpoint del modello"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'latest_epoch': self.latest_epoch,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.save_last:
            latest_path = self.checkpoint_dir / "last.pt"
            torch.save(checkpoint, latest_path)

        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_name):
        """Carica un checkpoint del modello"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in {self.checkpoint_dir}")
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def _make_exp_name():
        # ToDo: decidere se creare il nome qui o fuori dalla classe. In caso qui mettere nome di fallback.
        #       il nome dovrebbe avere gli iperparametri identificativi univoci dell'esperimento
        #       magari sovrascritti da quelli che metto da command line
        # ToDo: valutare la questione wandb
        pass




if  __name__ == "__main__":
    model = toyModel(32, 1)  # Istanza del modello PyTorch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Ottimizzatore PyTorch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Scheduler per il learning rate
    metrics = {'val_loss': 0.23}  # Metriche da salvare

    checkpointer = Checkpointer(project_dir="./checkpoints", experiment_name="my_experiment")
    checkpoint = checkpointer.load_checkpoint("checkpoint_epoch_5.pt")
    print()