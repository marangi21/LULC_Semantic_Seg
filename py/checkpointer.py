from pathlib import Path
from datetime import datetime

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

        # Crea directory per i checkpoints
        self.checkpoint_dir = self.project_dir / "checkpoints" / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        pass