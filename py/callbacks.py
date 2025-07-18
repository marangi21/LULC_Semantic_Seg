import numpy as np
from pathlib import Path

class EarlyStopping:
    """
    Classe che implementa il meccanismo di Early Stopping

    Params:
    patience                    # numero epoche di pazienza
    min_delta                   # eventuale delta di perdita per valutare se contare o no l'epoca corrente
    
    Attributes:
    self.counter                # numero epoche senza miglioramenti
    self.best_loss              # tracciamento loss migliore
    self.early_stop             # Flag per permettere l'interruzione del training
    self.best_model             # modello migliore di cui salvare lo state dict (sul valifdation set)
    self.best_epoch             # epoca in cui è stato ottenuto il modello migliore (sul validation set)
    self.current_epoch          # epoca attuale
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        self.best_opt = None
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_loss, model, optimizer):
        self.current_epoch += 1

        if self.best_loss is None:
            self.best_loss = val_loss
            #self.best_model = model.state_dict()
            #self.best_opt = optimizer.state_dict()
            self.best_epoch = self.current_epoch
        # Se la validation loss è migliorata
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            #self.best_model = model.state_dict()
            #self.best_opt = optimizer.state_dict()
            self.best_epoch = self.current_epoch
            self.counter = 0
        # Se la validation loss non è migliorata
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        

# Funzione per creare directory dell'esperimento
def create_experiment_dir():
    base_dir = Path("experiments/A3TGCN")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Trova l'indice del prossimo esperimento
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    if not existing_dirs:
        next_exp_idx = 1
    else:
        indices = [int(d.name.split("_")[1]) for d in existing_dirs]
        next_exp_idx = max(indices) + 1
    
    exp_dir = base_dir / f"exp_{next_exp_idx}"
    exp_dir.mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}\n")
    return exp_dir

# Conversione in tipi nativi di python per la serializzazione JSON
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj