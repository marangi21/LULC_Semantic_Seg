import numpy as np
from pathlib import Path

class EarlyStopper:
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