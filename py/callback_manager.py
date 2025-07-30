from abc import ABC
from typing import override

class Callback(ABC):
    """Classe base astratte per tutti i callback"""
    def on_train_start(self, trainer):
        """Chiamato all'inizio del training."""
        pass
    
    def on_train_end(self, trainer):
        """Chiamato alla fine del training."""
        pass
    
    def on_epoch_start(self, trainer, epoch):
        """Chiamato all'inizio di ogni epoca."""
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Chiamato alla fine di ogni epoca."""
        pass
    
    def on_batch_start(self, trainer, batch_idx):
        """Chiamato all'inizio di ogni batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs):
        """Chiamato alla fine di ogni batch."""
        pass
    
    def on_validation_start(self, trainer):
        """Chiamato all'inizio della validazione."""
        pass
    
    def on_validation_end(self, trainer, logs):
        """Chiamato alla fine della validazione."""
        pass

class CallbackManager:
    """
    Gestisce una lista di callbacks chiamandoli nei punti appropriati durante il training
    L'oggetto callback manager sarà passato al trainer in modo da permettergli di gestire le chiamate 
    """
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def add_callback(self, callback):
        """Aggiunge un callback alla lista"""
        if not isinstance(callback, Callback):
            raise TypeError("callback must be a subclass of Callback class")
        self.callbacks.append(callback)

    def remove_callback(self, callback_class):
        """Rimuove una callback dalla lista"""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_class)]

    def get_callbacks(self):
        """Restituisce la lista dei callback registrati"""
        return self.callbacks

    def clear_callbacks(self):
        """Rimuove tutti i callback dalla lista"""
        self.callbacks =[]

    def fire_event(self, event_name, *args, **kwargs):
        """Chiama l'evento specificato su tutti i callback registrati
        l'evento corrisponde a uno dei metodi della classe astratta Callback
        Args:
            event_name (str): Il nome dell'evento da chiamare
            *args: Argomenti posizionali da passare all'evento
            **kwargs: Argomenti keyword da passare all'evento
        """
        for callback in self.callbacks:
            if hasattr(callback, event_name):
                try:
                    getattr(callback, event_name)(*args, **kwargs)
                except Exception as e:
                    print(f"Error occurred while firing event '{event_name}' in {callback.__class__.__name__}: {e}")

class EarlyStoppingCallback(Callback):
    # Todo: testare
    def __init__(self, patience=10, min_delta=0, mode='min', monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor=monitor
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        self.current_epoch = 0

    @override
    def on_validation_end(self, trainer, logs):
        """Chiamato alla fine della validazione per gestire l'early stopping sulla base della metrica monitorata"""
        self.current_epoch += 1
        if self.monitor not in logs:
            raise ValueError(f"Metrica '{self.monitor}' non presente in logs. Metriche disponibili per early stopping: {list(logs.keys())}")
        current_value = logs[self.monitor]

        if self.mode == 'min':
            self._check_improvement_min(current_value)
        elif self.mode == 'max':
            self._check_improvement_max(current_value)
        else:
            raise ValueError("'mode' passato ad EarlyStoppingCallback deve essere 'min' o 'max'")
        
    def _check_improvement_min(self, current_value):
        """Controlla se la metrica monitorata è migliorata in modalità min"""
        if self.best_loss is None:
            self.best_loss = current_value
            self.best_epoch = self.current_epoch
        elif current_value < self.best_loss - self.min_delta:
            self.best_loss = current_value
            self.best_epoch = self.current_epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def _check_improvement_max(self, current_value):
        """Controlla se la metrica monitorata è migliorata in modalità max"""
        if self.best_loss is None:
            self.best_loss = current_value
            self.best_epoch = self.current_epoch
        elif current_value > self.best_loss + self.min_delta:
            self.best_loss = current_value
            self.best_epoch = self.current_epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ToDo: implementare
class WandbCallback(Callback):
    def __init__(self):
        pass
    
# ToDo: implementare
class LRSchedulerCallback(Callback):
    def __init__(self, scheduler):
        pass

# ToDo: implementare
class ProgressBarCallback(Callback):
    def __init__(self):
        pass