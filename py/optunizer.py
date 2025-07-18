import optuna
import os
from optuna.samplers import TPESampler 

class Optunizer:
    """Classe per gestire l'ottimizzazione degli iperparametri con Optuna."""
    
    def __init__(self, study_name, parameter_space, n_trials, project_dir=None, direction='minimize', seed=None):
        """
        Inizializza un oggetto Optunizer.
        
        Args:
            study_name: Nome dello studio Optuna
            parameter_space: Dizionario con spazi di ricerca per ogni parametro
            n_trials: Numero di trial da eseguire
            project_dir: Percorso del progetto per storage
            direction: 'minimize' o 'maximize'
        """
        self.study_name = study_name
        self.parameter_space = parameter_space
        self.n_trials = n_trials
        self.direction = direction
        
        # Crea un sampler con seed fisso per risultati replicabili
        sampler = TPESampler(seed=seed) if seed is not None else None

        # Crea directory per Optuna storage
        if project_dir:
            storage_dir = os.path.join(project_dir, "optuna_storage")
            os.makedirs(storage_dir, exist_ok=True)
            storage_name = f"sqlite:///{storage_dir}/optuna.db"
            self.study = optuna.create_study(
                direction=direction, 
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=True,
                sampler=sampler
            )
        else:
            self.study = optuna.create_study(
                direction=direction, 
                study_name=study_name,
                sampler=sampler
            )
            
        self.current_trial = None
    
    def next_trial(self):
        """Ottiene i parametri per il prossimo trial."""
        self.current_trial = self.study.ask()
        parameters = {}
        
        for param_name, param_config in self.parameter_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                parameters[param_name] = self.current_trial.suggest_categorical(
                    param_name, param_config['values']
                )
            elif param_type == 'int':
                parameters[param_name] = self.current_trial.suggest_int(
                    param_name, param_config['min'], param_config['max']
                )
            elif param_type == 'float':
                log_scale = param_config.get('log_scale', False)
                parameters[param_name] = self.current_trial.suggest_float(
                    param_name, param_config['min'], param_config['max'], log=log_scale
                )
                
        return parameters
    
    def report_result(self, value):
        """Riporta il risultato del trial corrente ad Optuna."""
        if self.current_trial is None:
            raise ValueError("current_trial = None. Chiamare next_trial() prima di report_result()")
        self.study.tell(self.current_trial, value)
    
    def get_best_params(self):
        """Ottiene i migliori parametri trovati."""
        return self.study.best_params
    
    def get_best_value(self):
        """Ottiene il miglior valore trovato."""
        return self.study.best_value
    
    def __iter__(self):
        """Rende la classe iterabile per i trials."""
        self.trial_count = 0
        return self
    
    def __next__(self):
        """Ottiene il prossimo set di parametri quando iterato."""
        if self.trial_count < self.n_trials:
            self.trial_count += 1
            return self.next_trial()
        raise StopIteration
