import torch
from tqdm import tqdm
from callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import wandb

WANDB_API_KEY = os.getenv('WANDB_API_KEY')

class Trainer():
    """
    Classe che rappresenta un trainer per un modello. Rende possibile addestrare un modello passando:
    Attributi:
        model: modello da addestrare
        optimizer: ottimizzatore da utilizzare
        train_dataloader: dataloader per il training set
        val_dataloader: dataloader per il validation set
        device: dispositivo su cui eseguire il training
        es_patience: pazienza per early stopping
        es_min_delta: soglia di miglioramento minimo per early stopping

    Metodi:
        train_one_epoch: addestra il modello per un'epoca
        validate: valuta il modello sui dati di validation per questa epoca
        train: addestra il modello per un numero di epoche specificato

    Usa:
        EarlyStopping(): classe i cui oggetti implementano il meccanismo di early stopping
    """
    
    def __init__(self,
                 model,
                 optimizer,
                 train_dataloader,
                 val_dataloader,
                 device= 'cuda' if torch.cuda.is_available() else 'cpu',
                 es_patience=10,
                 es_min_delta=0,
                 lr_scheduler = None,
                 lr_factor=0.5,
                 lr_patience=5,
                 min_lr=1e-7,
                 verbose=False
                 ):
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.verbose=verbose
        self.early_stopper = EarlyStopping(
            patience=self.es_patience,
            min_delta=self.es_min_delta
            )

        # Se non viene fornito lr scheduler usa ReduceLROnPlateau, se viene fornito usa quello
        if lr_scheduler == None:
            self.lr_factor=lr_factor
            self.lr_patience = lr_patience
            self.min_lr = min_lr
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=self.lr_factor,
                patience=self.lr_patience,
                min_lr=self.min_lr
            )
            print(f"\nNessun lr_scheduler fornito, utilizzo ReduceLROnPlateau con factor={self.lr_factor} e patience={self.lr_patience}")
        else:
            self.scheduler = lr_scheduler
            self.lr_factor = self.scheduler.factor
            self.lr_patience =  self.scheduler.patience

        # Roba per tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = {
            'epoch': [],        # epoca in cui quel lr[i] viene impostato
            'lr': []            # valore di lr[i] (i-esimo lr utilizzato)
        }

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        for i , (img, mask) in enumerate(self.train_dataloader):
            img, mask = img.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(img, mask)
            loss = outputs.loss
            # Backward pass
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                print(f"Batch: {i}/{len(self.train_dataloader)}, loss: {loss.item():.4f}")
        
        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for img, mask in self.val_dataloader:
                img, mask = img.to(self.device), mask.to(self.device)
                outputs = self.model(img, mask)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches +=1

        return total_loss / num_batches

    def train(self, num_epochs):
        print(f"\n🚀 Starting training for {num_epochs} epochs")
        print(f"⏳ Early Stopping: patience={self.es_patience}, min_delta={self.es_min_delta}")
        print(f"📉 LR Scheduler: patience={self.scheduler.patience}, factor={self.scheduler.factor}")
        print(f"📏 Initial LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 80)
        print()
        self.learning_rates['epoch'].append(0)
        self.learning_rates['lr'].append(self.optimizer.param_groups[0]['lr'])
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.train_losses.append(self.train_one_epoch())
            self.val_losses.append(self.validate())
            # lr scheduler step
            self.scheduler.step(self.val_losses[-1])
            self.learning_rates['epoch'].append(epoch+1)
            self.learning_rates['lr'].append(self.optimizer.param_groups[0]['lr'])
            # Early stopping check
            self.early_stopper(self.val_losses[-1], self.model, self.optimizer)            
            if self.early_stopper.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {self.train_losses[epoch]:.4f}, Val Loss: {self.val_losses[epoch]:.4f}, Early Stopping Counter: {self.early_stopper.counter}/{self.es_patience}")
        return

