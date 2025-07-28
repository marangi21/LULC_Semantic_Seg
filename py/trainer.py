import torch
from tqdm import tqdm
from callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
from metrics import *
import torch.nn.functional as F

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
        self.val_metrics = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc='Training (batches)', leave=False)
        for i , (img, mask) in enumerate(progress_bar):
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

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = [] # accumulatore di predizioni (per calcolare metriche di valutazione)
        all_targets = [] # accumulatore di target (ground truth)
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc='Validating', leave=False)
            for img, mask in progress_bar:
                img, mask = img.to(self.device), mask.to(self.device)
                outputs = self.model(img, mask)
                loss = outputs.loss

                logits = outputs.logits
                target_size = mask.shape[-2:]
                upsampled_logits = F.interpolate(
                    input=logits,
                    size = target_size,
                    mode='bilinear',
                    align_corners=False
                ) 
                predictions = torch.argmax(upsampled_logits, dim=1) # logits.shape = [batch_seize, num_classes (2), width, heigth]
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(mask.cpu().numpy())

                total_loss += loss.item()
                num_batches +=1

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        # Concatena predizioni e ground truth per calcolare le metriche 1 sola volta per tutta l'epoca di validazione
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_metrics = self.calculate_metrics(all_predictions, all_targets)
        avg_loss = total_loss / num_batches

        return avg_loss, val_metrics

    def train(self, num_epochs):
        print(f"\n🚀 Starting training for {num_epochs} epochs")
        print(f"⏳ Early Stopping: patience={self.es_patience}, min_delta={self.es_min_delta}")
        print(f"📉 LR Scheduler: patience={self.scheduler.patience}, factor={self.scheduler.factor}")
        print(f"📏 Initial LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 80)
        print()
        
        for epoch in tqdm(range(num_epochs), desc="Training (epochs)"):
            train_loss = self.train_one_epoch()
            self.train_losses.append(train_loss)
            val_loss, val_metrics = self.validate_one_epoch()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # lr scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            # Early stopping call
            self.early_stopper(val_loss, self.model, self.optimizer)   
         
            # log a wandb
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': new_lr,
                'early_stopping_counter': self.early_stopper.counter,

                # Tutte le metriche di segmentazione
                'val_pixel_accuracy': val_metrics['pixel_accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1_score': val_metrics['f1'],
                'val_iou_building': val_metrics['iou_building'],
                'val_mean_iou': val_metrics['mean_iou'],
                'val_dice_coefficient': val_metrics['dice_coefficient']
            }

            if new_lr != current_lr:
                log_dict['lr_reduced'] = True
                print(f"LR reduced: {current_lr:.4f} -> {new_lr:.4f}")
            wandb.log(log_dict)

            if self.early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {self.train_losses[epoch]:.4f}, Val Loss: {self.val_losses[epoch]:.4f}, Early Stopping Counter: {self.early_stopper.counter}/{self.es_patience}")

        return

    def calculate_metrics(self, all_predictions, all_targets):
        pred_flat = all_predictions.flatten()
        target_flat = all_targets.flatten()

        metrics = {
            'precision': calculate_precision(pred_flat, target_flat),
            'recall': calculate_recall(pred_flat, target_flat),
            'f1': calculate_f1(pred_flat, target_flat),
            'iou_building': calculate_jaccard_building(pred_flat, target_flat),
            'mean_iou': calculate_mean_iou(pred_flat, target_flat),
            'pixel_accuracy': calculate_pixel_accuracy(pred_flat, target_flat),
            'dice_coefficient': calculate_dice_coefficient(pred_flat, target_flat)
        }
        return metrics



