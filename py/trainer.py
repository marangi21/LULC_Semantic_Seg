import torch
from tqdm import tqdm
from callbacks import EarlyStopping

class Trainer():
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, device, es_patience=10, es_min_delta=0):
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.early_stopper = EarlyStopping(patience=self.es_patience, min_delta=self.es_min_delta)
        self.train_losses = []
        self.val_losses = []

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
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.train_losses.append(self.train_one_epoch())
            self.val_losses.append(self.validate())
            self.early_stopper(self.val_losses[-1], self.model, self.optimizer)
            if self.early_stopper.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {self.train_losses[epoch]:.4f}, Val Loss: {self.val_losses[epoch]:.4f}, Early Stopping Counter: {self.early_stopper.counter}/{self.es_patience}")
        return

