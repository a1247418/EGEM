import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


class PLModel(pl.LightningModule):
    def __init__(self, model, lr=1e-3, layers_to_train=None, loss_fn=F.cross_entropy):
        super().__init__()
        self.model = model
        self.lr = lr
        self.layers_to_train = layers_to_train
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # paper Supp. E: Adam for all models
        return optimizer


def train(model: pl.LightningModule, train_loader, val_loader, max_epochs=20, device="cuda"):
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device)
    trainer.fit(model, train_loader, val_loader)


def evaluate(test_loader, model, n_classes, device="cuda"):
    model.eval()
    model.to(device)
    correct_per_class = torch.zeros(n_classes, device=device)
    total_per_class = torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_per_class += torch.bincount(labels, minlength=n_classes)
            correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                correct_per_class[label] += correct[i].item()
    class_accuracy = correct_per_class / total_per_class
    total_accuracy = correct_per_class.sum() / total_per_class.sum()
    return class_accuracy.cpu().numpy(), float(total_accuracy)