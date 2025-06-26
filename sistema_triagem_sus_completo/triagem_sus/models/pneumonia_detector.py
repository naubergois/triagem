import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

@dataclass
class PneumoniaXRayModel:
    """Detecção de pneumonia em raio-X usando transfer learning."""
    model: Optional[nn.Module] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, data_dir: str = "data/chest_xray") -> float:
        """Treina o modelo com o dataset Chest X-Ray Pneumonia."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        train_ds = datasets.ImageFolder(train_path, transform=transform)
        val_ds = datasets.ImageFolder(val_path, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=16)

        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in base_model.parameters():
            p.requires_grad = False
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, 2)
        self.model = base_model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=1e-3)

        self.model.train()
        for epoch in range(1):
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0
        return acc

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Modelo não treinado")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, 2)
        base_model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = base_model.to(self.device)
