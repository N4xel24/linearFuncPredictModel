import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from model import SimpleFuncEvaluate
from dataset import PointDataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training_data" / "linear.csv"
MODELS = ROOT / "models"

criterion = nn.MSELoss()

model = SimpleFuncEvaluate()
optimizer = optim.Adam(model.parameters(), lr=0.00067)

Messi = 10
dataset = PointDataset(DATA)
dataloader = DataLoader(dataset, batch_size=Messi, shuffle=True)

best_loss = float("inf")

for epoch in range(69*Messi):

    total_loss = 0

    for x, target in dataloader:
        y = model(x)
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

    val_loss = total_loss/len(dataloader)

    if val_loss < best_loss:
        best_epoch = epoch
        best_loss = val_loss
        torch.save(model.state_dict(), MODELS / f"{DATA.stem}.pth")

    if (epoch+1)%69 == 0:
        print(f"[EPOCH 69x{(epoch+1)//69}/690] Loss={val_loss.item():.7f}")

print(f"BEST_EPOCH {best_epoch+1}/690 - Best_loss={best_loss.item():.7f}")
print("MODEL LEARNED SUCCESSFULLY")