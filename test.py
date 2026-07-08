import torch
from pathlib import Path
from src.model import SimpleFuncEvaluate

model = SimpleFuncEvaluate()
function_path = Path(__file__).resolve().parent / "models" / "linear.pth"
model.load_state_dict(torch.load(function_path))
model.eval()

x = float(input("Integer x value: "))
input_data = torch.tensor([x], dtype=torch.float32)
with torch.no_grad():
    prediction = model(input_data)
    
print(f"Model prediction of f(x) value: {prediction.item()}")