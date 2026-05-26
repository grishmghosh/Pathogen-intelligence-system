import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = r'C:\\Pathogen-intelligence-system\\checkpoints'
DATASET_ROOT   = r'C:\Pathogen-intelligence-system\\dataset_split'

val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'val'), transform=val_transform)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'test'), transform=val_transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_features, 4))
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'efficientnet_b0_best.pth'), map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.to(DEVICE).eval()

all_logits, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        all_logits.append(model(imgs.to(DEVICE)).cpu())
        all_labels.append(labels)
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

log_temp = nn.Parameter(torch.zeros(1))
ts_optim = optim.LBFGS([log_temp], lr=0.01, max_iter=50)

def ts_eval():
    ts_optim.zero_grad()
    loss = cross_entropy(all_logits / torch.exp(log_temp), all_labels)
    loss.backward()
    return loss

ts_optim.step(ts_eval)
temp = torch.exp(log_temp).item()
if temp <= 0 or temp > 10:
    temp = 1.0
    print(f'Temperature clamped to 1.0')
print(f'Optimal temperature: {temp:.4f}')
torch.save({'temperature': temp}, os.path.join(CHECKPOINT_DIR, 'efficientnet_b0_temperature.pth'))

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        logits = model(imgs.to(DEVICE)) / temp
        preds  = torch.softmax(logits, dim=1).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
print(f'Test Accuracy: {correct/total:.4f}')
print('Done.')