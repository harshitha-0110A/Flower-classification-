# Flower-classification-
import os
import time
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from PIL import Image
from tqdm import tqdm

print("All libraries imported successfully")
print(f"   PyTorch version: {torch.__version__}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")

if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   Running on CPU — training will work, just slower (~20–40 min)")
    print("   On H200 later, same code runs 40x faster")


# ─────────────────────────────────────────────────────────────
# CELL 3 │ Download Flowers Dataset
# ─────────────────────────────────────────────────────────────
# Google's 5-class flower dataset: ~3,600 images total

DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org"
    "/example_images/flower_photos.tgz"
)
DATASET_DIR = "flower_photos"

if not os.path.exists(DATASET_DIR):
    print("📥 Downloading flowers dataset (~220 MB) ...")
    urllib.request.urlretrieve(DATASET_URL, "flower_photos.tgz")
    print("📦 Extracting ...")
    with tarfile.open("flower_photos.tgz", "r:gz") as tar:
        try:
            tar.extractall(".", filter='data')
        except TypeError:
            tar.extractall(".")
    os.remove("flower_photos.tgz")
    print(f"✅ Dataset ready in '{DATASET_DIR}/'")
else:
    print(f"✅ Dataset already exists — skipping download")

# List classes and image counts
class_dirs = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
    and not d.startswith('.')
])
print(f"\n📂 Classes found: {class_dirs}")
total = 0
for cls in class_dirs:
    n = len([
        f for f in os.listdir(os.path.join(DATASET_DIR, cls))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    total += n
    print(f"   {cls:<12}: {n} images")
print(f"   {'TOTAL':<12}: {total} images")


# ─────────────────────────────────────────────────────────────
# CELL 4 │ Transforms, Dataset Split, DataLoaders
# ─────────────────────────────────────────────────────────────

IMG_SIZE   = 224
BATCH_SIZE = 32      # H200 UPGRADE → 128 or 256
NUM_WORKERS = 0      # H200 UPGRADE → 4  (Windows: keep 0)

# Augmented transform for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Clean transform for testing (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load full dataset twice — one with each transform
train_data_full = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
test_data_full  = datasets.ImageFolder(DATASET_DIR, transform=test_transform)

CLASS_NAMES  = train_data_full.classes
NUM_CLASSES  = len(CLASS_NAMES)

# Reproducible 80/20 split
np.random.seed(42)
indices     = np.random.permutation(len(train_data_full))
split       = int(0.8 * len(indices))
train_idx   = indices[:split]
test_idx    = indices[split:]

train_dataset = torch.utils.data.Subset(train_data_full, train_idx)
test_dataset  = torch.utils.data.Subset(test_data_full,  test_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda')
)
test_loader  = DataLoader(
    test_dataset,  batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda')
)

print(f"✅ Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"   Train samples : {len(train_dataset)}")
print(f"   Test  samples : {len(test_dataset)}")


# ─────────────────────────────────────────────────────────────
# CELL 5 │ Build Model — Transfer Learning Helper
# ─────────────────────────────────────────────────────────────

def build_model(arch='resnet18', num_classes=5, freeze_backbone=True):
    """
    Load a pretrained model, optionally freeze the backbone,
    then replace the classifier head for our dataset.

    freeze_backbone=True  → Only trains final layer (fast, low memory)
    freeze_backbone=False → Trains entire network (better accuracy)
                           H200 UPGRADE → set freeze_backbone=False
    """
    if arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"✅ {arch} loaded — trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")
    return model


# ─────────────────────────────────────────────────────────────
# CELL 6 │ Training Loop
# ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, test_loader,
                epochs=10, lr=1e-3, save_name='best_model.pth'):
    """
    Full training loop.
    Returns: history dict, best test accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    # Drop LR by 10x halfway through
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 2), gamma=0.1
    )

    history = {'train_loss': [], 'train_acc': [],
               'test_loss':  [], 'test_acc':  []}
    best_acc   = 0.0
    start_time = time.time()

    for epoch in range(epochs):

        # ── Train ──────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:02d}/{epochs} [Train]",
            leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            correct  += out.argmax(1).eq(labels).sum().item()
            total    += labels.size(0)

        tr_loss = run_loss / len(train_loader)
        tr_acc  = 100.0 * correct / total

        # ── Evaluate ───────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out  = model(images)
                loss = criterion(out, labels)
                val_loss += loss.item()
                correct  += out.argmax(1).eq(labels).sum().item()
                total    += labels.size(0)

        te_loss = val_loss / len(test_loader)
        te_acc  = 100.0 * correct / total

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        # Save best checkpoint
        tag = ''
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), save_name)
            tag = '  💾 saved!'

        scheduler.step()

        print(
            f"  Epoch {epoch+1:02d}/{epochs} │ "
            f"Train {tr_acc:.1f}% loss {tr_loss:.4f} │ "
            f"Test  {te_acc:.1f}% loss {te_loss:.4f}{tag}"
        )

    elapsed = time.time() - start_time
    print(f"\n  ⏱  Done in {elapsed/60:.1f} min  │  Best test accuracy: {best_acc:.2f}%")
    return history, best_acc


# ─────────────────────────────────────────────────────────────
# CELL 7 │ Train ResNet-18  ← MAIN MODEL
# ─────────────────────────────────────────────────────────────

EPOCHS = 10    # H200 UPGRADE → 20 or 30

print(f"\n🚀 Training ResNet-18 for {EPOCHS} epochs on {device} ...")
print("   (on CPU this takes ~20–40 min — go grab a coffee ☕)\n")

model = build_model('resnet18', num_classes=NUM_CLASSES, freeze_backbone=True)
history, best_acc = train_model(
    model, train_loader, test_loader,
    epochs=EPOCHS, lr=1e-3, save_name='resnet18_best.pth'
)


# ─────────────────────────────────────────────────────────────
# CELL 8 │ Plot Training Curves
# ─────────────────────────────────────────────────────────────

def plot_curves(history, title='ResNet-18', fname='training_curves.png'):
    eps = range(1, len(history['train_acc']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(eps, history['train_acc'], 'b-o', lw=2, label='Train')
    ax1.plot(eps, history['test_acc'],  'r-o', lw=2, label='Test')
    ax1.set(xlabel='Epoch', ylabel='Accuracy (%)',
            title=f'{title} — Accuracy', ylim=[0, 100])
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(eps, history['train_loss'], 'b-o', lw=2, label='Train')
    ax2.plot(eps, history['test_loss'],  'r-o', lw=2, label='Test')
    ax2.set(xlabel='Epoch', ylabel='Loss', title=f'{title} — Loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {fname}")

plot_curves(history, title='ResNet-18', fname='resnet18_curves.png')


# ─────────────────────────────────────────────────────────────
# CELL 9 │ Confusion Matrix + Classification Report
# ─────────────────────────────────────────────────────────────

def eval_model(model, loader, class_names):
    """Run inference on the full test set, return predictions + labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion(preds, labels, class_names,
                   title='ResNet-18', fname='confusion_matrix.png'):
    cm   = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(f'{title} — Confusion Matrix', fontsize=14,
                 fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {fname}")
    print(f"\n📊 Classification Report — {title}:")
    print(classification_report(labels, preds, target_names=class_names))


preds, labels = eval_model(model, test_loader, CLASS_NAMES)
plot_confusion(preds, labels, CLASS_NAMES,
               title='ResNet-18', fname='resnet18_confusion.png')


# ─────────────────────────────────────────────────────────────
# CELL 10 │ Sample Predictions Grid (3 × 3)
# ─────────────────────────────────────────────────────────────

def show_predictions(model, test_dataset, class_names, n=9,
                     fname='sample_predictions.png'):
    model.eval()
    unnorm = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    indices = np.random.choice(len(test_dataset), n, replace=False)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.5 * rows))
    fig.suptitle('Sample Predictions  (green = correct, red = wrong)',
                 fontsize=14, fontweight='bold')

    for ax, idx in zip(axes.flat, indices):
        img_t, true = test_dataset[idx]
        with torch.no_grad():
            out   = model(img_t.unsqueeze(0).to(device))
            probs = torch.softmax(out, 1)[0]
            pred  = probs.argmax().item()
            conf  = probs[pred].item() * 100

        display = unnorm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(display)
        ax.axis('off')
        color = 'green' if pred == true else 'red'
        ax.set_title(
            f"Pred: {class_names[pred]} ({conf:.0f}%)\nTrue: {class_names[true]}",
            color=color, fontsize=10, fontweight='bold'
        )

    # Hide any unused subplots
    for ax in axes.flat[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {fname}")

show_predictions(model, test_dataset, CLASS_NAMES)


# ─────────────────────────────────────────────────────────────
# CELL 11 │ Model Comparison  (ResNet-18 vs ResNet-50 vs EfficientNet-B0)
# ─────────────────────────────────────────────────────────────
# ⚠️  Trains 2 extra models — skip this cell if short on time.
#     COMPARE_EPOCHS=5 is intentionally short to save time on CPU.
# ─────────────────────────────────────────────────────────────

COMPARE_EPOCHS = 5   # Fewer epochs — just to compare, not to maximise accuracy

results = {
    'ResNet-18': {'acc': best_acc, 'time': None}
}

for arch, name in [('resnet50', 'ResNet-50'),
                   ('efficientnet_b0', 'EfficientNet-B0')]:
    print(f"\n📦 Training {name} for {COMPARE_EPOCHS} epochs ...")
    m   = build_model(arch, num_classes=NUM_CLASSES, freeze_backbone=True)
    t0  = time.time()
    _, acc = train_model(
        m, train_loader, test_loader,
        epochs=COMPARE_EPOCHS, save_name=f'{arch}_best.pth'
    )
    results[name] = {'acc': acc, 'time': (time.time() - t0) / 60}

# Print table
print("\n" + "=" * 45)
print(f"  {'Model':<20} {'Best Acc':>10}")
print("-" * 45)
for name, r in results.items():
    print(f"  {name:<20} {r['acc']:>9.2f}%")
print("=" * 45)

# Bar chart
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
colors = ['#4C72B0', '#DD8452', '#55A868']
bars   = ax.bar(names, accs, color=colors, width=0.5,
                edgecolor='white', linewidth=2)
ax.set(ylabel='Best Test Accuracy (%)', ylim=[0, 105],
       title='Model Comparison — Flower Classification')
ax.grid(axis='y', alpha=0.3)
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f'{a:.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Saved: model_comparison.png")


# ─────────────────────────────────────────────────────────────
# CELL 12 │ Single-Image Prediction  ← Use this in your live demo!
# ─────────────────────────────────────────────────────────────

_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_flower(image_path, model=model, class_names=CLASS_NAMES):
    """
    Predict the flower class from ANY image file.

    Usage (in demo):
        predict_flower("rose.jpg")
        predict_flower("C:/Users/you/Downloads/random_flower.jpg")
    """
    model.eval()
    img    = Image.open(image_path).convert('RGB')
    tensor = _infer_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, 1)[0]
        pred  = probs.argmax().item()

    print(f"\n🌸  Image   : {image_path}")
    print(f"🏆  Predicted: {class_names[pred].upper()}")
    print(f"\n    Class probabilities:")
    for i, (name, p) in enumerate(zip(class_names, probs)):
        bar    = '█' * int(p.item() * 30)
        marker = ' ← winner' if i == pred else ''
        print(f"    {name:<12}: {p.item()*100:5.1f}%  {bar}{marker}")

    # Side-by-side: image + probability bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.imshow(img);  ax1.axis('off')
    ax1.set_title('Input Image', fontsize=12)

    bar_colors = ['#4CAF50' if i == pred else '#90CAF9'
                  for i in range(NUM_CLASSES)]
    ax2.barh(class_names, [p.item() * 100 for p in probs],
             color=bar_colors)
    ax2.set(xlabel='Confidence (%)', xlim=[0, 100],
            title=f"Prediction: {class_names[pred]}  "
                  f"({probs[pred].item()*100:.1f}%)")
    plt.tight_layout()
    plt.savefig('single_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

    return class_names[pred], probs[pred].item()


# Quick self-test using a sample from the dataset
_sample_path = os.path.join(DATASET_DIR, class_dirs[0],
                             os.listdir(os.path.join(DATASET_DIR, class_dirs[0]))[0])
print(f"\n📸 Self-test on: {_sample_path}")
predict_flower(_sample_path)

print("\n✅  predict_flower() is ready for your demo!")
print("    Just call:  predict_flower('any_flower_image.jpg')")


# ─────────────────────────────────────────────────────────────
# CELL 13 │ GPU Stats  (shows info; on CPU prints upgrade guide)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("💻  SYSTEM SUMMARY")
print("=" * 55)
print(f"  PyTorch : {torch.__version__}")
print(f"  Device  : {device}")

if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB total  "
          f"│  {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
else:
    print("""
  ─── H200 UPGRADE CHECKLIST ───────────────────────────────
  When you move this notebook to JupyterHub on the H200:

  1. BATCH_SIZE   = 32    →  128  (4x more data per step)
  2. NUM_WORKERS  = 0     →  4    (parallel data loading)
  3. freeze_backbone=True →  False  (fine-tune whole network)
  4. EPOCHS       = 10   →  20 or 30
  5. COMPARE_EPOCHS= 5   →  10
  6. After model = build_model(...), add:
       model = torch.compile(model)   # PyTorch 2.0 speed boost
  7. In train loop, enable mixed precision:
       scaler = torch.cuda.amp.GradScaler()
       with torch.cuda.amp.autocast():
           out  = model(images)
           loss = criterion(out, labels)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
  ──────────────────────────────────────────────────────────
""")

print("\n📁  Files saved this run:")
for f in [
    'resnet18_best.pth',
    'resnet18_curves.png',
    'resnet18_confusion.png',
    'sample_predictions.png',
    'model_comparison.png',
    'single_prediction.png',
]:
    exists = '✅' if os.path.exists(f) else '⬜'
    print(f"  {exists}  {f}")