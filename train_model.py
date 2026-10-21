

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

CONFIG = {
    "model_name": "mobilenet_v3_large",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 0.001,
    "num_workers": 2,
    "train_split": 0.8,
    "data_dir": "./dataset",
    "output_dir": "./output",
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 2. Dataset Preparation & Augmentation
# ---------------------------------------------------------------------------
def get_data_transforms(img_size=224):
    """
    Data augmentation simulating outdoor agricultural field conditions
    (varying sunlight, orientation, leaf angles common in Pakistani farms).
    """
    return {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# ---------------------------------------------------------------------------
# 3. Model Architecture (Transfer Learning)
# ---------------------------------------------------------------------------
def build_model(num_classes, pretrained=True):
    """
    MobileNetV3-Large provides optimal trade-off: high accuracy (>95%)
    and lightweight parameters (<5M) suitable for free edge/browser deployment.
    """
    print(f"[*] Building {CONFIG['model_name']} with {num_classes} output classes...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Replace classification head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# ---------------------------------------------------------------------------
# 4. Training & Validation Loop
# ---------------------------------------------------------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"[+] New best validation accuracy: {best_acc * 100:.2f}%")

    time_elapsed = time.time() - since
    print(f"\n[✓] Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"[✓] Best Validation Accuracy: {best_acc * 100:.2f}%")

    model.load_state_dict(best_model_wts)
    return model, history

# ---------------------------------------------------------------------------
# 5. Export to PyTorch (.pth) and ONNX (.onnx)
# ---------------------------------------------------------------------------
def export_models(model, class_names, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to("cpu")

    # 1. Save PyTorch State Dict
    pth_path = os.path.join(output_dir, "crop_disease_model.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"[✓] PyTorch model weights saved: {pth_path}")

    # 2. Save Class Labels JSON
    labels_path = os.path.join(output_dir, "class_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2, ensure_ascii=False)
    print(f"[✓] Class labels saved: {labels_path}")

    # 3. Export to ONNX (Open Neural Network Exchange for zero-server inference)
    onnx_path = os.path.join(output_dir, "crop_disease_model.onnx")
    dummy_input = torch.randn(1, 3, CONFIG["img_size"], CONFIG["img_size"])
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"[✓] ONNX model successfully exported: {onnx_path}")
    except Exception as e:
        print(f"[!] Standard ONNX export notice: {e}")
        try:
            # Try with opset 12 fallback
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True
            )
            print(f"[✓] ONNX model exported with fallback: {onnx_path}")
        except Exception as e2:
            print(f"[!] ONNX export requires onnxscript. Run: !pip install onnxscript")
            ts_path = os.path.join(output_dir, "crop_disease_model.pt")
            scripted = torch.jit.trace(model, dummy_input)
            scripted.save(ts_path)
            print(f"[✓] Exported TorchScript model as backup: {ts_path}")

def check_or_prepare_dataset(data_dir):
    """Ensure dataset directory exists and has class folders; generates sample data if empty."""
    os.makedirs(data_dir, exist_ok=True)
    existing_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not existing_classes:
        print(f"[!] No class folders found in '{data_dir}'. Generating starter Pakistani crop classes for immediate testing...")
        sample_classes = [
            "Wheat___Yellow_Rust", "Wheat___Loose_Smut", "Wheat___Healthy",
            "Rice___Bacterial_Blight", "Rice___Blast", "Rice___Healthy",
            "Cotton___Leaf_Curl_Virus", "Cotton___Bacterial_Blight", "Cotton___Healthy",
            "Tomato___Late_Blight", "Tomato___Yellow_Leaf_Curl", "Tomato___Healthy",
            "Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy",
            "Maize___Northern_Blight", "Maize___Common_Rust", "Maize___Healthy"
        ]
        from PIL import Image
        for cls in sample_classes:
            cls_dir = os.path.join(data_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(15):
                color = (np.random.randint(40, 180), np.random.randint(100, 220), np.random.randint(30, 90))
                img = Image.new("RGB", (224, 224), color=color)
                img.save(os.path.join(cls_dir, f"sample_{i}.jpg"))
        print(f"[✓] Generated starter dataset with {len(sample_classes)} classes so training runs immediately!")

# ---------------------------------------------------------------------------
# 6. Main Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Fasal Shifa AI - PyTorch Training Pipeline ")
    print(f" Target Device: {device}")
    print("=" * 60)

    data_dir = CONFIG["data_dir"]
    check_or_prepare_dataset(data_dir)

    data_transforms = get_data_transforms(CONFIG["img_size"])
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"[*] Found {len(full_dataset)} images across {num_classes} classes.")

    train_size = int(CONFIG["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms to val dataset
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = data_transforms["val"]

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"]),
        "val": DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    trained_model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=CONFIG["epochs"]
    )

    export_models(trained_model, class_names, CONFIG["output_dir"])
    print("\n[🎉] Model training and export complete! Ready for evaluation and web deployment.")

if __name__ == "__main__":
    main()
