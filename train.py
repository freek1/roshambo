import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Simple1DCNN
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def train(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    featidx = list(range(0, 8))  # use all features

    # Datasets
    seed = 42
    X = np.load("X_roshambo.npy")
    y = np.load("y_roshambo.npy")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = torch.from_numpy(X_train).float()
    val_data = torch.from_numpy(X_val).float()

    train_dataset = torch.utils.data.TensorDataset(
        train_data, torch.from_numpy(y_train).long()
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_data, torch.from_numpy(y_val).long()
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = Simple1DCNN(num_sensors=len(featidx), num_classes=3).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    # Initialize best validation metric
    best_val_acc = 0.0

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

        # Update the learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(
            f"Resuming training from epoch {start_epoch} \
              with learning rate {args.learning_rate}"
        )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_total = 0, 0
        train_correct = 0
        all_train_preds, all_train_targets = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            X, y = batch
            X, y = (
                X.to(device),
                y.to(device),
            )

            optimizer.zero_grad()
            X = X.squeeze(1).permute(0, 2, 1)
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += y.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == y).sum().item()

            # Store predictions and targets for F1 score
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_targets.extend(y.cpu().numpy())

            # Calculate metrics
            train_acc = train_correct / train_total
            train_f1 = f1_score(all_train_targets, all_train_preds, average="weighted")

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{train_loss/train_total:.4f}",
                    "acc": f"{train_acc:.4f}",
                    "f1": f"{train_f1:.4f}",
                }
            )

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_f1 = f1_score(all_train_targets, all_train_preds, average="weighted")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)

        # Validation
        model.eval()
        val_loss, val_total = 0, 0
        val_correct = 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X, y = (
                    X.to(device),
                    y.to(device),
                )
                X = X.squeeze(1).permute(0, 2, 1)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                val_total += y.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == y).sum().item()

                # Store predictions and targets for F1 score
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_val_targets, all_val_preds, average="weighted")

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoch [{epoch+1}/{args.epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoint/best_model.pth")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"checkpoint/checkpoint_epoch_{epoch+1}.pth")

    # Plotting
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and validation loss")

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and validation accuracy")

    # F1 score plot
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label="Train F1")
    plt.plot(val_f1s, label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Training and validation F1 score")

    plt.tight_layout()
    plt.savefig("plots/training_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model")

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--save_every", type=int, default=100, help="Save checkpoint every n epochs"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
