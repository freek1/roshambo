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
import os


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for saving
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Initialize lists to store metrics across all folds
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    all_train_f1s = []
    all_val_f1s = []

    # Get unique subjects (assuming first dimension is subjects)
    X = np.load("X_roshambo.npy")
    subjects = np.unique(np.arange(X.shape[0]))

    # TODO: remove this, to do LOSO we need to change the data processing in roshambo.ipynb
    for subject_idx in subjects[:10]:
        # Load data
        X = np.load("X_roshambo.npy")
        y = np.load("y_roshambo.npy")

        print(f"\nTraining on subject {subject_idx}")

        # Split data for this subject
        X_train = X[np.arange(X.shape[0]) != subject_idx]
        X_val = X[subject_idx : subject_idx + 1]  # Single subject for validation
        y_train = y[np.arange(y.shape[0]) != subject_idx]
        y_val = y[subject_idx : subject_idx + 1]

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
        model = Simple1DCNN(num_sensors=X.shape[-1], num_classes=3).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-5,
        )

        # Initialize best validation metric
        best_val_acc = 0.0

        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []

        for epoch in range(args.epochs):
            model.train()
            train_loss, train_total = 0, 0
            train_correct = 0
            all_train_preds, all_train_targets = [], []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

            for batch in pbar:
                X, y = batch
                X, y = X.to(device), y.to(device)

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
                train_f1 = f1_score(
                    all_train_targets, all_train_preds, average="weighted"
                )

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
                    X, y = X.to(device), y.to(device)
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
                torch.save(
                    model.state_dict(),
                    f"checkpoint/best_model_subject_{subject_idx}.pth",
                )

        # Store metrics for this fold
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)
        all_train_f1s.append(train_f1s)
        all_val_f1s.append(val_f1s)

    # Convert lists to numpy arrays for easier averaging
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    all_train_accs = np.array(all_train_accs)
    all_val_accs = np.array(all_val_accs)
    all_train_f1s = np.array(all_train_f1s)
    all_val_f1s = np.array(all_val_f1s)

    # Calculate mean and std across folds
    mean_train_loss = np.mean(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    mean_train_acc = np.mean(all_train_accs, axis=0)
    mean_val_acc = np.mean(all_val_accs, axis=0)
    mean_train_f1 = np.mean(all_train_f1s, axis=0)
    mean_val_f1 = np.mean(all_val_f1s, axis=0)

    std_train_loss = np.std(all_train_losses, axis=0)
    std_val_loss = np.std(all_val_losses, axis=0)
    std_train_acc = np.std(all_train_accs, axis=0)
    std_val_acc = np.std(all_val_accs, axis=0)
    std_train_f1 = np.std(all_train_f1s, axis=0)
    std_val_f1 = np.std(all_val_f1s, axis=0)

    # Plotting
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(mean_train_loss, label="Train Loss")
    plt.plot(mean_val_loss, label="Validation Loss")
    plt.fill_between(
        range(len(mean_train_loss)),
        mean_train_loss - std_train_loss,
        mean_train_loss + std_train_loss,
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mean_val_loss)),
        mean_val_loss - std_val_loss,
        mean_val_loss + std_val_loss,
        alpha=0.2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Average Training and validation loss")

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(mean_train_acc, label="Train Accuracy")
    plt.plot(mean_val_acc, label="Validation Accuracy")
    plt.fill_between(
        range(len(mean_train_acc)),
        mean_train_acc - std_train_acc,
        mean_train_acc + std_train_acc,
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mean_val_acc)),
        mean_val_acc - std_val_acc,
        mean_val_acc + std_val_acc,
        alpha=0.2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Average Training and validation accuracy")

    # F1 score plot
    plt.subplot(1, 3, 3)
    plt.plot(mean_train_f1, label="Train F1")
    plt.plot(mean_val_f1, label="Validation F1")
    plt.fill_between(
        range(len(mean_train_f1)),
        mean_train_f1 - std_train_f1,
        mean_train_f1 + std_train_f1,
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mean_val_f1)),
        mean_val_f1 - std_val_f1,
        mean_val_f1 + std_val_f1,
        alpha=0.2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Average Training and validation F1 score")

    plt.tight_layout()
    plt.savefig("plots/training_plot_loso.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model")

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--save_every", type=int, default=51, help="Save checkpoint every n epochs"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
