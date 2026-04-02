import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from lstm.model import FloodLSTM
from lstm.dataset import LSTMDataset


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dataset
    train_dataset = LSTMDataset("data/lstm_data", prefix="train")
    test_dataset  = LSTMDataset("data/lstm_data", prefix="test")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # -- model
    model = FloodLSTM(input_size=10).to(device)

    # 불균형 처리 - 튜닝 가능
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- train
    for epoch in range(10):

        model.train()
        total_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X).squeeze()

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --- evaluation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in test_loader:

                X = X.to(device)

                pred = model(X).squeeze()
                prob = torch.sigmoid(pred).cpu().numpy()

                pred_label = (prob > 0.3).astype(int)

                all_preds.extend(pred_label)
                all_labels.extend(y.numpy())

        f1 = f1_score(all_labels, all_preds)

        print(f"epoch {epoch} | loss={total_loss:.4f} | f1={f1:.4f}")