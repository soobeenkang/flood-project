from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoaders


# =========================
# 학습 설정
# =========================

@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5

    lstm_hidden_dim: int = 64
    lstm_layers: int = 2
    dropout: float = 0.2

    # class imbalance 대응용
    # flooded=1 비율이 낮으면 pos_weight를 주는 게 좋음
    use_pos_weight: bool = True

    # threshold
    threshold: float = 0.5

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 모델
# =========================

class FloodLSTMClassifier(nn.Module):
    """
    x_seq: [B, seq_len, dynamic_dim]
    x_static: [B, static_dim]

    흐름:
    1) dynamic sequence를 LSTM으로 인코딩
    2) 마지막 hidden state와 static feature를 concat
    3) MLP 통과
    4) 이진분류 logit 1개 출력
    """
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=dynamic_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, D]
        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        # 마지막 layer의 hidden state 사용
        seq_feat = h_n[-1]  # [B, hidden_dim]

        static_feat = self.static_proj(x_static)  # [B, hidden_dim]

        combined = torch.cat([seq_feat, static_feat], dim=1)  # [B, hidden_dim * 2]
        logits = self.classifier(combined).squeeze(1)         # [B]

        return logits


# =========================
# metric
# =========================

def binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    logits -> sigmoid -> threshold로 예측 생성
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    targets = targets.long()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =========================
# pos_weight 계산
# =========================

def compute_pos_weight(dataset) -> torch.Tensor:
    """
    BCEWithLogitsLoss(pos_weight=...)용
    pos_weight = negative / positive
    """
    positives = 0
    negatives = 0

    for i in range(len(dataset)):
        y = dataset[i]["y"].item()
        if y >= 0.5:
            positives += 1
        else:
            negatives += 1

    if positives == 0:
        return torch.tensor(1.0)

    pos_weight = negatives / max(positives, 1)
    return torch.tensor(float(pos_weight), dtype=torch.float32)


# =========================
# 1 epoch train
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    threshold: float = 0.5,
) -> dict:
    model.train()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        x_seq = batch["x_seq"].to(device)         # [B, T, D]
        x_static = batch["x_static"].to(device)   # [B, S]
        y = batch["y"].to(device)                 # [B]

        optimizer.zero_grad()

        logits = model(x_seq, x_static)           # [B]
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    metrics = binary_metrics_from_logits(all_logits, all_targets, threshold=threshold)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


# =========================
# 1 epoch valid
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    threshold: float = 0.5,
) -> dict:
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        x_seq = batch["x_seq"].to(device)
        x_static = batch["x_static"].to(device)
        y = batch["y"].to(device)

        logits = model(x_seq, x_static)
        loss = criterion(logits, y)

        total_loss += loss.item() * x_seq.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    metrics = binary_metrics_from_logits(all_logits, all_targets, threshold=threshold)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


# =========================
# 전체 학습
# =========================

def fit_model(
    train_dataset,
    test_dataset,
    dynamic_dim: int,
    static_dim: int,
    config: TrainConfig,
):
    device = config.device

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = FloodLSTMClassifier(
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        hidden_dim=config.lstm_hidden_dim,
        num_layers=config.lstm_layers,
        dropout=config.dropout,
    ).to(device)

    if config.use_pos_weight:
        pos_weight = compute_pos_weight(train_dataset).to(device)
        print(f"[INFO] pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=config.threshold,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        })

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_f1={train_metrics['f1']:.4f}, "
            f"test_loss={test_metrics['loss']:.4f}, "
            f"test_f1={test_metrics['f1']:.4f}, "
            f"test_recall={test_metrics['recall']:.4f}"
        )

        # 침수 탐지는 recall/f1이 중요해서 f1 기준 저장
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    print(f"\n[INFO] Best test f1 = {best_f1:.4f}")

    return model, history


# =========================
# 예시 실행
# =========================

if __name__ == "__main__":
    # 네가 앞에서 만든 코드
    # outputs = make_lstm_datasets(sequence_config)
    #
    # outputs 안에 있다고 가정:
    # train_dataset, test_dataset
    # dynamic_features, static_features

    # 예시:
    # sequence_config = SequenceConfig(...)
    # outputs = make_lstm_datasets(sequence_config)

    train_dataset = outputs["train_dataset"]
    test_dataset = outputs["test_dataset"]

    dynamic_dim = len(outputs["dynamic_features"])
    static_dim = len(outputs["static_features"])

    train_config = TrainConfig(
        batch_size=64,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-5,
        lstm_hidden_dim=64,
        lstm_layers=2,
        dropout=0.2,
        use_pos_weight=True,
        threshold=0.5,
    )

    model, history = fit_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        config=train_config,
    )