import os
import json
import random
import urllib.request
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ============================================================
# 基本設定
# ============================================================
SEED = 42
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
VAL_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 3
DATA_URL = "https://raw.githubusercontent.com/veripromiseesg/veripromiseesgdataset/ac91c1c8b5d116edf6fc44cccc1ee3b618f5a207/vpesg4ktrain1000v1.json"
DATA_PATH = "vpesg4k_train_1000.json"
OUTPUT_DIR = "outputs_weighted"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
VAL_PRED_PATH = os.path.join(OUTPUT_DIR, "prediction_val.json")
LABEL_DIST_PATH = os.path.join(OUTPUT_DIR, "label_distribution.png")
TEXT_LEN_PATH = os.path.join(OUTPUT_DIR, "text_length_distribution.png")
TRAIN_CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curve.png")
F1_SCORE_PATH = os.path.join(OUTPUT_DIR, "f1_scores.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "final_metrics.json")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")
CLASS_WEIGHT_PATH = os.path.join(OUTPUT_DIR, "class_weights.json")

EVAL_FIELDS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

FIELD_WEIGHTS = {
    "promise_status": 0.20,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
    "verification_timeline": 0.15,
}

label2id = {field: {lab: i for i, lab in enumerate(labels)} for field, labels in EVAL_FIELDS.items()}
id2label = {field: {i: lab for i, lab in enumerate(labels)} for field, labels in EVAL_FIELDS.items()}
num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}


# ============================================================
# 工具函式
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_data_if_needed():
    if not os.path.exists(DATA_PATH):
        print("[1/10] 下載資料集...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"資料已下載：{DATA_PATH}")
    else:
        print(f"[1/10] 已找到資料檔：{DATA_PATH}")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_class_weights(train_data):
    """
    依訓練集計算每個任務的 class weights。
    用法：n_samples / (n_classes * count)
    再做 sqrt 平滑，避免極端少數類別權重過大導致訓練震盪。
    """
    class_weights = {}
    class_weights_serializable = {}

    for field, labels in EVAL_FIELDS.items():
        counts = Counter(item[field] for item in train_data)
        total = sum(counts.values())
        n_classes = len(labels)
        weights = []

        for label in labels:
            count = counts.get(label, 0)
            if count == 0:
                weight = 0.0
            else:
                raw_weight = total / (n_classes * count)
                weight = raw_weight ** 0.5
            weights.append(weight)

        # 讓平均權重約為 1，避免整體 loss 尺度飄太大
        non_zero = [w for w in weights if w > 0]
        mean_non_zero = sum(non_zero) / len(non_zero) if non_zero else 1.0
        normalized = [w / mean_non_zero if w > 0 else 0.0 for w in weights]

        class_weights[field] = torch.tensor(normalized, dtype=torch.float)
        class_weights_serializable[field] = {
            label: round(normalized[idx], 6) for idx, label in enumerate(labels)
        }

    return class_weights, class_weights_serializable


# ============================================================
# Dataset
# ============================================================
class ESGDataset(Dataset):
    def __init__(self, data, tokenizer, label2id_map):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["data"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = {}
        for field, mapping in self.label2id.items():
            labels[field] = mapping[item[field]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = {
        field: torch.tensor([b["labels"][field] for b in batch], dtype=torch.long)
        for field in EVAL_FIELDS
    }
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# 模型
# ============================================================
class MultiTaskBert(nn.Module):
    def __init__(self, num_labels_dict):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleDict({
            field: nn.Linear(hidden_size, n)
            for field, n in num_labels_dict.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        logits = {field: clf(pooled) for field, clf in self.classifiers.items()}
        return logits


# ============================================================
# 訓練 / 預測 / 評估
# ============================================================
def build_criteria(class_weights, device):
    criteria = {}
    for field in EVAL_FIELDS:
        criteria[field] = nn.CrossEntropyLoss(weight=class_weights[field].to(device))
    return criteria


def train_one_epoch(model, dataloader, optimizer, scheduler, criteria, device):
    model.train()
    total_loss = 0.0
    total_task_loss = {field: 0.0 for field in EVAL_FIELDS}

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {field: batch["labels"][field].to(device) for field in EVAL_FIELDS}

        logits = model(input_ids, attention_mask)

        task_losses = {}
        for field in EVAL_FIELDS:
            task_loss = criteria[field](logits[field], labels[field])
            task_losses[field] = task_loss

        # loss 也依照比賽欄位權重加權
        loss = sum(task_losses[field] * FIELD_WEIGHTS[field] for field in EVAL_FIELDS)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        for field in EVAL_FIELDS:
            total_task_loss[field] += task_losses[field].item()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{len(dataloader)} | Weighted Loss = {loss.item():.4f}")

    avg_task_loss = {field: total_task_loss[field] / len(dataloader) for field in EVAL_FIELDS}
    return total_loss / len(dataloader), avg_task_loss


def predict(model, dataloader, device, id2label_map):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                pred = {}
                for field in EVAL_FIELDS:
                    pred_id = logits[field][i].argmax().item()
                    pred[field] = id2label_map[field][pred_id]
                predictions.append(pred)

    return predictions


def evaluate_hybrid(gt_data, pred_data):
    assert len(gt_data) == len(pred_data), f"gt={len(gt_data)} 與 pred={len(pred_data)} 筆數不一致"

    results = {}
    weighted_score = 0.0

    for field, labels in EVAL_FIELDS.items():
        y_true = [item[field] for item in gt_data]
        y_pred = [item[field] for item in pred_data]

        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

        weight = FIELD_WEIGHTS[field]
        weighted_score += macro_f1 * weight

        results[field] = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weight": weight,
            "report": report,
        }

    results["final_weighted_score"] = weighted_score
    return results


# ============================================================
# 視覺化
# ============================================================
def plot_label_distribution(train_df):
    plt.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Label Distribution", fontsize=16, fontweight="bold")

    for idx, (field, labels) in enumerate(EVAL_FIELDS.items()):
        ax = axes[idx // 2][idx % 2]
        counts = Counter(train_df[field])
        ordered_counts = {label: counts.get(label, 0) for label in labels}

        bars = ax.bar(list(ordered_counts.keys()), list(ordered_counts.values()))
        ax.set_title(field, fontsize=12, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

        total = sum(ordered_counts.values())
        for bar, (_, count) in zip(bars, ordered_counts.items()):
            pct = count / total * 100 if total else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(LABEL_DIST_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def plot_text_length(train_df):
    train_df = train_df.copy()
    train_df["text_length"] = train_df["data"].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_df["text_length"], bins=50, alpha=0.8, edgecolor="black")
    axes[0].axvline(train_df["text_length"].mean(), linestyle="--", label=f"mean={train_df['text_length'].mean():.0f}")
    axes[0].axvline(MAX_LEN * 1.5, linestyle="--", label=f"~truncate={MAX_LEN * 1.5:.0f}")
    axes[0].set_title("Text Length Distribution")
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(train_df["text_length"], bins=50, alpha=0.8, edgecolor="black")
    axes[1].axvline(MAX_LEN, linestyle="--", label=f"MAX_LEN={MAX_LEN}")
    axes[1].set_title("Approximate Token Length")
    axes[1].set_xlabel("Approx. Token Count")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(TEXT_LEN_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    truncated = int((train_df["text_length"] > MAX_LEN).sum())
    pct = truncated / len(train_df) * 100 if len(train_df) else 0
    print(f"超過 MAX_LEN={MAX_LEN} 的樣本：{truncated} 筆 ({pct:.1f}%)")


def plot_training_curve(history):
    epochs_range = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(epochs_range), history["loss"], marker="o", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(epochs_range), history["weighted_score"], marker="o", linewidth=2)
    axes[1].set_title("Validation Weighted Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)

    if history["weighted_score"]:
        best_epoch = history["weighted_score"].index(max(history["weighted_score"])) + 1
        axes[1].axvline(best_epoch, linestyle="--", alpha=0.6, label=f"best_epoch={best_epoch}")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(TRAIN_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def plot_f1_scores(final_results):
    fields = list(EVAL_FIELDS.keys())
    macro_f1s = [final_results[f]["macro_f1"] for f in fields]
    micro_f1s = [final_results[f]["micro_f1"] for f in fields]
    weights = [FIELD_WEIGHTS[f] for f in fields]

    x = range(len(fields))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width / 2 for i in x], macro_f1s, width, label="Macro F1", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], micro_f1s, width, label="Micro F1", alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Task")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Task F1 Scores | Final Weighted Score = {final_results['final_weighted_score']:.5f}", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{f}\n(w={w})" for f, w in zip(fields, weights)], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(F1_SCORE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 主程式
# ============================================================
def main():
    print("開始執行 ESG 多任務 BERT 訓練流程（class weights 強化版）")
    ensure_output_dir()
    set_seed(SEED)
    download_data_if_needed()

    print("[2/10] 載入資料...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    train_data, val_data = train_test_split(all_data, test_size=VAL_SIZE, random_state=SEED)
    train_df = pd.DataFrame(train_data)

    print(f"總資料筆數：{len(all_data)}")
    print(f"訓練集：{len(train_data)}")
    print(f"驗證集：{len(val_data)}")

    print("[3/10] 計算 class weights...")
    class_weights, class_weights_serializable = compute_class_weights(train_data)
    save_json(class_weights_serializable, CLASS_WEIGHT_PATH)
    for field in EVAL_FIELDS:
        print(f"  {field}: {class_weights_serializable[field]}")

    print("[4/10] 畫資料分布圖...")
    plot_label_distribution(train_df)
    plot_text_length(train_df)

    print("[5/10] 載入 tokenizer 與建立 DataLoader...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ESGDataset(train_data, tokenizer, label2id)
    val_dataset = ESGDataset(val_data, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    print("[6/10] 建立模型...")
    model = MultiTaskBert(num_labels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型總參數：{total_params:,}")
    print(f"可訓練參數：{trainable_params:,}")

    print("[7/10] 設定 optimizer / scheduler / criteria...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criteria = build_criteria(class_weights, device)

    print("[8/10] 開始訓練...")
    best_score = 0.0
    best_epoch = 0
    no_improve_count = 0
    history = {"loss": [], "weighted_score": [], "task_loss": []}

    for epoch in range(EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        avg_loss, avg_task_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criteria, device)
        history["loss"].append(avg_loss)
        history["task_loss"].append(avg_task_loss)
        print(f"平均 Weighted Loss：{avg_loss:.4f}")
        for field in EVAL_FIELDS:
            print(f"  train loss - {field}: {avg_task_loss[field]:.4f}")

        preds = predict(model, val_loader, device, id2label)
        results = evaluate_hybrid(val_data, preds)
        current_score = results["final_weighted_score"]
        history["weighted_score"].append(current_score)

        print(f"Validation Weighted Score：{current_score:.5f}")
        for field in EVAL_FIELDS:
            print(
                f"  {field}: Macro F1 = {results[field]['macro_f1']:.4f}, "
                f"Micro F1 = {results[field]['micro_f1']:.4f}, "
                f"Weight = {results[field]['weight']}"
            )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"已儲存最佳模型：{MODEL_SAVE_PATH}")
        else:
            no_improve_count += 1
            print(f"本輪未刷新最佳分數，early stopping counter = {no_improve_count}/{EARLY_STOPPING_PATIENCE}")

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print("觸發 early stopping，提前結束訓練。")
            break

    print("[9/10] 繪製結果圖並做最終評估...")
    plot_training_curve(history)
    save_json(history, HISTORY_PATH)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    final_preds = predict(model, val_loader, device, id2label)
    final_results = evaluate_hybrid(val_data, final_preds)
    final_results["best_epoch"] = best_epoch
    plot_f1_scores(final_results)
    save_json(final_results, METRICS_PATH)

    print("\n最終評估結果")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    for field in EVAL_FIELDS:
        r = final_results[field]
        print(f"\n--- {field} (weight={r['weight']}) ---")
        print(f"Macro F1: {r['macro_f1']:.4f}")
        print(f"Micro F1: {r['micro_f1']:.4f}")
    print("=" * 60)
    print(f"Final Weighted Score: {final_results['final_weighted_score']:.5f}")
    print("=" * 60)

    print("[10/10] 輸出驗證集預測檔...")
    output_data = []
    for orig, pred in zip(val_data, final_preds):
        item = dict(orig)
        item.update(pred)
        output_data.append(item)
    save_json(output_data, VAL_PRED_PATH)

    print("\n全部完成，輸出檔如下：")
    print(f"- 最佳模型：{MODEL_SAVE_PATH}")
    print(f"- 驗證集預測：{VAL_PRED_PATH}")
    print(f"- 指標：{METRICS_PATH}")
    print(f"- 訓練歷史：{HISTORY_PATH}")
    print(f"- 類別權重：{CLASS_WEIGHT_PATH}")
    print(f"- 標籤分布圖：{LABEL_DIST_PATH}")
    print(f"- 文字長度圖：{TEXT_LEN_PATH}")
    print(f"- 訓練曲線圖：{TRAIN_CURVE_PATH}")
    print(f"- F1 圖：{F1_SCORE_PATH}")


if __name__ == "__main__":
    main()
