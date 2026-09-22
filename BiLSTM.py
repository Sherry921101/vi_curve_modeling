import argparse
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import decimate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. Data loading -- VARIABLE length episodes
# ==========================================
def load_episode_files(file_dir, pattern, u_col, i_col, event_col, downsample_factor=1):
    """Each raw CSV file contains multiple events marked row-by-row via `event_col`.
    We group rows by that column to split each file into its individual events (episodes).
    """
    paths = sorted(glob.glob(os.path.join(file_dir, pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No files matched '{pattern}' in directory '{file_dir}'.\n"
            f"Please ensure your dataset files are placed in '{file_dir}' "
            f"or specify a custom path using --data_dir <path>."
        )

    U_list, I_list, episode_ids = [], [], []
    for p in paths:
        print(f"  Reading: {os.path.basename(p)} ...", flush=True)
        df = pd.read_csv(p)
        print(f"  -> {len(df):,} rows loaded. Checking columns...", flush=True)
        if event_col not in df.columns:
            raise KeyError(
                f"Column '{event_col}' not found in {p}. "
                f"Available columns: {list(df.columns)}"
            )
        if u_col not in df.columns or i_col not in df.columns:
            raise KeyError(
                f"Columns '{u_col}' or '{i_col}' not found in {p}. "
                f"Available columns: {list(df.columns)}"
            )

        file_tag = os.path.splitext(os.path.basename(p))[0]
        print(f"  -> Splitting into events via groupby...", flush=True)

        for event_no, group in df.groupby(event_col, sort=False):
            u = group[u_col].values.astype(np.float64)  # decimate requires float64
            i = group[i_col].values.astype(np.float64)
            if downsample_factor > 1 and len(u) > downsample_factor * 10:
                u = decimate(u, downsample_factor, ftype='fir', zero_phase=True)
                i = decimate(i, downsample_factor, ftype='fir', zero_phase=True)
            U_list.append(u.astype(np.float32))
            I_list.append(i.astype(np.float32))
            episode_ids.append(f"{file_tag}_event{event_no}")
        print(f"  -> Done: {len([e for e in episode_ids if file_tag in e])} events from this file.", flush=True)

    lengths = [len(u) for u in U_list]
    print(f"Loaded {len(U_list)} episodes (events) from {len(paths)} files.")
    print(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    return U_list, I_list, episode_ids, lengths


class EpisodeDataset(Dataset):
    """Each item is one full episode at its original length."""
    def __init__(self, I_list, U_list):
        self.I = [torch.tensor(i, dtype=torch.float32).unsqueeze(-1) for i in I_list]  # (T_i, 1)
        self.U = [torch.tensor(u, dtype=torch.float32).unsqueeze(-1) for u in U_list]

    def __len__(self):
        return len(self.I)

    def __getitem__(self, idx):
        return self.I[idx], self.U[idx]


def collate_variable_length(batch):
    """Pads sequences within the batch to the batch's max length and provides a binary mask."""
    i_seqs, u_seqs = zip(*batch)
    lengths = torch.tensor([len(s) for s in i_seqs], dtype=torch.long)

    i_padded = pad_sequence(i_seqs, batch_first=True, padding_value=0.0)  # (batch, T_max, 1)
    u_padded = pad_sequence(u_seqs, batch_first=True, padding_value=0.0)

    T_max = i_padded.size(1)
    mask = (torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)

    return i_padded, u_padded, lengths, mask


# ==========================================
# 2. Model: BiLSTM with variable-length handling
# ==========================================
class BiLSTMSurrogate(nn.Module):
    """Bidirectional LSTM Surrogate model for V-I mapping."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, i_seq, lengths):
        # i_seq: (batch, T_max, 1); lengths: (batch,)
        packed = pack_padded_sequence(i_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=i_seq.size(1))
        u_pred = self.fc(out)  # (batch, T_max, 1)
        return u_pred


def masked_mse(pred, target, mask):
    """MSE calculated only across valid (unpadded) timesteps."""
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1.0)


# ==========================================
# 3. Training & Evaluation
# ==========================================
def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training {model.__class__.__name__} (epochs={epochs}, device={device}) ---", flush=True)
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_bar:
        # ---- Train ----
        model.train()
        running, running_count = 0.0, 0
        t0 = time.time()
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{epochs} [train]",
            unit="batch",
            leave=False,
            position=1,
        )
        for bi, bu, lengths, mask in batch_bar:
            bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(bi, lengths)
            loss = masked_mse(pred, bu, mask)
            loss.backward()
            optimizer.step()
            running += loss.item() * bi.size(0)
            running_count += bi.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.5f}", seq_len=bi.size(1))
        train_loss = running / running_count
        batch_bar.close()

        # ---- Validate ----
        model.eval()
        val_running, val_count = 0.0, 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1}/{epochs} [val]  ",
                unit="batch",
                leave=False,
                position=1,
            )
            for bi, bu, lengths, mask in val_bar:
                bi, bu, mask = bi.to(device), bu.to(device), mask.to(device)
                pred = model(bi, lengths)
                loss = masked_mse(pred, bu, mask)
                val_running += loss.item() * bi.size(0)
                val_count += bi.size(0)
                val_bar.set_postfix(val_loss=f"{loss.item():.5f}")
            val_bar.close()
        val_loss = val_running / val_count

        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            train=f"{train_loss:.5f}",
            val=f"{val_loss:.5f}",
            sec=f"{elapsed:.1f}s",
        )
        tqdm.write(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"Train MSE: {train_loss:.5f}  Val MSE: {val_loss:.5f}  "
            f"({elapsed:.1f}s)",
            file=sys.stdout,
        )

    epoch_bar.close()
    return model


def predict_per_episode(model, loader, device):
    """Returns predictions per episode, trimmed to true length."""
    model.eval()
    preds = []
    with torch.no_grad():
        for bi, _, lengths, _ in tqdm(loader, desc="Predicting", unit="batch"):
            bi = bi.to(device)
            out = model(bi, lengths).cpu().numpy()
            for b in range(out.shape[0]):
                preds.append(out[b, :lengths[b], :])
    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM Surrogate Model for V-I Curve Modeling")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing CSV files (default: auto-detect B6031600_event_all or data/)")
    parser.add_argument("--file_pattern", type=str, default="B6031600_*.csv",
                        help="Glob pattern for event CSV files (default: 'B6031600_*.csv')")
    parser.add_argument("--u_col", type=str, default="U1",
                        help="Target voltage column name (default: 'U1')")
    parser.add_argument("--i_col", type=str, default="I1",
                        help="Input current column name (default: 'I1')")
    parser.add_argument("--event_col", type=str, default="EventNo",
                        help="Event identifier column name (default: 'EventNo')")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsampling factor using anti-aliasing decimate (default: 1)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer (default: 0.001)")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Hidden units in BiLSTM (default: 32)")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of stacked BiLSTM layers (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate between layers (default: 0.2)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save output plots and metrics (default: 'results')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def resolve_data_dir(user_specified_dir):
    if user_specified_dir:
        return user_specified_dir
    # Auto-detection priority:
    candidates = ["./B6031600_event_all", "./data", "."]
    for c in candidates:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "B6031600_*.csv"))) > 0:
            return c
    return "./data"


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("Loading and splitting files into events...", flush=True)

    U_list, I_list, episode_ids, lengths = load_episode_files(
        data_dir, args.file_pattern, args.u_col, args.i_col, args.event_col, args.downsample
    )
    n_episodes = len(U_list)

    # Train / Val / Test split (file/episode level)
    idx = np.arange(n_episodes)
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test episodes")

    # Fit scalers on TRAINING episodes only
    print("Fitting scalers...", flush=True)
    train_U_concat = np.concatenate([U_list[i] for i in train_idx]).reshape(-1, 1)
    train_I_concat = np.concatenate([I_list[i] for i in train_idx]).reshape(-1, 1)

    scaler_U = StandardScaler().fit(train_U_concat)
    scaler_I = StandardScaler().fit(train_I_concat)
    print("Scalers fitted. Scaling all episodes...", flush=True)

    def scale_list(arr_list, scaler):
        return [scaler.transform(a.reshape(-1, 1)).flatten() for a in arr_list]

    U_scaled_all = scale_list(U_list, scaler_U)
    I_scaled_all = scale_list(I_list, scaler_I)

    def subset(lst, indices):
        return [lst[i] for i in indices]

    train_ds = EpisodeDataset(subset(I_scaled_all, train_idx), subset(U_scaled_all, train_idx))
    val_ds = EpisodeDataset(subset(I_scaled_all, val_idx), subset(U_scaled_all, val_idx))
    test_ds = EpisodeDataset(subset(I_scaled_all, test_idx), subset(U_scaled_all, test_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_variable_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Baseline: pointwise static linear regression (memory-less)
    lin_reg = LinearRegression()
    lin_reg.fit(train_I_concat, train_U_concat)

    # Initialize and train BiLSTM model
    model = BiLSTMSurrogate(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    model = train_model(model, train_loader, val_loader, device, args.epochs, lr=args.lr)

    # Save model weights
    model_path = os.path.join(args.output_dir, "vi_bilstm_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")

    # Evaluate on test set
    preds_scaled = predict_per_episode(model, test_loader, device)

    per_file_mse_model, per_file_mse_baseline = [], []
    results_rows = []
    for local_i, global_i in tqdm(enumerate(test_idx), total=len(test_idx), desc="Evaluating", unit="ep"):
        u_true_scaled = U_scaled_all[global_i].reshape(-1, 1)
        i_true_scaled = I_scaled_all[global_i].reshape(-1, 1)

        u_true = scaler_U.inverse_transform(u_true_scaled).flatten()
        u_pred = scaler_U.inverse_transform(preds_scaled[local_i]).flatten()
        u_baseline = lin_reg.predict(scaler_I.inverse_transform(i_true_scaled)).flatten()

        model_mse = np.mean((u_pred - u_true) ** 2)
        baseline_mse = np.mean((u_baseline - u_true) ** 2)
        per_file_mse_model.append(model_mse)
        per_file_mse_baseline.append(baseline_mse)
        results_rows.append({
            "episode_id": episode_ids[global_i],
            "length": lengths[global_i],
            "baseline_mse": baseline_mse,
            "bilstm_mse": model_mse,
        })

    per_file_mse_model = np.array(per_file_mse_model)
    per_file_mse_baseline = np.array(per_file_mse_baseline)

    print("\n=== Test-set Accuracy: BiLSTM vs. Static Linear Baseline ===")
    overall_baseline = per_file_mse_baseline.mean()
    overall_model = per_file_mse_model.mean()
    improvement = (1 - overall_model / overall_baseline) * 100 if overall_baseline > 0 else float("nan")
    print(f"Static linear baseline  mean per-episode MSE: {overall_baseline:.5f}")
    print(f"BiLSTM                  mean per-episode MSE: {overall_model:.5f}  (vs baseline: {improvement:+.1f}%)")
    print(f"BiLSTM per-episode MSE: worst={per_file_mse_model.max():.5f}")

    results_csv_path = os.path.join(args.output_dir, "vi_bilstm_test_results.csv")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Per-episode test results saved to: {results_csv_path}")

    # Plot sample test episodes
    example_local = {
        "typical": np.argsort(per_file_mse_model)[len(per_file_mse_model) // 2],
        "worst_case": np.argmax(per_file_mse_model),
    }

    for label, local_idx in example_local.items():
        global_i = test_idx[local_idx]
        u_true = scaler_U.inverse_transform(U_scaled_all[global_i].reshape(-1, 1)).flatten()
        u_pred = scaler_U.inverse_transform(preds_scaled[local_idx]).flatten()
        i_vals = scaler_I.inverse_transform(I_scaled_all[global_i].reshape(-1, 1)).flatten()
        u_baseline = lin_reg.predict(i_vals.reshape(-1, 1)).flatten()

        plt.figure(figsize=(14, 5))
        plt.plot(u_true, label="True U", color="black", linewidth=2)
        plt.plot(u_pred, label="BiLSTM prediction", color="tab:blue", linestyle="--", linewidth=2)
        plt.plot(u_baseline, label="Static linear baseline", color="gray", linestyle=":", linewidth=1.5)
        eid = episode_ids[global_i]
        plt.title(f"BiLSTM - {label} episode ({eid}, length={lengths[global_i]})")
        plt.xlabel("Sample index within episode")
        plt.ylabel("Voltage (U1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_name = os.path.join(args.output_dir, f"vi_bilstm_example_{label}.png")
        plt.savefig(out_name, dpi=300)
        print(f"Plot saved as: {out_name}")
        plt.close()


if __name__ == "__main__":
    main()