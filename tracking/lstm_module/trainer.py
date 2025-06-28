import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Progress bar
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any # 타입 힌트 추가
import warnings # 경고 메시지

# Import necessary components from the module
try:
    from .models import TrackLSTM
    from .losses import giou_loss, ciou_loss
except ImportError: # Fallback for running script directly
    from models import TrackLSTM
    from losses import giou_loss, ciou_loss


# --- 체크포인트 저장/로드 함수 (모델 config 포함 저장) ---
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Saves model checkpoint (model state_dict and config)."""
    print(f"Saving checkpoint to {filepath}...")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 상태는 필요시 저장
        'loss': loss, # 검증 손실 또는 해당 시점 손실
        'config': getattr(model, 'cfg', {}) # 모델 초기화 시 사용된 config 저장
    }
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, filepath, device):
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}. Starting from scratch.")
        return 0, float('inf'), {} # 시작 에포크, 손실, 빈 config 반환

    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)

    # 모델 state_dict 로드 (non-strict 시도)
    try: model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e: warnings.warn(f"Could not load model state_dict: {e}")

    # 옵티마이저 state_dict 로드 (선택 사항)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e: warnings.warn(f"Could not load optimizer state_dict: {e}")

    epoch = checkpoint.get('epoch', -1) + 1
    loss = checkpoint.get('loss', float('inf'))
    loaded_cfg = checkpoint.get('config', {}) # 저장된 config 로드
    print(f"Loaded checkpoint from epoch {epoch-1} with loss {loss:.4f}")

    return epoch, loss, loaded_cfg


# --- 시간적 활성화 규제 (TAR) 함수 ---
def temporal_activation_regularization(lstm_all_outputs: torch.Tensor) -> torch.Tensor:
    """Calculates Temporal Activation Regularization (TAR) loss."""
    if lstm_all_outputs is None or lstm_all_outputs.size(1) < 2:
        return torch.tensor(0.0, device=lstm_all_outputs.device if lstm_all_outputs is not None else 'cpu')
    diff = lstm_all_outputs[:, 1:, :] - lstm_all_outputs[:, :-1, :]
    tar_loss = torch.mean(torch.norm(diff, p=2, dim=2))
    return tar_loss

# --- 메인 학습 함수 ---
def train_lstm(
    model: TrackLSTM,
    train_dataset: Dataset,
    val_dataset: Dataset, # Can be None
    cfg: Dict[str, Any],
    checkpoint_dir: str = "checkpoints"
    ):
    """
    [최종] TrackLSTM 모델을 학습합니다. 조건부 TAR 손실, WeightDrop 비활성화.
    """
    print("--- Starting LSTM Training (Trainer Final) ---")

    # --- Configuration ---
    device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    epochs = cfg['epochs']
    learning_rate = cfg['learning_rate']
    early_stopping_patience = cfg.get('early_stopping_patience', 50)
    loss_type = cfg['loss_type'].lower()
    lambda_mse = cfg['lambda_mse']
    lambda_iou = cfg['lambda_iou']
    lambda_accel = cfg['lambda_accel']
    lambda_tar = cfg.get('lambda_tar', 0.0)
    clip_grad_norm = cfg.get('clip_grad_norm', 1.0)
    output_weights_path = Path(cfg['lstm_weights']) # 최종 best 모델 저장 경로
    window_size = cfg['window_size']
    use_enhanced = cfg.get('use_enhanced_pipeline', False)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = Path(checkpoint_dir) / "lstm_model_best.pth"
    last_checkpoint_path = Path(checkpoint_dir) / "lstm_model_last.pth"

    print(f"Configuration Snippet:")
    print(f"  Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"  Loss: {loss_type.upper()} (IoU Lam: {lambda_iou}, MSE Lam: {lambda_mse}, Accel Lam: {lambda_accel})")
    print(f"  Using Enhanced Pipeline: {use_enhanced}")
    if use_enhanced: print(f"  Temporal Activation Regularization (TAR) Lambda: {lambda_tar}")
    print(f"  Checkpoints Dir: {checkpoint_dir}")
    print(f"  Final Model Path: {output_weights_path}")

    # --- DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type != 'cpu'), drop_last=True)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type != 'cpu'), drop_last=False)
    else:
        warnings.warn("Validation dataset is None or empty. Validation step will be skipped.")


    # --- Model, Optimizer, Criterion ---
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=cfg.get('lr_patience', 15), verbose=True, min_lr=1e-6)
    print(f"Using ReduceLROnPlateau scheduler with patience={cfg.get('lr_patience', 15)}.")
    mse_criterion = nn.MSELoss()
    iou_loss_func = giou_loss if loss_type == 'giou' else ciou_loss

    # --- 손실 기록용 history ---
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [],
        'train_iou': [], 'val_iou': [], 'train_accel': [], 'val_accel': [],
        'train_tar': [], 'val_tar': []
    }
    start_epoch = 0; best_val_loss = float('inf'); early_stopping_counter = 0
    loaded_config_from_ckpt = {} # 체크포인트에서 로드된 설정 저장

    # --- Load Checkpoint ---
    if last_checkpoint_path.exists():
        start_epoch, last_chkpt_loss, loaded_config_from_ckpt = load_checkpoint(model, optimizer, str(last_checkpoint_path), device)
        best_val_loss = last_chkpt_loss
    if best_checkpoint_path.exists():
         _, loss_from_best_ckpt, _ = load_checkpoint(model, None, str(best_checkpoint_path), device)
         best_val_loss = min(best_val_loss, loss_from_best_ckpt)
    # TODO: loaded_config_from_ckpt와 현재 cfg 비교/병합 로직 추가 가능
    print(f"Starting training from epoch {start_epoch}. Initial best validation loss: {best_val_loss:.4f}")
    print(f"Early stopping patience: {early_stopping_patience}")


    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_losses = {'total': 0.0, 'mse': 0.0, 'iou': 0.0, 'accel': 0.0, 'tar': 0.0}
        num_train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for i, (seq_features, target_bbox) in enumerate(pbar):
            seq_features = seq_features.to(device)
            target_bbox = target_bbox.to(device)

            # --- Forward Pass ---
            lstm_out_all, _ = model.lstm(seq_features)
            last_time_step_output = lstm_out_all[:, -1, :]
            pred_delta = model.fc(last_time_step_output)

            # --- Loss Calculation ---
            loss_tar = torch.tensor(0.0, device=device)
            loss_mse = torch.tensor(0.0, device=device)
            loss_iou = torch.tensor(0.0, device=device)
            loss_accel = torch.tensor(0.0, device=device)

            # TAR Loss (조건부)
            if use_enhanced and lambda_tar > 0:
                loss_tar = temporal_activation_regularization(lstm_out_all)

            # MSE & IoU Loss
            try:
                 # 마지막 bbox 특징 추출
                 pos_indices = [train_dataset.feature_cols.index(c) for c in ['x_center', 'y_center', 'w_norm', 'h_norm']]
                 last_bbox_in_seq = seq_features[:, -1, pos_indices]
                 # 실제 델타 계산
                 actual_delta = target_bbox.contiguous() - last_bbox_in_seq.contiguous()
                 # MSE 계산
                 loss_mse = mse_criterion(pred_delta.contiguous(), actual_delta.contiguous())
                 # IoU 계산
                 pred_bbox = last_bbox_in_seq.contiguous() + pred_delta.contiguous()
                 loss_iou = iou_loss_func(pred_bbox, target_bbox)
            except (AttributeError, ValueError, IndexError, RuntimeError) as e: # 모든 예외 처리
                 warnings.warn(f"Error calculating MSE/IoU loss (Batch {i}): {e}. Skipping.")

            # Acceleration Loss
            if lambda_accel > 0 and window_size >= 3:
                 try:
                     pos_indices_accel = [train_dataset.feature_cols.index(c) for c in ['x_center', 'y_center', 'w_norm', 'h_norm']]
                     bboxes_in_seq = seq_features[..., pos_indices_accel]
                     seq_deltas = bboxes_in_seq[:, 1:, :] - bboxes_in_seq[:, :-1, :]
                     full_deltas = torch.cat([pred_delta.unsqueeze(1), seq_deltas], dim=1) # 순서 변경: 예측값 다음 과거값
                     acceleration = full_deltas[:, :-1, :] - full_deltas[:, 1:, :] # 미래->과거 방향으로 차이 계산
                     loss_accel = acceleration.pow(2).mean()
                 except (AttributeError, ValueError, IndexError, RuntimeError) as e:
                      warnings.warn(f"Error calculating Accel loss (Batch {i}): {e}. Skipping.")

            # Total Weighted Loss
            total_loss = (lambda_mse * loss_mse +
                          lambda_iou * loss_iou +
                          lambda_accel * loss_accel)
            if use_enhanced and lambda_tar > 0:
                 total_loss += lambda_tar * loss_tar

            # Backward Pass & Optimization
            optimizer.zero_grad()
            try:
                 total_loss.backward()
            except RuntimeError as e_backward:
                 print(f"\n!!! RuntimeError during backward pass (Batch {i}) !!! Error: {e_backward}")
                 print(f"Loss components: MSE={loss_mse.item():.4f}, IoU={loss_iou.item():.4f}, Accel={loss_accel.item():.4f}, TAR={loss_tar.item():.4f}")
                 # 학습 중단 또는 해당 배치 건너뛰기 결정 필요
                 warnings.warn("Skipping optimizer step due to backward error.")
                 continue # 다음 배치로 (또는 break)

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            # 손실 누적
            epoch_train_losses['total'] += total_loss.item()
            epoch_train_losses['mse'] += loss_mse.item()
            epoch_train_losses['iou'] += loss_iou.item()
            epoch_train_losses['accel'] += loss_accel.item()
            epoch_train_losses['tar'] += loss_tar.item()
            num_train_batches += 1
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", avg=f"{epoch_train_losses['total'] / num_train_batches:.4f}")
        # --- 에포크 Train 완료 ---
        if num_train_batches == 0: # 데이터 로더 문제 등으로 배치가 0개인 경우
            warnings.warn(f"Epoch {epoch+1} [Train] No batches processed. Check dataset and dataloader.")
            continue # 다음 에포크로

        avg_train_loss = epoch_train_losses['total'] / num_train_batches
        avg_train_mse = epoch_train_losses['mse'] / num_train_batches
        avg_train_iou = epoch_train_losses['iou'] / num_train_batches
        avg_train_accel = epoch_train_losses['accel'] / num_train_batches
        avg_train_tar = epoch_train_losses['tar'] / num_train_batches
        print(f"Epoch {epoch+1}/{epochs} [Train] Avg Loss: {avg_train_loss:.4f} (MSE:{avg_train_mse:.4f}, IoU:{avg_train_iou:.4f}, Accel:{avg_train_accel:.4f}, TAR:{avg_train_tar:.4f})")
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss); history['train_mse'].append(avg_train_mse)
        history['train_iou'].append(avg_train_iou); history['train_accel'].append(avg_train_accel)
        history['train_tar'].append(avg_train_tar)

        # --- Validation Step ---
        avg_val_loss = float('inf') # 기본값 초기화
        if val_loader is not None: # 검증 로더 있을 때만 실행
            model.eval()
            epoch_val_losses = {'total': 0.0, 'mse': 0.0, 'iou': 0.0, 'accel': 0.0, 'tar': 0.0}
            num_val_batches = 0
            print(f"--- Starting Validation Step (Epoch {epoch+1}) ---")
            try:
                with torch.no_grad():
                    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                    for val_i, (seq_features, target_bbox) in enumerate(pbar_val):
                        seq_features = seq_features.to(device)
                        target_bbox = target_bbox.to(device)

                        # --- 검증 단계 Forward 및 손실 계산 ---
                        lstm_out_all, _ = model.lstm(seq_features)
                        last_time_step_output = lstm_out_all[:, -1, :]
                        pred_delta = model.fc(last_time_step_output)

                        v_loss_tar = torch.tensor(0.0, device=device)
                        if use_enhanced and lambda_tar > 0:
                            v_loss_tar = temporal_activation_regularization(lstm_out_all)

                        v_loss_mse = torch.tensor(0.0, device=device)
                        v_loss_iou = torch.tensor(0.0, device=device)
                        v_loss_accel = torch.tensor(0.0, device=device)
                        try:
                            pos_indices = [val_dataset.feature_cols.index(c) for c in ['x_center', 'y_center', 'w_norm', 'h_norm']]
                            last_bbox_in_seq = seq_features[:, -1, pos_indices]
                            actual_delta = target_bbox.contiguous() - last_bbox_in_seq.contiguous()
                            v_loss_mse = mse_criterion(pred_delta.contiguous(), actual_delta.contiguous())

                            pred_bbox = last_bbox_in_seq.contiguous() + pred_delta.contiguous()
                            v_loss_iou = iou_loss_func(pred_bbox, target_bbox)
                        except (AttributeError, ValueError, IndexError, RuntimeError) as e:
                             warnings.warn(f"Error calculating Val MSE/IoU loss (Batch {val_i}): {e}. Skipping.")

                        if lambda_accel > 0 and window_size >= 3:
                            try:
                                pos_indices_accel = [val_dataset.feature_cols.index(c) for c in ['x_center', 'y_center', 'w_norm', 'h_norm']]
                                bboxes_in_seq = seq_features[..., pos_indices_accel]
                                seq_deltas = bboxes_in_seq[:, 1:, :] - bboxes_in_seq[:, :-1, :]
                                full_deltas = torch.cat([pred_delta.unsqueeze(1), seq_deltas], dim=1)
                                acceleration = full_deltas[:, :-1, :] - full_deltas[:, 1:, :]
                                v_loss_accel = acceleration.pow(2).mean()
                            except (AttributeError, ValueError, IndexError, RuntimeError) as e:
                                 warnings.warn(f"Error calculating Val Accel loss (Batch {val_i}): {e}. Skipping.")

                        v_total_loss = (lambda_mse * v_loss_mse +
                                        lambda_iou * v_loss_iou +
                                        lambda_accel * v_loss_accel)
                        if use_enhanced and lambda_tar > 0:
                             v_total_loss += lambda_tar * v_loss_tar

                        epoch_val_losses['total'] += v_total_loss.item()
                        epoch_val_losses['mse'] += v_loss_mse.item()
                        epoch_val_losses['iou'] += v_loss_iou.item()
                        epoch_val_losses['accel'] += v_loss_accel.item()
                        epoch_val_losses['tar'] += v_loss_tar.item()
                        num_val_batches += 1
                        pbar_val.set_postfix(loss=f"{v_total_loss.item():.4f}")
                # --- 검증 루프 with torch.no_grad() 종료 ---

            except Exception as e_val_loop: # 데이터 로딩 등 루프 진입 시 오류 처리
                print(f"\n!!! Error during validation loop (Epoch {epoch+1}) !!!")
                print(f"Error: {e_val_loop}")
                avg_val_loss = float('inf') # 검증 실패 처리
            # --- 검증 루프 try 종료 ---

            if num_val_batches > 0:
                avg_val_loss = epoch_val_losses['total'] / num_val_batches
                avg_val_mse = epoch_val_losses['mse'] / num_val_batches
                avg_val_iou = epoch_val_losses['iou'] / num_val_batches
                avg_val_accel = epoch_val_losses['accel'] / num_val_batches
                avg_val_tar = epoch_val_losses['tar'] / num_val_batches
                print(f"Epoch {epoch+1} [Val] Avg Loss: {avg_val_loss:.4f} (MSE:{avg_val_mse:.4f}, IoU:{avg_val_iou:.4f}, Accel:{avg_val_accel:.4f}, TAR:{avg_val_tar:.4f})")
                history['val_loss'].append(avg_val_loss); history['val_mse'].append(avg_val_mse)
                history['val_iou'].append(avg_val_iou); history['val_accel'].append(avg_val_accel)
                history['val_tar'].append(avg_val_tar)
            elif 'avg_val_loss' not in locals() or avg_val_loss == float('inf'): # 처리된 배치 0개 또는 오류 발생 시
                 print(f"Epoch {epoch+1} [Val] Validation did not yield results.")
                 avg_val_loss = float('inf') # 스케줄러/저장 로직 위해 설정
                 history['val_loss'].append(np.nan); history['val_mse'].append(np.nan)
                 history['val_iou'].append(np.nan); history['val_accel'].append(np.nan)
                 history['val_tar'].append(np.nan)
        else: # val_loader가 None인 경우
            print(f"Epoch {epoch+1} [Val] Skipping validation step.")
            avg_val_loss = float('inf') # 검증 안 함


        # --- LR Scheduler & Checkpointing ---
        if avg_val_loss != float('inf') and not np.isnan(avg_val_loss):
             current_lr = optimizer.param_groups[0]['lr']
             scheduler.step(avg_val_loss)
             if optimizer.param_groups[0]['lr'] < current_lr: print("Learning rate reduced.")

             if avg_val_loss < best_val_loss:
                 print(f"Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving best model...")
                 best_val_loss = avg_val_loss
                 save_checkpoint(model, optimizer, epoch, best_val_loss, str(best_checkpoint_path))
                 # 최종 모델 경로에도 저장 (State Dict + Config)
                 torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, output_weights_path)
                 early_stopping_counter = 0
             else:
                 early_stopping_counter += 1
                 print(f"Validation loss did not improve from {best_val_loss:.4f}. EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
        else: # 검증 실패 또는 건너뜀
             # best_val_loss 업데이트 안 함, early stopping 카운터 증가? (정책 결정 필요)
             # early_stopping_counter += 1 # 예: 검증 못하면 개선 안 된 것으로 간주
             print("Skipping best model check and LR scheduler step due to invalid validation loss.")

        # 마지막 체크포인트는 항상 저장 (avg_val_loss가 inf/nan이면 loss=-1.0 저장)
        save_checkpoint(model, optimizer, epoch, avg_val_loss if avg_val_loss != float('inf') and not np.isnan(avg_val_loss) else -1.0, str(last_checkpoint_path))

        # Early stopping 체크
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

    # --- 학습 루프 종료 ---
    print(f"--- Training Finished ---")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Final model state_dict (+config) saved to: {output_weights_path}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Last checkpoint saved to: {last_checkpoint_path}")

    # --- 손실 그래프 그리기 ---
    plot_loss_history(history, output_dir=output_weights_path.parent, cfg=cfg)


# --- 손실 그래프 그리는 함수 (이전 답변과 동일 - NameError 수정됨) ---
def plot_loss_history(history: Dict[str, list], output_dir: Path, cfg: Dict[str, Any]):
    """학습/검증 손실 기록을 그립니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_enhanced = cfg.get('use_enhanced_pipeline', False)
    pipeline_type = 'Enhanced' if use_enhanced else 'Original'
    plot_filename = f"loss_history_{pipeline_type.lower()}.png"
    plot_save_path = output_dir / plot_filename
    print(f"Plotting loss history to {plot_save_path}...")

    epochs_range = history['epoch']
    plot_keys = ['loss', 'mse', 'iou', 'accel', 'tar']
    num_plots = len(plot_keys)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axs = [axs]
    plt.style.use('seaborn-v0_8-darkgrid')

    plot_titles = {'loss': 'Total Loss', 'mse': 'MSE Loss (Delta)',
                   'iou': f"{cfg.get('loss_type', 'iou').upper()} Loss (Box)",
                   'accel': 'Acceleration Loss', 'tar': 'TAR Loss'}
    train_keys = {k: f'train_{k}' for k in plot_keys}
    val_keys = {k: f'val_{k}' for k in plot_keys}

    for i, key in enumerate(plot_keys):
        # 학습 데이터 플롯 (데이터가 있는지 확인)
        if train_keys[key] in history and history[train_keys[key]]:
             axs[i].plot(epochs_range, history[train_keys[key]], label=f'Training {plot_titles[key]}', marker='.')
        # 검증 데이터 플롯 (데이터가 있는지 확인)
        if val_keys[key] in history and history[val_keys[key]] and any(not np.isnan(x) for x in history[val_keys[key]]): # NaN 아닌 값 있을 때만
             axs[i].plot(epochs_range, history[val_keys[key]], label=f'Validation {plot_titles[key]}', marker='.')
        axs[i].set_title(plot_titles[key])
        axs[i].set_ylabel('Loss')
        axs[i].legend(loc='best'); axs[i].grid(True)

    # Total Loss 그래프에 best epoch 표시
    if 'val_loss' in history and history['val_loss'] and any(not np.isnan(x) for x in history['val_loss']):
        try:
            valid_val_loss = [x for x in history['val_loss'] if not np.isnan(x)]
            if valid_val_loss: # NaN 아닌 값이 있을 때만
                 best_epoch_idx_overall = np.nanargmin(history['val_loss'])
                 best_epoch = history['epoch'][best_epoch_idx_overall]
                 min_val_loss = history['val_loss'][best_epoch_idx_overall]
                 axs[0].scatter(best_epoch, min_val_loss, s=100, c='red', marker='*', zorder=5, label=f'Best Val Loss (Ep {best_epoch})')
                 axs[0].legend(loc='best')
        except Exception as e: print(f"Could not plot best epoch marker: {e}")

    axs[-1].set_xlabel('Epoch')
    fig.suptitle(f'LSTM Training Loss History ({pipeline_type} Pipeline)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(plot_save_path)
    plt.close(fig)
    print("Plot saved.")
# --------------------------