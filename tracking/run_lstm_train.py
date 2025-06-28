#!/usr/bin/env python
import yaml
from pathlib import Path
import torch
import pickle
import argparse # 인자 처리 추가
import datetime # 타임스탬프 사용
import warnings
import sys # sys 추가

# Import necessary components from the module
# 경로가 lstm_module 패키지 내에 있다고 가정
try:
    from lstm_module.aug_data_utils import LSTMDataset, FeatureNormalizer
    from lstm_module.models import TrackLSTM
    from lstm_module.trainer import train_lstm
except ImportError: # 예외 처리: 패키지 외부에서 실행 시
    print("Error: Could not import from lstm_module. Make sure the script is run correctly relative to the package.")
    # 현재 디렉토리에서 import 시도 (대안)
    try:
        from aug_data_utils import LSTMDataset, FeatureNormalizer
        from models import TrackLSTM
        from trainer import train_lstm
    except ImportError as e:
        print(f"Failed to import components directly: {e}")
        exit(1)


def main():
    # --- 인자 파서 설정 ---
    parser = argparse.ArgumentParser(description='Train TrackLSTM model.')
    parser.add_argument('--cfg', type=str, default='../boxmot/configs/lstm.yaml',
                        help='Path to the LSTM configuration file (lstm.yaml).')
    parser.add_argument('--output_dir', type=str, default='lstm_checkpoints',
                        help='Base directory to save checkpoints and results.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag to append to the output subdirectory name.')
    # 추가: 특정 체크포인트에서 학습 재개 옵션
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to a specific checkpoint (.pth) to resume training from.')

    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config_path = Path(args.cfg)
    if not config_path.exists(): print(f"Error: Config file not found: {config_path}"); return
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    print(f"Configuration loaded from '{config_path.resolve()}'")
    cfg['config_path'] = str(config_path.resolve())
    use_enhanced = cfg.get('use_enhanced_pipeline', False)
    pipeline_mode = 'enhanced' if use_enhanced else 'original'

    # --- 결과 저장 디렉토리 설정 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(args.output_dir)
    run_name = f"{config_path.stem}_{pipeline_mode}"
    if args.tag: run_name += f"_{args.tag}"
    run_name += f"_{timestamp}"
    checkpoint_dir = output_base_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {checkpoint_dir}")

    # cfg에 최종 저장 경로 업데이트 (trainer에서 사용)
    cfg['lstm_weights'] = str(checkpoint_dir / "lstm_model_best.pth")
    cfg['normalizer_path'] = str(checkpoint_dir / "lstm_normalizer.pkl")

    # --- 2. Prepare Datasets ---
    print("Initializing training dataset...")
    try:
        train_dataset = LSTMDataset(mot_root=cfg['mot_root'], cfg=cfg, normalizer=None, is_train=True)
        if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
        if not hasattr(train_dataset, 'normalizer') or not train_dataset.normalizer.is_fitted:
             raise ValueError("Normalizer not fitted during training dataset init.")
        fitted_normalizer = train_dataset.normalizer
    except Exception as e: print(f"Error initializing training dataset: {e}"); return
    print("Training dataset initialized.")

    print("Initializing validation dataset...")
    val_dataset = None # 기본값 None
    val_mot_root = cfg.get('val_mot_root', cfg.get('mot_root')) # val 없으면 train 경로 사용 (경고 발생 가능)
    if val_mot_root == cfg.get('mot_root'): warnings.warn("Validation data source is the same as training.")
    try:
        val_dataset = LSTMDataset(mot_root=val_mot_root, cfg=cfg, normalizer=fitted_normalizer, is_train=False)
        if len(val_dataset) == 0: warnings.warn("Validation dataset is empty."); val_dataset = None # 비어있으면 None 처리
    except Exception as e: print(f"Warning: Error initializing validation dataset: {e}. Validation will be skipped.")
    if val_dataset: print("Validation dataset initialized.")
    else: print("Validation dataset could not be initialized or is empty.")


    # --- 3. Initialize Model ---
    if 'feat_dim' not in cfg or cfg['feat_dim'] <= 0:
         print(f"Error: Invalid 'feat_dim' ({cfg.get('feat_dim')}) after dataset init.")
         return
    try:
        model = TrackLSTM(cfg=cfg) # feat_dim은 cfg에 이미 업데이트되어 있음
    except Exception as e: print(f"Error initializing TrackLSTM model: {e}"); return
    print(f"LSTM model initialized with feat_dim = {cfg['feat_dim']}.")

    # --- 4. (선택적) 체크포인트 로드하여 학습 재개 ---
    if args.resume_checkpoint:
         resume_path = Path(args.resume_checkpoint)
         if resume_path.exists():
              print(f"\nResuming training from specified checkpoint: {resume_path}")
              # trainer의 load_checkpoint는 모델, 옵티마이저 상태를 로드함
              # 주의: 옵티마이저 상태 로드 시 현재 LR 등 설정과 충돌 가능성 있음
              # 여기서는 모델 가중치만 로드하는 것을 권장 (더 안전)
              try:
                  checkpoint = torch.load(resume_path, map_location=cfg.get('device', 'cpu'))
                  model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                  print(f"Successfully loaded model weights from {resume_path}.")
                  # 필요시 epoch 번호 등 다른 정보도 로드 가능
                  # start_epoch = checkpoint.get('epoch', -1) + 1
              except Exception as e:
                   print(f"Warning: Could not load specified checkpoint {resume_path}: {e}. Starting from scratch.")
         else:
              print(f"Warning: Specified resume checkpoint not found: {resume_path}. Starting from scratch.")

    # --- 5. Start Training ---
    try:
        train_lstm(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset, # None일 수 있음
            cfg=cfg,
            checkpoint_dir=str(checkpoint_dir)
        )
    except Exception as e:
        print(f"\n!!! Error during training process !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return # 오류 발생 시 종료

    # --- 6. Save Fitted Normalizer (최종 확인) ---
    normalizer_save_path = Path(cfg['normalizer_path'])
    if hasattr(train_dataset, 'normalizer') and train_dataset.normalizer.is_fitted:
         if not normalizer_save_path.exists(): # trainer에서 저장 실패 시 대비
              try:
                  with open(normalizer_save_path, 'wb') as f: pickle.dump(train_dataset.normalizer, f)
                  print(f"Saved fitted normalizer to: {normalizer_save_path}")
              except Exception as e: print(f"Error saving normalizer: {e}")
         else: # 이미 저장됨 (정상)
              print(f"Normalizer already saved by trainer at: {normalizer_save_path}")
    else:
         print("Warning: Normalizer was not available or not fitted after training.")


    print(f"\nTraining process finished.")
    print(f"Results saved in directory: {checkpoint_dir}")
    print(f"Best model state_dict (+config) saved to: {cfg['lstm_weights']}")
    print(f"Normalizer saved to: {cfg['normalizer_path']}")

if __name__ == "__main__":
    main()