import os
import sys
import torch
import torch.nn as nn
from collections import deque
import pandas as pd
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- 데이터 로딩 ---
def load_mot_annotations(mot_root: str) -> pd.DataFrame:
    """
    mot_root 아래 각 시퀀스 폴더의 gt/gt.txt를 모두 읽고
    ['seq','frame','track_id','x','y','w','h'] 컬럼의 DataFrame으로 반환.
    """
    dfs = []
    print(f"Loading annotations from: {mot_root}")
    if not Path(mot_root).is_dir():
        print(f"Error: mot_root directory not found: {mot_root}")
        return pd.DataFrame() # 빈 데이터프레임 반환

    for seq_dir in Path(mot_root).iterdir():
        if not seq_dir.is_dir(): # 디렉토리인지 확인
            continue
        gt_path = seq_dir / 'gt' / 'gt.txt'
        print(f"Checking sequence: {seq_dir.name}, GT path: {gt_path}")
        if not gt_path.exists():
            print(f"  gt.txt not found in {seq_dir.name}")
            continue
        try:
            df = pd.read_csv(
                gt_path, header=None,
                names=['frame','track_id','x','y','w','h','conf','cls','vis']
            )
            # 필요한 컬럼만 선택하고 seq 이름 추가
            df = df[['frame','track_id','x','y','w','h']].copy() # SettingWithCopyWarning 방지
            df['seq'] = seq_dir.name
            dfs.append(df)
            print(f"  Loaded {len(df)} annotations from {seq_dir.name}")
        except Exception as e:
            print(f"  Error reading {gt_path}: {e}")

    if not dfs:
        print("No annotations loaded.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# --- 데이터셋 ---
class TrackSequenceDataset(Dataset):
    def __init__(self, annotations, window_size=5):
        """
        annotations: pd.DataFrame with columns ['seq', 'frame', 'track_id', 'x', 'y', 'w', 'h']
        window_size: LSTM 입력 시퀀스 길이
        """
        self.samples = []
        self.window_size = window_size

        if annotations.empty:
            print("Warning: Annotations DataFrame is empty. Dataset will be empty.")
            return

        # track_id 와 seq 기준으로 그룹화
        grouped = annotations.groupby(['seq', 'track_id'])
        print(f"Processing {len(grouped)} tracks...")
        processed_tracks = 0
        for (seq, tid), df in grouped:
            df = df.sort_values('frame')
            # bbox 좌표 (x, y, w, h) 만 사용
            boxes = df[['x', 'y', 'w', 'h']].values  # (T, 4)

            # 시퀀스 생성을 위한 최소 길이 확인
            if len(boxes) < window_size + 1:
                continue

            # 슬라이딩 윈도우 방식으로 샘플 생성
            for i in range(len(boxes) - window_size):
                x_seq = boxes[i:i+window_size]   # 입력 시퀀스 (window_size 길이)
                y_target = boxes[i+window_size]  # 다음 스텝의 bbox (타겟)
                self.samples.append((x_seq, y_target))
            processed_tracks += 1
        print(f"Processed {processed_tracks} tracks. Created {len(self.samples)} samples.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_target = self.samples[idx]
        # numpy 배열을 torch 텐서로 변환
        x_seq = torch.tensor(x_seq, dtype=torch.float32)      # (window_size, 4)
        y_target = torch.tensor(y_target, dtype=torch.float32) # (4,)
        return x_seq, y_target

# --- LSTM 모델 ---
class TrackLSTM(nn.Module):
    def __init__(self, feat_dim=4, hidden_dim=128, num_layers=2, bidirectional=False):
        """
        feat_dim: 입력 특징 차원 (bbox 좌표만 사용하므로 4)
        hidden_dim: LSTM hidden state 차원
        num_layers: LSTM 레이어 수
        bidirectional: 양방향 LSTM 사용 여부
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # 입력 텐서 형태: (batch, seq_len, feature)
            bidirectional=bidirectional,
        )
        # 양방향인 경우 출력 차원은 hidden_dim * 2
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # 출력 레이어: 다음 bbox의 변화량 (delta x, delta y, delta w, delta h) 예측
        self.fc = nn.Linear(out_dim, 4) # 출력 차원 4 (dx, dy, dw, dh)

    def forward(self, seq_coords):
        """
        seq_coords: 입력 시퀀스 (B, window_size, 4)
        """
        lstm_out, _ = self.lstm(seq_coords) # lstm_out: (B, window_size, hidden_dim * num_directions)
        # 마지막 타임 스텝의 출력만 사용
        last_output = lstm_out[:, -1, :]    # (B, hidden_dim * num_directions)
        # 선형 레이어를 통과하여 변화량 예측
        delta_box = self.fc(last_output)    # (B, 4)
        return delta_box

# --- 학습 함수 ---
def train_lstm(loader, model, optimizer, criterion_box, epochs=10, device='cpu'):
    """
    LSTM 모델 학습 함수
    """
    model.to(device)
    model.train() # 모델을 학습 모드로 설정
    print(f"Starting LSTM training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        processed_batches = 0
        for seq, tgt in loader:
            seq, tgt = seq.to(device), tgt.to(device) # 데이터 로더에서 받은 데이터를 device로 이동

            # 모델 예측 (delta box)
            pred_delta = model(seq) # [B, 4]

            # 실제 delta 계산: target_box - last_box_in_sequence
            # seq[:, -1, :] 는 입력 시퀀스의 마지막 bbox (B, 4)
            actual_delta = tgt - seq[:, -1, :]

            # 손실 계산 (MSE Loss)
            loss = criterion_box(pred_delta, actual_delta)

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad() # 그래디언트 초기화
            loss.backward()       # 역전파 수행
            optimizer.step()      # 모델 파라미터 업데이트

            total_loss += loss.item()
            processed_batches += 1

        avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    print("LSTM training finished.")

# --- DeepOCSORT 패치 ---
# boxmot 라이브러리가 설치되어 있다고 가정
try:
    from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort
    from boxmot.utils.matching import NearestNeighborDistanceMetric
    # 필요한 다른 boxmot 구성 요소들을 임포트할 수 있습니다.
except ImportError:
    print("Error: boxmot library not found or DeepOCSort cannot be imported.")
    print("Please install boxmot: pip install boxmot")
    # boxmot 임포트 실패 시 더 이상 진행하지 않도록 처리
    sys.exit(1)

N = 5  # LSTM 입력 시퀀스 길이 (TrackSequenceDataset의 window_size와 일치해야 함)
lstm_model = None # 전역 변수로 선언, 나중에 학습된 모델 할당

# 원본 update 메소드 저장 (패치하기 전에)
# DeepOCSort 클래스가 로드된 후에 실행되어야 함
if 'DeepOCSort' in globals():
    original_update = DeepOCSort.update
else:
    original_update = None # DeepOCSort 로드 실패 시 None

def update_with_lstm(self, dets, img_info, img):
    """
    패치된 DeepOCSort update 메소드.
    원본 update 호출 후 LSTM으로 상태 보정.
    원본 update 메서드의 시그니처를 맞춰야 합니다. boxmot 버전에 따라 다를 수 있음.
    """
    if original_update is None:
         raise RuntimeError("Original DeepOCSort.update method not saved.")

    # 원본 DeepOCSort 업데이트 로직 수행
    ## boxmot 버전에 따라 update 메서드의 인자가 다를 수 있으므로 확인 필요
    ## 예시: original_update(self, dets, img_info, img) 또는 original_update(self, detections)
    ## 아래는 일반적인 인자를 사용하는 예시입니다. 실제 boxmot 버전에 맞게 수정하세요.
    original_update(self, dets, img_info, img) # boxmot 최신 버전 기준 시그니처

    global lstm_model
    if lstm_model is None:
        # print("Warning: LSTM model is not loaded. Skipping LSTM update.")
        return

    # 현재 활성화된 트랙들에 대해 LSTM 보정 적용
    for track in self.tracked_stracks: # 활성 트랙 목록 (속성 이름은 버전에 따라 다를 수 있음)
        # 각 트랙에 bbox 좌표를 저장할 버퍼 초기화 (없으면 생성)
        if not hasattr(track, 'bbox_buffer'):
            track.bbox_buffer = deque(maxlen=N)

        # 현재 트랙의 칼만 필터 상태에서 bbox 추출 (x, y, w, h)
        # 칼만 필터 상태 표현 방식 (예: tlwh, xyah 등) 확인 필요
        # 예시: track.tlwh 사용
        current_bbox = torch.tensor(track.tlwh, dtype=torch.float32) # (4,)

        # 버퍼에 현재 bbox 추가
        track.bbox_buffer.append(current_bbox)

        # 버퍼가 충분히 채워졌으면 LSTM으로 다음 상태 변화량 예측
        if len(track.bbox_buffer) == N:
            # 버퍼의 bbox들을 시퀀스 텐서로 변환 (1, N, 4)
            # .clone().detach() 를 사용하여 그래디언트 흐름 방지 및 원본 데이터 보호
            seq = torch.stack(list(track.bbox_buffer), dim=0).unsqueeze(0).to(next(lstm_model.parameters()).device) # 모델과 같은 device로 이동

            # LSTM 모델로 변화량 예측 (평가 모드)
            lstm_model.eval()
            with torch.no_grad(): # 그래디언트 계산 비활성화
                pred_delta = lstm_model(seq).squeeze(0) # (4,)

            # 예측된 변화량을 현재 칼만 필터 상태에 더하여 보정
            # 주의: 칼만 필터 상태 업데이트 방식 및 의미 고려 필요
            # 예시: 칼만 필터의 mean 상태를 직접 수정
            # track.mean[:4] += pred_delta.cpu().numpy() # 칼만 필터 상태가 numpy 배열일 경우
            # 또는 칼만 필터 객체의 update 메서드를 활용하는 방식 고려

            # --- 칼만 필터 상태 업데이트 (예시) ---
            # DeepOCSORT의 칼만 필터 구현에 따라 수정 필요
            # 여기서는 간단히 numpy 배열로 변환하여 더하는 예시를 보여줍니다.
            # 실제로는 칼만 필터 객체의 상태를 업데이트해야 합니다.
            try:
                # 예시: 칼만 필터 상태가 numpy 배열이고 x,y,w,h 순서라고 가정
                # 실제 DeepOCSort 구현 확인 필요
                current_mean = track.mean[:4].copy() # 현재 상태 복사
                corrected_mean = current_mean + pred_delta.cpu().numpy()

                # 보정된 상태를 다시 칼만 필터에 설정 (이 부분은 KF 구현에 따라 다름)
                # track.mean[:4] = corrected_mean
                # print(f"Track {track.track_id}: Applied LSTM correction.")

                # 중요: 위 예시는 개념 설명용이며, 실제 DeepOCSort의
                # KalmanFilter 클래스 (e.g., boxmot.trackers.kalman_filter.KalmanFilter)의
                # 상태 표현(mean, covariance)과 업데이트 방식을 확인하고
                # 그에 맞게 수정해야 합니다. 직접 mean을 수정하는 것보다
                # KF의 update/predict 메서드를 활용하는 것이 더 안정적일 수 있습니다.
                pass # 실제 KF 업데이트 로직 구현 필요

            except Exception as e:
                 print(f"Error updating Kalman filter state for track {track.track_id}: {e}")

# --- 메인 실행 로직 ---
if __name__ == '__main__':

    # --- 설정 ---
    MOT_ROOT = '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train' # 실제 학습 데이터 경로로 수정
    VIDEO_PATH = 'test_04-01_obj2_30s.avi' # 실제 테스트 비디오 경로로 수정
    LSTM_WEIGHTS_PATH = 'lstm_model_weights_{EPOCHS}.pth' # 학습된 모델 가중치 파일 경로
    WINDOW_SIZE = N # LSTM 윈도우 크기
    BATCH_SIZE = 32
    EPOCHS = 350 # LSTM 학습 에포크
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 사용 가능한 경우 GPU 사용

    # --- 1. 데이터 로드 및 데이터셋/로더 생성 ---
    print("--- Step 1: Loading Data ---")
    annotations = load_mot_annotations(MOT_ROOT)
    if annotations.empty:
        print("Failed to load annotations. Exiting.")
        sys.exit(1)

    dataset = TrackSequenceDataset(annotations, window_size=WINDOW_SIZE)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot train LSTM. Exiting.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset size: {len(dataset)}, Loader ready.")

    # --- 2. LSTM 모델 초기화 및 학습 ---
    print("\n--- Step 2: Training LSTM ---")
    # Bbox 좌표만 사용하므로 feat_dim=4
    lstm_model = TrackLSTM(feat_dim=4, hidden_dim=128, num_layers=2, bidirectional=False)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    criterion_box = nn.MSELoss() # Bbox 예측을 위한 MSE 손실

    train_lstm(loader, lstm_model, optimizer, criterion_box, epochs=EPOCHS, device=DEVICE)

    # 학습된 가중치 저장
    torch.save(lstm_model.state_dict(), LSTM_WEIGHTS_PATH)
    print(f"LSTM model weights saved to {LSTM_WEIGHTS_PATH}")

    # --- 3. 학습된 LSTM 모델 로드 및 DeepOCSORT 패치 ---
    print("\n--- Step 3: Loading Trained LSTM and Patching DeepOCSORT ---")
    # 모델 구조 다시 정의 (동일하게)
    lstm_model = TrackLSTM(feat_dim=4, hidden_dim=128, num_layers=2, bidirectional=False)
    try:
        lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=DEVICE))
        lstm_model.to(DEVICE)
        lstm_model.eval() # 평가 모드로 설정
        print("Trained LSTM model loaded successfully.")

        # DeepOCSort의 update 메소드를 패치
        if 'DeepOCSort' in globals() and original_update is not None:
            DeepOCSort.update = update_with_lstm
            print("DeepOCSort.update method patched successfully.")
        else:
            print("Skipping patching: DeepOCSort class not loaded or original update not saved.")

    except FileNotFoundError:
        print(f"Error: LSTM weights file not found at {LSTM_WEIGHTS_PATH}. Cannot run tracker.")
        lstm_model = None # 모델 로드 실패 시 None으로 설정
        sys.exit(1)
    except Exception as e:
        print(f"Error loading LSTM model or patching DeepOCSort: {e}")
        lstm_model = None
        sys.exit(1)

    # --- 4. 수정된 추적기 실행 ---
    print("\n--- Step 4: Running Tracker ---")
    if lstm_model is not None and 'DeepOCSort' in globals():
        try:
            from boxmot.trackers.deepocsort import deep_ocsort as Tracker
            from boxmot.configs import deepocsort
            # trackers 함수에 필요한 설정값들을 정의합니다.
            # boxmot 버전에 따라 필요한 인자가 다를 수 있습니다.
            args = deepocsort() # boxmot의 기본 설정을 사용하거나 필요한 값을 직접 설정
            args.tracker_type = 'deepocsort' # 추적기 타입 지정
            args.source = VIDEO_PATH       # 비디오 파일 경로
            args.save = True               # 결과 비디오 저장 여부
            args.project = './runs/track'  # 결과 저장 경로
            args.name = 'exp_lstm'         # 실험 이름
            # 필요한 경우 추가적인 DeepOCSort 파라미터 설정
            args.model_weights = 'osnet_ain_x1_0_polyp.pt' # ReID 모델 가중치 (필요 시)
            # args.conf = 0.5 # 탐지 신뢰도 임계값 등

            # 패치된 DeepOCSort 클래스를 직접 사용하도록 tracker 인스턴스 생성
            # tracker 인스턴스를 run_tracker에 전달하는 방식이 더 명확할 수 있습니다.
            # tracker = DeepOCSort(model_weights=args.model_weights, ...) # 필요한 인자 전달

            # trackers 실행 (boxmot 구현에 따라 인자 전달 방식 확인)
            # run_tracker가 내부적으로 Tracker 클래스를 인스턴스화한다면,
            # 패치된 클래스가 사용될 것입니다.
            tracker = Tracker[args.tracker_type](args)

            print("Tracker finished.")

        except ImportError as e:
            print(f"ImportError: {e}")
            print("Try to install the required packages or update boxmot.")
        except Exception as e:
            print(f"Error running tracker: {e}")
    else:
        print("Cannot run tracker because LSTM model is not loaded or DeepOCSort is not patched.")

    # --- (선택적) 패치 복원 ---
    # 프로그램 종료 전에 원본 메소드로 복원 (필요한 경우)
    # if 'DeepOCSort' in globals() and original_update is not None:
    #     DeepOCSort.update = original_update
    #     print("Restored original DeepOCSort.update method.")

