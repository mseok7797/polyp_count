from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort
from boxmot.utils import ROOT, WEIGHTS, logger as LOGGER
import torch
from DeepOCSORT_LSTM_Train import TrackLSTM
from boxmot.tracking import track

tracker = DeepOCSort(...)

tracker.model.lstm_model = lstm_model

'''
# 1) LSTM 모델 정의 (unchanged)
device_for_load = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 2) 모델 인스턴스 생성 (feat_dim, hidden_dim 등은 본인이 학습한 값)
# 기본 LSTM 실행 : feat_dim = 4 + 128
lstm_model = TrackLSTM(feat_dim=4, hidden_dim=128, num_layers=2, bidirectional=False)
# 3) : pass a torch.device or string, not an int
checkpoint = torch.load('lstm_model_weights.pth', map_location=device_for_load)
lstm_model.load_state_dict(checkpoint)
# 4) 모델을 해당 디바이스로 올리고 eval 모드
lstm_model.to(device_for_load).eval()

# 5) DeepOCSort expects an int or string for `device`, so keep that separate:
tracker_device = 0  # GPU #0
# 6) Tracker 인스턴트화 -> Class 상속으로 5) Tracker 설정 포함
tracker = DeepOCSort(
    model_weights=WEIGHTS / 'osnet_ain_x1_0_msmt17.pt',
    device             = tracker_device,  # integer here
    fp16               = False,
    tracker_type       = 'deepocsort',
    with_reid          = False,
    track_high_thresh  = 0.6,
    track_low_thresh   = 0.1,
    new_track_thresh   = 0.7,
    track_buffer       = 60,
    match_thresh       = 0.8,
    proximity_thresh   = 0.5,
    appearance_thresh  = 0.25,
    with_orientation   = True,
    with_angle         = False,
    # you can still pass your lstm kwargs here
    lstm_weights       = 'lstm_model_weights.pth',
    lstm_hidden_size   = 128,
)
# 7) LSTM 주입
tracker.model.lstm_model = lstm_model

try:
    # 8) --- Step 4: Running Tracker with LSTM ---
    print("\n--- Step 4: Running Tracker ---")
    VIDEO_PATH = './test_04-01_obj2_30s.avi'
    results = tracker.run(
            source=VIDEO_PATH,
            save=True,
            save_dir='./runs/track',
            exist_ok=True
            )
    print(f"Tracker finished. Results saved to {results.save_dir}")     

except Exception as e:
    print(f"Error running tracker: {e}")

'''