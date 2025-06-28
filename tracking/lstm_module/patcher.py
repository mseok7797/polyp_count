import torch
import numpy as np
from collections import deque
import pickle
import warnings
from pathlib import Path # Path import 추가

# --- 모듈 레벨 변수 (run_lstm_track.py에서 설정됨) ---
lstm_model = None             # 로드된 LSTM 모델 객체
loaded_normalizer_obj = None  # 로드된 FeatureNormalizer 객체
device = None                 # 사용할 장치 ('cpu' or 'cuda:X')
window_size = 10              # LSTM 입력 시퀀스 길이
correction_weight = 0.5       # LSTM 예측 융합 가중치
fps = 30                      # 영상 FPS (특징 계산 시 참고 가능)
feat_dim = 8                  # LSTM 입력 특징 차원 (normalizer에서 확인 가능)

# --- 원본 KF Predict 메소드 저장을 위한 딕셔너리 ---
# {kf_class: original_predict_method} 형태로 저장됨
_original_kf_predict = {}

# --- Helper 함수: Track 객체에서 KF 인스턴스 가져오기 ---
def get_kf_from_track(track):
    """KalmanBoxTracker 인스턴스에서 KalmanFilter 객체를 반환합니다."""
    # tracker 구현에 따라 kf 속성 이름이 다를 수 있음 (예: kf, kalman_filter)
    if hasattr(track, 'kf') and track.kf is not None:
        return track.kf
    elif hasattr(track, 'kalman_filter') and track.kalman_filter is not None:
        return track.kalman_filter
    # 다른 가능한 속성 이름 추가
    return None

# --- LSTM 예측을 융합하는 Patched KF Predict 함수 ---
def predict_with_lstm(kf_instance):
    """
    Kalman Filter의 predict 메소드를 대체하여 LSTM 예측을 융합합니다.
    이 함수는 KF 클래스의 predict 메소드로 직접 설정됩니다.
    """
    global _original_kf_predict

    # --- >>> 함수 호출 확인 로그 추가 <<< ---
    track_info = "Unknown Track"
    parent_track_for_log = getattr(kf_instance, 'parent_track', None)
    if parent_track_for_log:
        track_info = f"Track ID: {getattr(parent_track_for_log, 'track_id', getattr(parent_track_for_log, 'id', 'N/A'))}"
    print(f"--- DEBUG: predict_with_lstm called for KF of {track_info} ---")
    # --- >>> 로그 추가 끝 <<< ---

    # --- 1. 원본 KF predict 실행 ---
    original_predict_func = None
    kf_class = kf_instance.__class__
    if kf_class in _original_kf_predict:
        original_predict_func = _original_kf_predict[kf_class]

    if original_predict_func:
        # 원본 predict 호출 (self 인자로 kf_instance 전달)
        original_predict_func(kf_instance)
    else:
        # 원본 함수를 찾지 못한 경우 (이론상 발생하면 안 됨)
        warnings.warn(f"Original predict function not found for {kf_class.__name__} in patcher. Skipping original KF predict.")
        # 여기서 return 하거나, KF 상태를 직접 업데이트 (예: A @ x)
        # 일단 경고만 하고 진행 (KF 예측이 안 될 수 있음)
        pass # 또는 기본 예측 수행: kf_instance.x = kf_instance.F @ kf_instance.x ...

    # --- 2. LSTM 예측 수행 및 융합 ---
    # 모듈 변수에 필요한 객체들이 설정되어 있는지 확인
    if lstm_model is None or loaded_normalizer_obj is None or device is None:
        # --- >>> 조기 리턴 로그 추가 <<< ---
        print(f"DEBUG: predict_with_lstm returning early (LSTM components not ready).")
        # --- >>> 로그 추가 끝 <<< ---
        return

    

    # --- 2.1 부모 Track 객체 찾기 ---
    # KF 인스턴스가 부모 Track 참조를 가지고 있다고 가정 (가장 안정적)
    # 예: tracker 구현 시 kf = KalmanFilter(parent_track=self) 처럼 설정
    track = getattr(kf_instance, 'parent_track', None)
    if track is None:
         # --- >>> 조기 리턴 로그 추가 <<< ---
        print(f"DEBUG: predict_with_lstm returning early (parent track not found).")
        # --- >>> 로그 추가 끝 <<< ---
        return

    # --- 2.2 특징 버퍼 확인 ---
    if not hasattr(track, 'features_buffer') or len(track.features_buffer) < window_size:
        buffer_len = len(getattr(track,'features_buffer',[]))
        print(f"DEBUG: predict_with_lstm returning early (buffer length {buffer_len} < {window_size}).")
        # --- >>> 로그 추가 끝 <<< ---
        return
    # --- 2.3 LSTM 입력 시퀀스 준비 ---
    # 버퍼에서 마지막 N개 특징 추출
    # buffer는 (feat_dim,) 형태의 numpy 배열을 저장한다고 가정
    try:
        raw_features_sequence = np.array(list(track.features_buffer))[-window_size:] # (window_size, feat_dim)
    except Exception as e:
        warnings.warn(f"Error creating sequence from buffer for track {getattr(track, 'id', '?')}: {e}. Skipping LSTM.")
        return
    # ... (특징 시퀀스 생성) ...
    if raw_features_sequence is None: return
    
    # --- 2.4 특징 정규화 ---
    seq_tensor = None
    if loaded_normalizer_obj.is_fitted:
         try:
             # Normalizer 입력 형태는 (samples, time_steps, features)
             if raw_features_sequence.ndim == 2:
                 raw_features_sequence_batch = raw_features_sequence[np.newaxis, :, :] # (1, window_size, feat_dim)
             else: # 이미 배치 차원이 있다면 그대로 사용 (일반적이지 않음)
                  raw_features_sequence_batch = raw_features_sequence

             # Normalizer의 feature_cols 순서대로 입력이 구성되었다고 가정
             norm_features_sequence = loaded_normalizer_obj.transform(raw_features_sequence_batch)
             seq_tensor = torch.tensor(norm_features_sequence, dtype=torch.float32).to(device)
         except Exception as e:
             warnings.warn(f"Normalizer transform error for track {getattr(track, 'id', '?')}: {e}. Skipping LSTM.")
             return
    else:
         # Normalizer가 fit되지 않았으면 LSTM 사용 불가
         warnings.warn("Normalizer is not fitted. Skipping LSTM correction.")
         return
    if seq_tensor is None: 
        # --- >>> 조기 리턴 로그 추가 <<< ---
        print(f"DEBUG: predict_with_lstm returning early (normalization failed).")
        # --- >>> 로그 추가 끝 <<< ---
        return # 정규화 실패 시 return

    # --- 2.5 LSTM 예측 (Delta) ---
    try:
        with torch.no_grad():
            # 모델 forward 호출
            lstm_pred_delta_norm = lstm_model(seq_tensor).squeeze(0).cpu().numpy() # (feat_dim,) 또는 (4,) - 모델 출력 확인 필요
            # 모델이 delta (dx, dy, dw, dh) 4개 값만 출력한다고 가정
            if lstm_pred_delta_norm.shape[0] != 4:
                 warnings.warn(f"LSTM output dim is {lstm_pred_delta_norm.shape[0]}, expected 4 (delta). Skipping LSTM.")
                 return
    except Exception as e:
         warnings.warn(f"LSTM forward pass error for track {getattr(track, 'id', '?')}: {e}. Skipping LSTM.")
         return

    # --- 2.6 예측 융합 (Kalman Filter 상태 업데이트) ---
    try:
        if current_bbox_norm is None: return # 계산 실패 시
        # 현재 KF 상태 (predict 직후 상태)
        kf_state = kf_instance.x.copy()
        cx, cy, s, r = kf_state[:4] # x, y, area, aspect_ratio
        # NaN/inf 방지 및 bbox 계산
        s_eps, r_eps = max(s, 1e-6), max(r, 1e-6)
        w = np.sqrt(s_eps * r_eps)
        h = np.sqrt(s_eps / r_eps)

        # 현재 bbox 특징 (정규화된 값) - Normalizer 정보 사용
        current_bbox_norm = np.zeros(4)
        img_w = loaded_normalizer_obj.img_width
        img_h = loaded_normalizer_obj.img_height
        if img_w and img_h:
             current_bbox_norm[0] = (cx) / img_w  # x_center
             current_bbox_norm[1] = (cy) / img_h  # y_center
             current_bbox_norm[2] = w / img_w     # w_norm
             current_bbox_norm[3] = h / img_h     # h_norm
        else:
             warnings.warn("Image dimensions not found in normalizer. Cannot get normalized current bbox. Skipping LSTM.")
             return
        
        # debug 
        track_id_for_log = getattr(track, 'track_id', getattr(track, 'id', 'N/A')) # 트랙 ID 가져오기 시도
        # 소수점 아래 4자리까지만 출력하도록 포맷팅
        current_bbox_str = np.array2string(current_bbox_norm, precision=4, suppress_small=True)
        lstm_pred_str = np.array2string(lstm_pred_delta_norm, precision=4, suppress_small=True)
        print(f"--- LSTM Correction Debug (Track ID: {track_id_for_log}) ---")
        print(f"  KF Predict (Norm):  [xc, yc, wn, hn] = {current_bbox_str}")
        print(f"  LSTM Predict Delta: [dx, dy, dw, dh] = {lstm_pred_str}")

        # 정규화된 delta 예측값과 가중치 적용하여 새로운 bbox (정규화 상태) 계산
        # lstm_pred_delta_norm shape: (4,) - dx_n, dy_n, dw_n, dh_n
        corrected_bbox_norm = current_bbox_norm + lstm_pred_delta_norm * correction_weight

        corrected_bbox_str = np.array2string(corrected_bbox_norm, precision=4, suppress_small=True)
        print(f"  Corrected BBox (Norm): [xc, yc, wn, hn] = {corrected_bbox_str} (Weight: {correction_weight})")

        # 융합된 정규화 bbox에서 새로운 cx, cy, w, h (원본 스케일) 계산
        new_cx_norm, new_cy_norm, new_w_norm, new_h_norm = corrected_bbox_norm
        new_w = new_w_norm * img_w
        new_h = new_h_norm * img_h
        new_cx = new_cx_norm * img_w
        new_cy = new_cy_norm * img_h

        # Kalman Filter 상태 업데이트 (x, y, s, r)
        new_w_eps, new_h_eps = max(new_w, 1e-6), max(new_h, 1e-6) # 0 방지
        new_s = new_w_eps * new_h_eps # area
        new_r = new_w_eps / new_h_eps # aspect ratio

        kf_instance.x[0] = new_cx
        kf_instance.x[1] = new_cy
        kf_instance.x[2] = new_s
        kf_instance.x[3] = new_r
        # 속도(vx, vy)는 KF 예측값 유지 또는 LSTM 델타 반영 등 추가 로직 가능
        # 여기서는 위치/크기만 업데이트

    except AttributeError as e:
        warnings.warn(f"Attribute error during state fusion (maybe normalizer issue?): {e}. Skipping LSTM.")
    except Exception as e:
        warnings.warn(f"Error during LSTM state fusion for track {getattr(track, 'id', '?')}: {e}. Skipping LSTM correction.")
        # import traceback
        # traceback.print_exc() # 디버깅 시 사용

# --- prepare_lstm_patching 함수는 삭제 ---