# run_lstm_track.py

import yaml
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import random
import os
import re
import argparse # 인자 처리
import datetime # 타임스탬프
import warnings
import pickle
from collections import deque # patcher 특징 버퍼 위해 import 필요

# --- Ultralytics YOLOv8 Import ---
try:
    from ultralytics import YOLO
    print("Ultralytics YOLO library imported successfully.")
except ImportError:
    print("Error: Ultralytics YOLO library not found. Please install it (`pip install ultralytics`)")
    YOLO = None

# --- LSTM Module Imports ---
# 패키지 구조 가정, 필요시 경로 수정
try:
    import lstm_module.patcher as lstm_patcher # patcher 모듈 import
    from lstm_module.metrics import compute_mot_metrics
    from lstm_module.models import TrackLSTM # 모델 클래스 import
    from lstm_module.aug_data_utils import FeatureNormalizer # Normalizer 클래스 import
except ImportError:
    print("Error: Could not import from lstm_module. Check paths and structure.")
    # 패키지 외부 실행 시 대안
    try:
        import patcher as lstm_patcher
        from metrics import compute_mot_metrics
        from models import TrackLSTM
        from aug_data_utils import FeatureNormalizer
    except ImportError as e:
        print(f"Failed to import components directly: {e}")
        exit(1)

# --- Tracker Import (예: DeepOCSort) ---
# 실제 사용하는 Tracker로 경로 및 클래스 수정 필요
try:
    # from boxmot import DeepOCSort # BoxMOT 라이브러리 사용 시
    from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort # 직접 경로 지정 예시
    print("Imported DeepOCSort tracker.")
except ImportError:
    print("Error: Could not import DeepOCSort tracker. Tracking will fail.")
    DeepOCSort = None # 임시 처리

# --- 기타 Helper 함수 (YOLO 로더, detector, 정렬, 색상 등 - 기존 유지) ---
def initialize_yolo_model(model_path, device):
    if YOLO is None: return None
    try:
        model = YOLO(model_path); model.to(device)
        print(f"YOLO model loaded from {model_path} to {device}")
        return model
    except Exception as e: print(f"Error loading YOLO model: {e}"); return None

def yolo_detector(frame: np.ndarray, model: YOLO, confidence_threshold: float = 0.5):
    detections = []
    if model is None: return np.empty((0, 6))
    try:
        results = model(frame, verbose=False) # verbose=False 로 로그 줄이기
        if results and results[0].boxes:
            for box in results[0].boxes:
                conf = box.conf[0].item()
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    detections.append([x1, y1, x2, y2, conf, cls_id])
    except Exception as e: print(f"Error during YOLO detection: {e}")
    return np.array(detections) if detections else np.empty((0, 6))

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', str(s))]

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return tuple(max(c, 60) for c in color) # 너무 어둡지 않게

# --- Tracking 함수 수정 ---
def run_tracking(
    tracker,                # 초기화된 Tracker 객체
    source_path: Path,      # 입력 비디오/이미지 시퀀스 경로 (Path 객체)
    output_dir: Path,       # 결과 저장 디렉토리 (Path 객체)
    detector_func,          # Detector 함수
    yolo_model,             # YOLO 모델 객체
    detection_conf_thresh: float, # 탐지 임계값
    cfg_lstm: dict,         # LSTM 설정 딕셔너리
    patch_with_lstm: bool,  # LSTM 패치 적용 여부
    # --- 추가 인자 ---
    loaded_lstm_model = None, # 로드된 LSTM 모델 (패치 시 필요)
    loaded_normalizer = None, # 로드된 Normalizer (패치 시 필요)
    # --- 옵션 ---
    is_image_sequence: bool = False, # 소스 타입
    save_output_video: bool = True,
    save_output_images: bool = False
    ):
    """
    비디오 파일 또는 이미지 시퀀스에 대해 추적을 실행합니다.
    [개선됨] patcher 모듈 변수 설정 및 KF 패치 로직 포함.
    """
    print(f"\n--- Running Tracking on {source_path.name} ---")
    print(f"LSTM Patching Enabled: {patch_with_lstm}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_name = source_path.stem
    results_filename = sequence_name + ".txt"
    results_file_path = output_dir / results_filename

    output_video_path = output_dir / f"{sequence_name}_tracked.avi" if save_output_video else None
    output_images_dir = output_dir / f"{sequence_name}_tracked_images" if save_output_images else None
    if output_images_dir: output_images_dir.mkdir(parents=True, exist_ok=True)

    frame_iterator = None
    num_frames, frame_width, frame_height, fps = 0, 0, 0, 30
    video_writer = None

    # --- 소스 로딩 (비디오 또는 이미지 시퀀스) ---
    if is_image_sequence:
        image_files = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')), key=natural_sort_key)
        if not image_files: print(f"Error: No images found in {source_path}"); return None
        num_frames = len(image_files)
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None: print(f"Error reading first image: {image_files[0]}"); return None
        frame_height, frame_width, _ = first_frame.shape
        frame_iterator = iter(image_files)
        fps = cfg_lstm.get('fps', 30) # 설정에서 fps 가져오기
        print(f"Image sequence: {num_frames} frames, {frame_width}x{frame_height}, FPS set to {fps}")
    else: # Video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened(): print(f"Error opening video: {source_path}"); return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS); fps = video_fps if video_fps > 0 else fps
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_iterator = cap
        print(f"Video: FPS {fps:.2f}, Res {frame_width}x{frame_height}, Frames {num_frames}")

    # --- 비디오 라이터 초기화 ---
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
        if video_writer.isOpened(): print(f"Saving tracked video to: {output_video_path}")
        else: print(f"Error opening video writer for {output_video_path}"); video_writer = None
    seen_track_ids = set() # 현재까지 추적된 모든 ID를 저장할 집합
    print(f"Saving MOT results (text) to: {results_file_path}")
    if save_output_images: print(f"Saving images for newly detected IDs to: {output_images_dir}")

    # --- LSTM 패치 준비 (patcher 모듈 변수 설정) ---
    if patch_with_lstm and loaded_lstm_model and loaded_normalizer:
        lstm_patcher.lstm_model = loaded_lstm_model
        lstm_patcher.loaded_normalizer_obj = loaded_normalizer
        lstm_patcher.device = torch.device(cfg_lstm.get('device', 'cpu'))
        lstm_patcher.window_size = cfg_lstm.get('window_size', 10) # cfg 값 사용
        lstm_patcher.correction_weight = cfg_lstm.get('lstm_correction_weight', 0.5)
        lstm_patcher.fps = fps # 실제 fps 사용
        # feat_dim은 normalizer에서 가져오거나 cfg에서 가져옴
        if hasattr(loaded_normalizer, 'feature_cols'):
             lstm_patcher.feat_dim = len(loaded_normalizer.feature_cols)
        else: # Normalizer에 정보 없을 시 cfg 값 사용 (모델과 일치해야 함)
             lstm_patcher.feat_dim = cfg_lstm.get('feat_dim')
             if lstm_patcher.feat_dim is None:
                  print("Error: Cannot determine feat_dim for patcher. Disabling LSTM.")
                  patch_with_lstm = False # feat_dim 모르면 패치 불가

        # Normalizer에 이미지 크기 정보 설정 (역변환 등에 필요 시)
        loaded_normalizer.img_width = frame_width
        loaded_normalizer.img_height = frame_height
        print(f"Patcher configured: feat_dim={lstm_patcher.feat_dim}, window={lstm_patcher.window_size}, weight={lstm_patcher.correction_weight}")
    elif patch_with_lstm:
        print("Warning: LSTM model or normalizer not loaded. Disabling LSTM patching.")
        patch_with_lstm = False # 필요한 객체 없으면 비활성화

    # --- 메인 추적 루프 ---
    frame_id = 0
    kf_predict_patched_classes = set() # 이미 패치된 KF 클래스 추적

    with open(results_file_path, 'w') as f_results:
        pbar = tqdm(total=num_frames, desc=f"Tracking {sequence_name}", unit="frame")
        while True:
            frame = None
            if is_image_sequence:
                try: img_path = next(frame_iterator); frame = cv2.imread(str(img_path))
                except StopIteration: break
                if frame is None: warnings.warn(f"Skipping unreadable image {img_path}"); pbar.update(1); continue
            else:
                ret, frame = frame_iterator.read()
                if not ret: break

            frame_id += 1
            vis_frame = frame.copy()
            new_id_detected_this_frame = False # 매 프레임 시작 시 False로 리셋
            newly_detected_ids_in_this_frame = [] # 현재 프레임의 새 ID 목록 (파일명용)

            # --- 1. Detection ---
            detections = detector_func(frame, yolo_model, detection_conf_thresh)

            # --- 2. (LSTM 패치 시) 사전 작업: 특징 버퍼 업데이트 및 KF 패치 ---
            if patch_with_lstm:
                active_tracks = getattr(tracker, 'tracked_stracks', []) + getattr(tracker, 'lost_stracks', [])
                for track in active_tracks: # track은 tracker의 track 객체 (예: STrack)
                    kf = lstm_patcher.get_kf_from_track(track) # KF 객체 가져오기
                    if kf is None: continue

                    # --- 2.1 특징 버퍼 초기화 및 업데이트 ---
                    if not hasattr(track, 'features_buffer'):
                        track.features_buffer = deque(maxlen=lstm_patcher.window_size)
                        # KF 객체에 부모 track 참조 추가 (patcher에서 사용)
                        kf.parent_track = track

                    # 특징 추출 (현재 KF 상태 기반) - patcher.py와 유사 로직
                    try:
                         current_state = kf.x.copy() # [cx, cy, s, r, vx, vy, vs]
                         current_features = np.zeros(lstm_patcher.feat_dim)
                         # 필요한 특징 추출 (Normalizer의 feature_cols 순서대로)
                         cx, cy, s, r, vx, vy = current_state[:6]
                         s_eps, r_eps = max(s, 1e-6), max(r, 1e-6)
                         w, h = np.sqrt(s_eps * r_eps), np.sqrt(s_eps / r_eps)

                         feature_map = { # Normalizer의 특징 이름과 값 매핑
                             'x_center': cx / frame_width if frame_width > 0 else 0,
                             'y_center': cy / frame_height if frame_height > 0 else 0,
                             'w_norm': w / frame_width if frame_width > 0 else 0,
                             'h_norm': h / frame_height if frame_height > 0 else 0,
                             'vx': vx / frame_width * fps if frame_width > 0 else 0, # 정규화 및 시간 단위 변환
                             'vy': vy / frame_height * fps if frame_height > 0 else 0,
                             # 가속도는 이전 특징 필요
                         }
                         # ax, ay 계산 (버퍼에 이전 특징 있을 때)
                         ax, ay = 0.0, 0.0
                         if len(track.features_buffer) > 0:
                             prev_features_map = track.features_buffer[-1] # 이전 특징 맵
                             prev_vx_norm = prev_features_map.get('vx', 0.0)
                             prev_vy_norm = prev_features_map.get('vy', 0.0)
                             dt = 1.0 / fps
                             ax = (feature_map['vx'] - prev_vx_norm) / dt if dt > 0 else 0
                             ay = (feature_map['vy'] - prev_vy_norm) / dt if dt > 0 else 0
                         feature_map['ax'] = ax
                         feature_map['ay'] = ay

                         # Context 특징 계산 (필요 시)
                         if 'aspect_ratio' in loaded_normalizer.feature_cols:
                              feature_map['aspect_ratio'] = w / (h + 1e-6)
                         if 'area_ratio' in loaded_normalizer.feature_cols:
                              feature_map['area_ratio'] = (w * h) / (frame_width * frame_height + 1e-6)

                         # Normalizer의 feature_cols 순서대로 current_features 배열 채우기
                         for i, col_name in enumerate(loaded_normalizer.feature_cols):
                              current_features[i] = feature_map.get(col_name, 0.0) # 없으면 0

                         track.features_buffer.append(feature_map) # 특징 맵 저장 (ax,ay 계산용)
                         # track.features_buffer.append(current_features) # 또는 numpy 배열 저장

                    except Exception as e:
                         warnings.warn(f"Feature extraction error for track {getattr(track,'track_id','?')}: {e}")
                         continue # 특징 추출 실패 시 다음 트랙으로

                    # --- 2.2 KF Predict 메소드 패치 (클래스당 최초 한 번) ---
                    kf = lstm_patcher.get_kf_from_track(track)
                    if kf:
                        kf_class = kf.__class__
                        if kf_class not in kf_predict_patched_classes:
                            if hasattr(kf_class, 'predict') and callable(getattr(kf_class, 'predict')):
                                if kf_class not in lstm_patcher._original_kf_predict:
                                     lstm_patcher._original_kf_predict[kf_class] = getattr(kf_class, 'predict')
                                     # print(f"Stored original predict method for {kf_class.__name__}") # 필요 시 주석 해제

                                # predict 메소드를 patcher의 함수로 교체
                                setattr(kf_class, 'predict', lstm_patcher.predict_with_lstm)
                                kf_predict_patched_classes.add(kf_class)
                                # --- >>> 패치 실행 확인 로그 추가 <<< ---
                                print(f"DEBUG: Patched predict method for {kf_class.__name__} in Frame {frame_id}")
                                # --- >>> 로그 추가 끝 <<< ---
                            else:
                                warnings.warn(f"Cannot patch predict method for {kf_class.__name__}")


            # --- 3. Tracking Update ---
            # tracker.update 호출 시 내부적으로 kf.predict()가 호출됨
            # 이때 패치된 predict_with_lstm 함수가 실행됨
            try:
                # tracker 종류에 따라 update 인자 다를 수 있음 (img_info, img_size 등)
                # DeepOCSort는 detections, img 를 받는 것으로 가정
                online_targets = tracker.update(detections, frame)
            except Exception as e:
                 print(f"Error during tracker update: {e}")
                 online_targets = [] # 오류 시 빈 결과

            # --- 4. 결과 저장 및 시각화 ---
            if online_targets is not None and len(online_targets) > 0:
                for t in online_targets:
                    # tracker 출력 형식 확인 (예: [x1, y1, x2, y2, track_id, cls_id, conf])
                    if len(t) >= 7:
                        x1, y1, x2, y2 = map(int, t[0:4])
                        tid = int(t[4])
                        conf = t[6]; cls_id = int(t[5]) # Class ID 필요시 사용
                        # MOT Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
                        f_results.write(f'{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.2f},-1,-1,-1\n')
                        # --- >>> 새로운 ID 확인 로직 <<< ---
                        if tid not in seen_track_ids:
                            new_id_detected_this_frame = True # 플래그 설정
                            newly_detected_ids_in_this_frame.append(tid) # 목록에 추가
                            seen_track_ids.add(tid) # 전체 집합에 추가
                            # print(f"New track ID {tid} detected at frame {frame_id}") # 로그 출력 옵션

                        # 시각화
                        color = get_color(tid)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        label = f'ID:{tid}' # ({conf:.2f})' # 신뢰도 표시 옵션
                        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else: warnings.warn(f"Skipping track output with unexpected format: {t}")

            # --- 5. 시각화 결과 저장 (비디오/이미지) ---
            if video_writer: video_writer.write(vis_frame)
            if output_images_dir and new_id_detected_this_frame:
                 # print(f"Debug Image Save: Frame {frame_id}, New IDs: {newly_detected_ids_in_this_frame}") # 디버깅 프린트

                 # 파일명 생성 (첫 번째 새 ID 사용 또는 다른 방식)
                 first_new_id = newly_detected_ids_in_this_frame[0]
                 img_save_path = output_images_dir / f"{frame_id:06d}_newID_{first_new_id}.jpg"
                 cv2.imwrite(str(img_save_path), vis_frame)

            pbar.update(1)
        # --- 루프 종료 ---
        pbar.close()
    # --- 파일 쓰기 종료 ---

    # --- 리소스 해제 ---
    if not is_image_sequence and frame_iterator: frame_iterator.release()
    if video_writer: video_writer.release(); print(f"Finished writing video.")
    if output_images_dir: print(f"Finished saving images.")

    # --- KF Predict 메소드 복원 (프로세스 종료 전) ---
    if patch_with_lstm and lstm_patcher._original_kf_predict:
        print("Restoring original Kalman Filter predict methods...")
        for kf_class, orig_method in lstm_patcher._original_kf_predict.items():
            if hasattr(kf_class, 'predict'):
                setattr(kf_class, 'predict', orig_method)
        lstm_patcher._original_kf_predict = {} # 초기화
        print("Restoration complete.")

    print(f"Tracking finished. MOT results saved to: {results_file_path}")
    return results_file_path

# --- main 함수 (argparse 사용) ---
def main():
    parser = argparse.ArgumentParser(description='Run LSTM-enhanced object tracking.')
    parser.add_argument('--source', required=True, help='Path to video file or image sequence directory.')
    parser.add_argument('--yolo_model', default='yolov8n.pt', help='Path to YOLOv8 model weights.')
    parser.add_argument('--reid_weights', type=str, default=None,
                        help='Path to custom ReID model weights. Overrides tracker_cfg if provided.')
    parser.add_argument('--tracker_cfg', default='../boxmot/configs/deepocsort.yaml', help='Path to tracker config file (e.g., deepocsort.yaml).')
    parser.add_argument('--lstm_cfg', default='../boxmot/configs/lstm.yaml', help='Path to LSTM config file (lstm.yaml).')
    parser.add_argument('--conf', type=float, default=0.5, help='Object detection confidence threshold.')
    parser.add_argument('--output_dir', default='mot_results', help='Base directory for tracking results.')
    parser.add_argument('--tag', default=None, help='Optional tag for output subdirectory.')
    # --- 제어 플래그 ---
    parser.add_argument('--no_lstm', action='store_true', help='Disable LSTM patching (run baseline tracker).')
    parser.add_argument('--save_video', action='store_true', help='Save output video with tracking results.')
    parser.add_argument('--save_images', action='store_true', help='Save output frames as images.')
    # --- 평가 관련 ---
    parser.add_argument('--eval', action='store_true', help='Run MOT evaluation after tracking.')
    parser.add_argument('--gt_dir', default=None, help='Path to the base directory of Ground Truth data (required for evaluation).')
    parser.add_argument('--benchmark', default='MOTChallenge', help='Benchmark name for evaluation (e.g., MOT17, MOT20).')
    parser.add_argument('--split', default='test', help='Data split for evaluation (e.g., train, test).')

    args = parser.parse_args()

    # --- 1. LSTM 설정 로드 ---
    lstm_config_path = Path(args.lstm_cfg)
    if not lstm_config_path.exists(): print(f"Error: LSTM Config not found: {lstm_config_path}"); return
    with open(lstm_config_path, 'r') as f: cfg_lstm = yaml.safe_load(f)
    device = torch.device(cfg_lstm.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    patch_with_lstm = not args.no_lstm

    # --- 2. LSTM 모델 및 Normalizer 로드 (patch_with_lstm=True 시) ---
    loaded_lstm_model = None
    loaded_normalizer = None
    if patch_with_lstm:
        lstm_weights_path = Path(cfg_lstm.get('lstm_weights', '')) # 경로 있는지 확인
        normalizer_path = Path(cfg_lstm.get('normalizer_path', ''))

        if not lstm_weights_path.exists() or not normalizer_path.exists():
             print(f"Warning: LSTM weights ({lstm_weights_path.name}) or normalizer ({normalizer_path.name}) not found. Disabling LSTM patching.")
             patch_with_lstm = False
        else:
             try:
                 # 모델 로드 (체크포인트에 config 포함 가정)
                 checkpoint = torch.load(lstm_weights_path, map_location=device)
                 if 'config' not in checkpoint: raise ValueError("Checkpoint missing 'config'.")
                 model_cfg_loaded = checkpoint['config']
                 # --- 중요: 로드된 설정의 feat_dim을 현재 cfg_lstm에 반영 ---
                 cfg_lstm['feat_dim'] = model_cfg_loaded.get('feat_dim')
                 if cfg_lstm['feat_dim'] is None: raise ValueError("'feat_dim' missing in loaded config.")

                 loaded_lstm_model = TrackLSTM(cfg=model_cfg_loaded) # 저장된 설정으로 모델 생성
                 loaded_lstm_model.load_state_dict(checkpoint['model_state_dict'])
                 loaded_lstm_model.to(device).eval()
                 print(f"LSTM model loaded from {lstm_weights_path} (feat_dim={cfg_lstm['feat_dim']})")

                 # Normalizer 로드
                 with open(normalizer_path, 'rb') as f: loaded_normalizer = pickle.load(f)
                 if not isinstance(loaded_normalizer, FeatureNormalizer) or not loaded_normalizer.is_fitted:
                      raise TypeError("Loaded object is not a fitted FeatureNormalizer.")
                 print(f"Normalizer loaded from {normalizer_path}")
                 # Normalizer와 모델 간 feature_cols 일치 확인 (선택적)
                 if hasattr(loaded_normalizer, 'feature_cols') and loaded_normalizer.feature_cols != model_cfg_loaded.get('feature_cols', []):
                      warnings.warn("Feature mismatch between loaded normalizer and model config!")

             except Exception as e:
                  print(f"ERROR loading LSTM components: {e}. Disabling LSTM patching.")
                  patch_with_lstm = False

    # --- 3. YOLO 모델 초기화 ---
    yolo_model = initialize_yolo_model(args.yolo_model, device)
    if yolo_model is None: print("Exiting due to YOLO model failure."); return

    # --- 4. Tracker 초기화 ---
    if DeepOCSort is None: print("Exiting because Tracker class is not available."); return
    tracker_config_path = Path(args.tracker_cfg)
    tracker_cfg_dict = {}
    if tracker_config_path.exists():
         with open(tracker_config_path, 'r') as f: tracker_cfg_dict = yaml.safe_load(f)
         print(f"Tracker config loaded from: {tracker_config_path}")
    else:
         print(f"Warning: Tracker config not found: {tracker_config_path}. Using defaults.")

    reid_model_path = args.reid_weights # 명령어 인자 값
    if reid_model_path is None: # 명령어 인자가 없으면
        # tracker_cfg_dict 에서 reid_weights 또는 유사한 키 찾기 (키 이름 확인 필요)
        reid_model_path = tracker_cfg_dict.get('reid_weights', 'osnet_x0_25_msmt17.pt') # 예시 키 이름 및 기본값
        print(f"Using ReID weights from tracker_cfg or default: {reid_model_path}")
    else:
        print(f"Using ReID weights specified by command line: {reid_model_path}")

    try:
        # DeepOCSort 초기화 시 결정된 ReID 경로 사용
        tracker = DeepOCSort(
            model_weights=Path(reid_model_path), # <--- 수정된 경로 사용
            device=device,
            fp16=tracker_cfg_dict.get('fp16', False),
            # **tracker_cfg_dict # 또는 필요한 인자 명시적으로 전달
             conf=tracker_cfg_dict.get('det_thresh', 0.3),
             iou_threshold=tracker_cfg_dict.get('iou_threshold', 0.3),
             max_age=tracker_cfg_dict.get('max_age', 30),
             min_hits=tracker_cfg_dict.get('min_hits', 3),
             # ... 기타 DeepOCSort 인자 ...
        )
        print(f"Initialized {tracker.__class__.__name__} tracker with ReID: {reid_model_path}")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return

    # --- 5. 결과 디렉토리 설정 ---
    source_path = Path(args.source)
    if not source_path.exists(): print(f"Error: Source path not found: {source_path}"); return
    sequence_name = source_path.stem
    output_base_dir = Path(args.output_dir)
    run_sub_dir_name = f"{sequence_name}_{'lstm' if patch_with_lstm else 'baseline'}"
    if args.tag: run_sub_dir_name += f"_{args.tag}"
    run_sub_dir_name += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = output_base_dir / run_sub_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tracking results will be saved in: {results_dir}")

    # --- 6. Run Tracking ---
    results_file_path = run_tracking(
        tracker=tracker,
        source_path=source_path,
        output_dir=results_dir,
        detector_func=yolo_detector,
        yolo_model=yolo_model,
        detection_conf_thresh=args.conf,
        cfg_lstm=cfg_lstm,
        patch_with_lstm=patch_with_lstm,
        loaded_lstm_model=loaded_lstm_model,
        loaded_normalizer=loaded_normalizer,
        is_image_sequence=source_path.is_dir(),
        save_output_video=args.save_video,
        save_output_images=args.save_images
    )

    if results_file_path is None:
        print("Tracking run failed.")
        return

    # --- 7. Run Evaluation (선택적) ---
    if args.eval:
        print("\n--- Running MOT Evaluation ---")
        if args.gt_dir is None:
            print("Error: Ground Truth directory (--gt_dir) is required for evaluation.")
        else:
            gt_base_path = Path(args.gt_dir)
            # 평가할 시퀀스 이름 (결과 폴더 이름과 일치시킨다고 가정)
            # seqmap 파일 생성 또는 자동 탐지 필요
            # 여기서는 단일 시퀀스 평가 가정
            gt_dir_for_eval = gt_base_path / args.split / sequence_name / 'gt' / 'gt.txt'
            seqinfo_path = gt_base_path / args.split / sequence_name / 'seqinfo.ini'

            if not gt_dir_for_eval.exists() or not seqinfo_path.exists():
                 print(f"ERROR: GT data or seqinfo not found for sequence '{sequence_name}' in '{gt_base_path / args.split}'. Skipping eval.")
            else:
                # TrackEval은 GT 루트와 결과 루트를 받음
                # 결과는 results_dir에 sequence_name.txt 로 저장되어 있음
                # GT 루트는 gt_base_path
                # 결과 루트는 results_dir의 부모 (tracker 이름 레벨이 필요할 수 있음)
                res_root_for_eval = results_dir.parent # tracker 이름 폴더 역할
                tracker_name = results_dir.name # 결과 폴더 이름이 tracker 이름

                metrics_output_csv = results_dir / f"{sequence_name}_metrics_summary.csv"
                try:
                    # seqmap 파일 생성 (단일 시퀀스용)
                    seqmap_content = f"name\n{sequence_name}\n"
                    seqmap_file = results_dir / "eval_seqmap.txt"
                    with open(seqmap_file, 'w') as f: f.write(seqmap_content)

                    metrics_df = compute_mot_metrics(
                        gt_root=str(gt_base_path),        # GT 데이터 루트
                        res_root=str(res_root_for_eval),  # 결과 루트 (tracker 이름 폴더 상위)
                        output_csv=str(metrics_output_csv),
                        benchmark_name=args.benchmark,
                        split_to_eval=args.split,
                        # metrics_to_compute=['CLEAR', 'Identity', 'HOTA'], # 원하는 지표
                        seqmap_filename=str(seqmap_file) # 생성한 seqmap 파일 경로 전달
                    )
                    if metrics_df is not None and not metrics_df.empty:
                         print("\n--- Evaluation Summary ---")
                         print(metrics_df)
                except Exception as e:
                     print(f"ERROR during MOT evaluation call: {e}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()