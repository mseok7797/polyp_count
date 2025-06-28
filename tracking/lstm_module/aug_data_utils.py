from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
import warnings
from collections import deque # deque import 추가 (LSTMDataset에서 사용 가능성 대비)

# --- 데이터 로딩 함수 (기존 유지) ---
def load_mot_data(mot_root: str, img_width: int = 640, img_height: int = 480) -> pd.DataFrame:
    """
    MOT gt.txt 파일을 로드하고, 기본적인 bbox 좌표 포함하여 반환.
    """
    cols = ['frame','track_id','x','y','w','h','conf','cls','vis']
    dfs = []
    mot_root_path = Path(mot_root) # Path 객체로 변환
    print(f"Loading MOT data from: {mot_root_path.resolve()}")
    # mot_root 경로가 존재하는지 확인
    if not mot_root_path.exists() or not mot_root_path.is_dir():
        print(f"Error: MOT root directory not found or is not a directory: {mot_root_path}")
        return pd.DataFrame(columns=['frame','seq','track_id','x','y','w','h'])

    processed_seq_count = 0
    skipped_count = 0
    for seq_dir in mot_root_path.iterdir():
        # --- >>> 수정된 부분 시작 <<< ---
        # 1. 디렉토리인지 확인
        # 2. 이름이 'seqmaps' 인지 확인 (소문자로 비교하여 대소문자 무시)
        if not seq_dir.is_dir() or seq_dir.name.lower() == 'seqmaps':
            skipped_count += 1
            continue
        # --- >>> 수정된 부분 끝 <<< ---
        gt_path = seq_dir / 'gt' / 'gt.txt'
        if not gt_path.exists():
            print(f"Warning: gt.txt not found in sequence directory: {seq_dir}")
            continue
        try:
            df = pd.read_csv(gt_path, header=None, names=cols)
            df = df[['frame','track_id','x','y','w','h']].copy()
            df['seq'] = seq_dir.name # 시퀀스 이름 저장
            dfs.append(df)
            processed_seq_count += 1
        except Exception as e:
            print(f"Error processing {gt_path}: {e}")

    if skipped_count > 0: print(f"Skipped {skipped_count} non-sequence directory items.")        

    if not dfs:
        print("Warning: No valid MOT data found after processing subdirectories.")
        return pd.DataFrame(columns=['frame','seq','track_id','x','y','w','h'])

    print(f"Loaded data from {processed_seq_count} sequences.")
    return pd.concat(dfs, ignore_index=True)

# --- 시퀀스 생성 함수 (기존 유지) ---
def create_sequences(df: pd.DataFrame, time_steps: int = 10, feature_cols: list = None) -> np.ndarray:
    """
    DataFrame에서 sliding window 방식으로 시퀀스 데이터를 생성합니다.
    """
    if df.empty or len(df) < time_steps + 1 or feature_cols is None or not feature_cols:
        # feature_cols가 None이거나 비어있는 경우 처리 추가
        return np.empty((0, time_steps, len(feature_cols or [])))

    sequences = []
    # feature_cols에 해당하는 컬럼이 df에 모두 있는지 확인
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Features missing in DataFrame for sequence creation: {missing_cols}. Returning empty array.")
        return np.empty((0, time_steps, len(feature_cols)))

    values = df[feature_cols].values
    for i in range(len(values) - time_steps):
        sequences.append(values[i : i + time_steps])

    if not sequences:
        return np.empty((0, time_steps, len(feature_cols)))

    return np.stack(sequences)

# --- 시퀀스 품질 검증 함수 (개선 파이프라인용) ---
def validate_sequence_physics(seq: np.ndarray, feature_cols: list, max_speed_norm: float = 1.0, max_accel_norm: float = 0.5) -> bool:
    """
    시퀀스 데이터의 물리적 타당성을 검증합니다 (정규화된 속도/가속도 기준).
    """
    try:
        vx_idx = feature_cols.index('vx')
        vy_idx = feature_cols.index('vy')
        ax_idx = feature_cols.index('ax')
        ay_idx = feature_cols.index('ay')
    except ValueError:
        warnings.warn("Cannot validate physics, required dynamic features not found.")
        return True # 특징 없으면 통과

    # 속도 검증
    with np.errstate(invalid='ignore'): # Ignore potential NaNs during calculation
        speed_norm = np.sqrt(seq[:, vx_idx]**2 + seq[:, vy_idx]**2)
        if np.any(speed_norm > max_speed_norm): return False

        # 가속도 검증
        accel_norm = np.sqrt(seq[:, ax_idx]**2 + seq[:, ay_idx]**2)
        if np.any(accel_norm > max_accel_norm): return False

    return True

# =====================================================
#  기존 방식 (Original Pipeline Components)
# =====================================================

def handle_frame_gaps_original(track: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    [기존 방식] 선형 보간으로 프레임 간격을 처리하고, max_gap 초과 시 분리합니다.
    """
    if track.empty or len(track) < 2: return pd.DataFrame()

    track = track.sort_values('frame').reset_index(drop=True)
    frames = track['frame'].values
    coords = track[['x', 'y', 'w', 'h']].values
    diffs = np.diff(frames)

    split_indices = np.where(diffs > max_gap)[0] + 1
    segments_frames = np.split(frames, split_indices)
    segments_coords = np.split(coords, split_indices)

    interpolated_segments = []
    for seg_frames, seg_coords in zip(segments_frames, segments_coords):
        if len(seg_frames) < 2: continue

        try:
            full_frames = np.arange(int(seg_frames.min()), int(seg_frames.max()) + 1)
            interp_func = interp1d(seg_frames, seg_coords, axis=0, kind='linear', fill_value="extrapolate")
            interpolated_values = interp_func(full_frames)

            df_interpolated = pd.DataFrame({
                'frame': full_frames,
                'x': interpolated_values[:, 0], 'y': interpolated_values[:, 1],
                'w': interpolated_values[:, 2], 'h': interpolated_values[:, 3]
            })
            # 경계 클리핑 추가 (원본 bbox 기준, img_size 필요)
            # img_width, img_height = cfg.get('img_width'), cfg.get('img_height') # cfg 접근 필요
            # ... clip logic ...
            interpolated_segments.append(df_interpolated)
        except ValueError as e:
            print(f"Original interpolation error for frames {seg_frames.min()}-{seg_frames.max()}: {e}")
            continue

    if not interpolated_segments: return pd.DataFrame()

    result_df = pd.concat(interpolated_segments).reset_index(drop=True)
    # track_id, seq 정보 다시 부여
    if 'track_id' in track.columns: result_df['track_id'] = track['track_id'].iloc[0]
    if 'seq' in track.columns: result_df['seq'] = track['seq'].iloc[0]
    return result_df


def calculate_dynamics_original(df: pd.DataFrame, fps: int = 30, img_width: int = 640, img_height: int = 480) -> pd.DataFrame:
    """
    [기존 방식] 기본적인 동적 특징(vx, vy, ax, ay)을 계산합니다.
    """
    if df.empty or 'frame' not in df.columns or len(df) < 3:
        return pd.DataFrame()

    df = df.sort_values('frame').reset_index(drop=True)
    dt = 1.0 / fps

    # 기본 정규화
    df['x_center'] = (df['x'] + df['w'] / 2) / img_width
    df['y_center'] = (df['y'] + df['h'] / 2) / img_height
    df['w_norm']   = df['w'] / img_width
    df['h_norm']   = df['h'] / img_height

    # 동적 특징
    df['vx'] = df['x_center'].diff() / dt
    df['vy'] = df['y_center'].diff() / dt
    df['ax'] = df['vx'].diff() / dt
    df['ay'] = df['vy'].diff() / dt

    return df.dropna().reset_index(drop=True)


class TrajectoryAugmenterOriginal:
    """
    [기존 방식] 기본적인 궤적 시퀀스 데이터 증강 (Jitter, Scale, Time Warp).
    """
    def __init__(self, noise_scale: float = 0.01, scale_range: tuple = (0.9, 1.1), time_warp_ratio: float = 0.1):
        self.noise_scale = noise_scale
        self.scale_range = scale_range
        self.time_warp_ratio = time_warp_ratio

    def jitter(self, seq: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=seq.shape)
        return seq + noise

    def scale(self, seq: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        return seq * factor

    def time_warp(self, seq: np.ndarray) -> np.ndarray:
        # (이전 답변의 time_warp 로직 구현)
        time_steps, num_features = seq.shape
        original_time = np.linspace(0, time_steps - 1, num=time_steps)
        cumulative_offset = np.cumsum(np.random.uniform(-self.time_warp_ratio, self.time_warp_ratio, size=time_steps))
        speed_factor = np.random.uniform(1 - self.time_warp_ratio, 1 + self.time_warp_ratio)
        warped_time = (original_time + cumulative_offset) * speed_factor
        warped_time = np.clip(np.sort(warped_time), 0, time_steps - 1)

        warped_seq = np.zeros_like(seq)
        try:
            for i in range(num_features):
                interp_func = interp1d(original_time, seq[:, i], kind='linear', fill_value="extrapolate")
                warped_seq[:, i] = interp_func(warped_time)
        except ValueError: return seq # 실패 시 원본 반환
        return warped_seq

    def augment(self, seq: np.ndarray, p_jitter=0.7, p_scale=0.7, p_warp=0.5) -> np.ndarray:
        augmented_seq = seq.copy()
        if np.random.rand() < p_jitter: augmented_seq = self.jitter(augmented_seq)
        if np.random.rand() < p_scale: augmented_seq = self.scale(augmented_seq)
        if np.random.rand() < p_warp: augmented_seq = self.time_warp(augmented_seq)
        return augmented_seq


# =====================================================
#  개선된 방식 (Enhanced Pipeline Components)
# =====================================================

def handle_frame_gaps_hierarchical(
    track: pd.DataFrame,
    max_gap_linear: int = 45,
    max_gap_spline: int = 5,
    img_width: int = 640,
    img_height: int = 480,
    max_speed_ratio: float = 1.5
    ) -> pd.DataFrame:
    """
    [개선됨] 계층적 보간 및 물리 제약 조건(최대 속도)을 적용합니다.
    """
    if track.empty or len(track) < 2: return pd.DataFrame()

    track = track.sort_values('frame').reset_index(drop=True)
    frames = track['frame'].values
    coords = track[['x', 'y', 'w', 'h']].values

    split_indices = np.where(np.diff(frames) > max_gap_linear)[0] + 1
    segments_frames = np.split(frames, split_indices)
    segments_coords = np.split(coords, split_indices)

    processed_segments = []

    for seg_frames, seg_coords in zip(segments_frames, segments_coords):
        if len(seg_frames) < 2: continue

        # --- 선형 보간 ---
        full_frames = np.arange(int(seg_frames.min()), int(seg_frames.max()) + 1)
        interp_func_linear = interp1d(seg_frames, seg_coords, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_linear = interp_func_linear(full_frames)
        df_interp = pd.DataFrame({
            'frame': full_frames,
            'x': interpolated_linear[:, 0], 'y': interpolated_linear[:, 1],
            'w': interpolated_linear[:, 2], 'h': interpolated_linear[:, 3]
        })

        # --- 스플라인 보간 (짧은 간격에만) ---
        gaps = np.diff(seg_frames)
        if len(seg_frames) >= 4: # Cubic 스플라인 최소 조건
             try:
                 interp_func_spline = interp1d(seg_frames, seg_coords, axis=0, kind='cubic', fill_value="extrapolate")
                 interpolated_spline = interp_func_spline(full_frames)
                 original_indices = np.searchsorted(full_frames, seg_frames)
                 for i in range(len(seg_frames) - 1):
                     if gaps[i] <= max_gap_spline:
                         start_idx, end_idx = original_indices[i], original_indices[i+1]
                         df_interp.iloc[start_idx:end_idx+1, 1:5] = interpolated_spline[start_idx:end_idx+1]
             except ValueError: pass # 스플라인 실패 시 선형 결과 유지

        # --- 물리 제약 (최대 속도) ---
        if len(seg_frames) >= 2:
             dx = np.diff(seg_coords[:, 0]); dy = np.diff(seg_coords[:, 1])
             dt_frames = np.diff(seg_frames)
             with np.errstate(divide='ignore', invalid='ignore'): # 0 프레임 간격 무시
                 speeds = np.sqrt(dx**2 + dy**2) / dt_frames
                 speeds = speeds[np.isfinite(speeds)] # NaN/inf 제거
             if len(speeds) > 0:
                max_speed_allowed = np.percentile(speeds, 95) * max_speed_ratio
                interp_dx = np.diff(df_interp['x'].values)
                interp_dy = np.diff(df_interp['y'].values)
                interp_speeds = np.sqrt(interp_dx**2 + interp_dy**2)
                exceed_indices = np.where(interp_speeds > max_speed_allowed)[0]
                for idx in exceed_indices:
                    scale = max_speed_allowed / interp_speeds[idx]
                    df_interp.loc[idx + 1, 'x'] = df_interp.loc[idx, 'x'] + interp_dx[idx] * scale
                    df_interp.loc[idx + 1, 'y'] = df_interp.loc[idx, 'y'] + interp_dy[idx] * scale

        # --- 경계 클리핑 ---
        df_interp['x'] = np.clip(df_interp['x'], 0, img_width - df_interp['w'])
        df_interp['y'] = np.clip(df_interp['y'], 0, img_height - df_interp['h'])
        df_interp['w'] = np.clip(df_interp['w'], 1, img_width)
        df_interp['h'] = np.clip(df_interp['h'], 1, img_height)

        processed_segments.append(df_interp)

    if not processed_segments: return pd.DataFrame()

    result_df = pd.concat(processed_segments).reset_index(drop=True)
    if 'track_id' in track.columns: result_df['track_id'] = track['track_id'].iloc[0]
    if 'seq' in track.columns: result_df['seq'] = track['seq'].iloc[0]
    return result_df


def calculate_dynamics_with_context(df: pd.DataFrame, fps: int = 30, img_width: int = 640, img_height: int = 480, use_context: bool = True) -> pd.DataFrame:
    """
    [개선됨] 기본 동적 특징 + 컨텍스트 특징(aspect_ratio, area_ratio) 계산.
    """
    if df.empty or 'frame' not in df.columns or len(df) < 3:
        return pd.DataFrame()

    df = df.sort_values('frame').reset_index(drop=True)
    dt = 1.0 / fps

    # 기본 정규화
    df['x_center'] = (df['x'] + df['w'] / 2) / img_width
    df['y_center'] = (df['y'] + df['h'] / 2) / img_height
    df['w_norm']   = df['w'] / img_width
    df['h_norm']   = df['h'] / img_height

    # 동적 특징
    df['vx'] = df['x_center'].diff() / dt
    df['vy'] = df['y_center'].diff() / dt
    df['ax'] = df['vx'].diff() / dt
    df['ay'] = df['vy'].diff() / dt

    # 컨텍스트 특징
    if use_context:
        df['aspect_ratio'] = df['w'] / (df['h'] + 1e-6)
        df['area_ratio'] = (df['w'] * df['h']) / (img_width * img_height + 1e-6)
        # 추가적으로 frame_ratio 등도 가능
        # if len(df) > 1: df['frame_ratio'] = (df['frame'] - df['frame'].min()) / (df['frame'].max() - df['frame'].min() + 1e-6) else: df['frame_ratio'] = 0.0

    return df.dropna().reset_index(drop=True)


class TrajectoryAugmenterPhysicsAware:
    """
    [개선됨] 물리 제약 조건을 고려한 궤적 시퀀스 데이터 증강.
    """
    def __init__(self, feature_cols: list, noise_scale: float = 0.01, scale_range: tuple = (0.9, 1.1), time_warp_ratio: float = 0.1, max_speed_norm: float = 1.0, max_accel_norm: float = 0.5):
        self.feature_cols = feature_cols
        self.noise_scale = noise_scale
        self.scale_range = scale_range
        self.time_warp_ratio = time_warp_ratio
        self.max_speed_norm = max_speed_norm
        self.max_accel_norm = max_accel_norm
        try: # 특징 인덱스 미리 찾기
            self.idx = {col: feature_cols.index(col) for col in ['x_center', 'y_center', 'w_norm', 'h_norm', 'vx', 'vy', 'ax', 'ay'] if col in feature_cols}
        except ValueError:
            warnings.warn("Physics Augmenter missing required features. Constraints might not work.")
            self.idx = {}

    def _apply_constraints(self, seq: np.ndarray) -> np.ndarray:
        if not all(k in self.idx for k in ['vx', 'vy', 'ax', 'ay', 'x_center', 'y_center', 'w_norm', 'h_norm']):
            return seq # 필수 특징 없으면 제약 적용 불가

        vx_idx, vy_idx, ax_idx, ay_idx = self.idx['vx'], self.idx['vy'], self.idx['ax'], self.idx['ay']
        x_idx, y_idx, w_idx, h_idx = self.idx['x_center'], self.idx['y_center'], self.idx['w_norm'], self.idx['h_norm']

        # 속도 제한
        speeds = np.sqrt(seq[:, vx_idx]**2 + seq[:, vy_idx]**2)
        speed_scale = np.minimum(1.0, self.max_speed_norm / (speeds + 1e-6))[:, np.newaxis] # 브로드캐스팅 위해 차원 추가
        seq[:, [vx_idx, vy_idx]] *= speed_scale

        # 가속도 제한
        accels = np.sqrt(seq[:, ax_idx]**2 + seq[:, ay_idx]**2)
        accel_scale = np.minimum(1.0, self.max_accel_norm / (accels + 1e-6))[:, np.newaxis]
        seq[:, [ax_idx, ay_idx]] *= accel_scale

        # 위치/크기 범위 제한 (Normalizer 범위 고려 필요 - 여기서는 -1~1 또는 0~1 가정)
        seq[:, x_idx] = np.clip(seq[:, x_idx], -1.1, 1.1)
        seq[:, y_idx] = np.clip(seq[:, y_idx], -1.1, 1.1)
        seq[:, w_idx] = np.clip(seq[:, w_idx], 1e-4, 1.1)
        seq[:, h_idx] = np.clip(seq[:, h_idx], 1e-4, 1.1)
        # Aspect ratio, area ratio 등 추가 특징도 필요시 clip 가능

        return seq

    # --- Jitter, Scale, Time Warp 함수는 Original 버전과 동일하게 사용 가능 ---
    def jitter(self, seq: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=seq.shape)
        return seq + noise

    def scale(self, seq: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        # 중요: 어떤 특징을 스케일링할지 결정 필요 (예: 위치, 크기만?)
        # 여기서는 예시로 전체 스케일링
        return seq * factor

    def time_warp(self, seq: np.ndarray) -> np.ndarray:
        # (Original 버전의 time_warp 로직 사용)
        time_steps, num_features = seq.shape
        original_time = np.linspace(0, time_steps - 1, num=time_steps)
        cumulative_offset = np.cumsum(np.random.uniform(-self.time_warp_ratio, self.time_warp_ratio, size=time_steps))
        speed_factor = np.random.uniform(1 - self.time_warp_ratio, 1 + self.time_warp_ratio)
        warped_time = (original_time + cumulative_offset) * speed_factor
        warped_time = np.clip(np.sort(warped_time), 0, time_steps - 1)
        warped_seq = np.zeros_like(seq)
        try:
            for i in range(num_features):
                interp_func = interp1d(original_time, seq[:, i], kind='linear', fill_value="extrapolate")
                warped_seq[:, i] = interp_func(warped_time)
        except ValueError: return seq
        return warped_seq

    def augment(self, seq: np.ndarray, p_jitter=0.7, p_scale=0.7, p_warp=0.5, apply_constraints=True) -> np.ndarray:
        augmented_seq = seq.copy()
        if np.random.rand() < p_jitter: augmented_seq = self.jitter(augmented_seq)
        if np.random.rand() < p_scale: augmented_seq = self.scale(augmented_seq)
        if np.random.rand() < p_warp: augmented_seq = self.time_warp(augmented_seq)

        if apply_constraints:
            augmented_seq = self._apply_constraints(augmented_seq) # 제약 조건 적용

        return augmented_seq

# =====================================================
#  Feature Normalizer (기존 방식 유지 - 유연성)
# =====================================================
class FeatureNormalizer:
    """
    입력 특징(위치, 크기, 동적 정보, 컨텍스트)을 정규화/표준화합니다.
    [개선됨] 컨텍스트 특징 처리 및 scaler 저장 방식 개선.
    """
    def __init__(self, feature_cols: list, pos_scaler='minmax', dyn_scaler='standard', size_scaler='minmax', ctx_scaler='minmax'):
        print(f"Initializing FeatureNormalizer for features: {feature_cols}")
        self.feature_cols = feature_cols
        self.scalers = {} # 각 특징 그룹별 Scaler 저장 { 'pos': scaler, 'size': scaler, ... }
        self.indices = {} # 각 특징 그룹별 인덱스 저장 { 'pos': [0, 1], ... }

        # 특징 그룹핑 및 인덱스 저장
        groups = {'pos': (['x_center', 'y_center'], pos_scaler),
                  'size': (['w_norm', 'h_norm'], size_scaler),
                  'dyn': (['vx', 'vy', 'ax', 'ay'], dyn_scaler),
                  'ctx': (['aspect_ratio', 'area_ratio'], ctx_scaler)} # 필요시 frame_ratio 등 추가

        for name, (cols, scaler_type) in groups.items():
            grp_indices = [i for i, col in enumerate(feature_cols) if col in cols]
            if grp_indices: # 해당 특징 그룹이 feature_cols에 존재하면
                self.indices[name] = grp_indices
                if scaler_type == 'minmax':
                    # size 그룹은 항상 (0, 1), 나머지는 (-1, 1) 기본값
                    range_ = (0, 1) if name == 'size' else (-1, 1)
                    self.scalers[name] = MinMaxScaler(feature_range=range_)
                elif scaler_type == 'standard':
                    self.scalers[name] = StandardScaler()
                else: raise ValueError(f"Invalid scaler_type '{scaler_type}' for group '{name}'.")

        self.is_fitted = False
        # 이미지 크기 정보 저장 (역변환 등에 필요 시)
        self.img_width = None
        self.img_height = None

    def _apply_scalers(self, X: np.ndarray, method: str) -> np.ndarray:
        if X.ndim != 3 or X.shape[-1] != len(self.feature_cols):
             raise ValueError(f"Input array X shape error: expected (samples, time_steps, {len(self.feature_cols)}), got {X.shape}")
        if X.shape[0] == 0: return X

        num_samples, time_steps, num_features = X.shape
        X_transformed = X.copy().astype(float) # 원본 복사 및 float 타입 변환
        X_reshaped = X_transformed.reshape(-1, num_features) # Scaler 적용 위해 reshape

        for name, scaler in self.scalers.items():
            if name in self.indices:
                grp_indices = self.indices[name]
                scaler_method = getattr(scaler, method) # 'fit_transform' or 'transform'
                try:
                     # 선택된 인덱스에 대해 scaler 적용
                     X_reshaped[:, grp_indices] = scaler_method(X_reshaped[:, grp_indices])
                except Exception as e:
                     print(f"Error applying scaler '{name}' ({method}) to indices {grp_indices}: {e}")
                     # 오류 발생 시 해당 그룹은 원본 값 유지 또는 다른 처리 가능
                     # 여기서는 일단 진행

        # 변환된 2D 배열을 다시 3D로 reshape
        return X_reshaped.reshape(num_samples, time_steps, num_features)

    def fit_transform(self, X: np.ndarray, img_width=None, img_height=None) -> np.ndarray:
        """Scaler fit 및 transform."""
        # 이미지 크기 저장 (옵션)
        if img_width: self.img_width = img_width
        if img_height: self.img_height = img_height

        result = self._apply_scalers(X, 'fit_transform')
        self.is_fitted = True
        print("Normalizer fitted.")
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """이미 fit된 Scaler로 transform."""
        if not self.is_fitted: raise RuntimeError("Scaler not fitted.")
        return self._apply_scalers(X, 'transform')

    # --- 필요 시 역변환 함수 추가 ---
    def inverse_transform_bbox(self, X_norm: np.ndarray) -> np.ndarray:
         """정규화된 bbox(x_c, y_c, w_n, h_n)를 원본 스케일로 역변환 (근사치)"""
         if not self.is_fitted or self.img_width is None:
             raise RuntimeError("Scaler not fitted or img_width/height not set.")
         if X_norm.ndim != 2 or X_norm.shape[-1] != 4:
             raise ValueError("Input must be (N, 4) bbox array.")

         X_inv = X_norm.copy()
         # 가정: pos는 (-1,1) minmax, size는 (0,1) minmax
         if 'pos' in self.scalers: X_inv[:, :2] = self.scalers['pos'].inverse_transform(X_norm[:, :2])
         if 'size' in self.scalers: X_inv[:, 2:] = self.scalers['size'].inverse_transform(X_norm[:, 2:])

         # 정규화 역연산 (근사치)
         x_center = X_inv[:, 0] * self.img_width
         y_center = X_inv[:, 1] * self.img_height
         w = X_inv[:, 2] * self.img_width
         h = X_inv[:, 3] * self.img_height
         x1 = x_center - w / 2
         y1 = y_center - h / 2
         return np.stack([x1, y1, w, h], axis=-1) # x1, y1, w, h 반환


# =====================================================
#  LSTM Dataset Class (조건부 로직 통합)
# =====================================================
class LSTMDataset(Dataset):
    """
    [최종] PyTorch Dataset 클래스. 설정에 따라 기존/개선된 파이프라인 선택 가능.
    """
    def __init__(
        self,
        mot_root: str,
        cfg: dict,
        normalizer: FeatureNormalizer = None,
        is_train: bool = True
    ):
        use_enhanced = cfg.get('use_enhanced_pipeline', False) # 메인 플래그
        pipeline_type = 'Enhanced' if use_enhanced else 'Original'
        print(f"--- Initializing LSTMDataset ({pipeline_type} Pipeline, {'Train' if is_train else 'Val/Test'}) ---")

        self.cfg = cfg
        self.is_train = is_train
        self.time_steps = cfg['window_size']
        self.img_width = cfg['img_width']
        self.img_height = cfg['img_height']
        self.fps = cfg['fps']

        # --- 1. 특징 컬럼 결정 ---
        self.base_features = ['x_center','y_center','w_norm','h_norm','vx','vy','ax','ay']
        self.context_features = ['aspect_ratio', 'area_ratio'] # 필요시 frame_ratio 추가
        if use_enhanced and cfg.get('use_context_features', False):
            self.feature_cols = self.base_features + self.context_features
            print("Using features with context.")
        else:
            self.feature_cols = self.base_features
            print("Using base features only.")
        self.num_features = len(self.feature_cols)
        # cfg에 실제 사용될 특징 차원 업데이트 (모델 초기화 시 사용)
        self.cfg['feat_dim'] = self.num_features

        # --- 2. 데이터 로드 ---
        df_raw = load_mot_data(mot_root, self.img_width, self.img_height)
        if df_raw.empty:
             self._handle_empty_dataset("No data loaded.")
             return

        all_sequences = []
        all_targets = []
        processed_tracks = 0
        total_tracks = df_raw.groupby(['seq', 'track_id']).ngroups
        print(f"Processing {total_tracks} tracks...")

        # --- 3. 트랙별 처리 루프 ---
        for (seq_name, track_id), track_df in df_raw.groupby(['seq', 'track_id']):
            if len(track_df) < 2: continue

            # --- 3.1 보간 (조건부) ---
            if use_enhanced and cfg.get('use_hierarchical_interpolation', True):
                 df_processed = handle_frame_gaps_hierarchical(track_df, cfg['max_gap_interpolation'], cfg.get('max_gap_spline', 5), self.img_width, self.img_height, cfg.get('interpolation_max_speed_ratio', 1.5))
            else:
                 df_processed = handle_frame_gaps_original(track_df, cfg['max_gap_interpolation'])

            if df_processed.empty or len(df_processed) < self.time_steps + 1 : continue

            # --- 3.2 특징 계산 (조건부) ---
            if use_enhanced:
                 df_processed = calculate_dynamics_with_context(df_processed, self.fps, self.img_width, self.img_height, cfg.get('use_context_features', False))
            else:
                 df_processed = calculate_dynamics_original(df_processed, self.fps, self.img_width, self.img_height)

            if df_processed.empty or len(df_processed) < self.time_steps + 1: continue

            # --- 3.3 시퀀스 생성 ---
            sequences_raw = create_sequences(df_processed, self.time_steps, self.feature_cols)
            if sequences_raw.shape[0] == 0: continue

            # --- 3.4 타겟 생성 (정규화된 x_c, y_c, w_n, h_n) ---
            target_indices = df_processed.index[self.time_steps : len(df_processed)]
            required_target_cols = ['x_center', 'y_center', 'w_norm', 'h_norm']
            if not all(col in df_processed.columns for col in required_target_cols):
                 warnings.warn(f"Target columns missing in track {track_id}. Skipping.")
                 continue

            # 길이 불일치 시 조정
            min_len = min(len(sequences_raw), len(target_indices))
            if min_len == 0: continue
            sequences_raw = sequences_raw[:min_len]
            target_indices = target_indices[:min_len]
            targets = df_processed.loc[target_indices, required_target_cols].values

            # --- 3.5 시퀀스 검증 (개선 파이프라인 & 플래그 True 시) ---
            if use_enhanced and cfg.get('validate_sequences', True):
                valid_mask = np.array([validate_sequence_physics(
                                         seq, self.feature_cols,
                                         max_speed_norm=cfg.get('validation_max_speed_norm', 1.0),
                                         max_accel_norm=cfg.get('validation_max_accel_norm', 0.5))
                                       for seq in sequences_raw])
                if not np.any(valid_mask): continue
                sequences_raw = sequences_raw[valid_mask]
                targets = targets[valid_mask]

            if sequences_raw.shape[0] > 0:
                all_sequences.append(sequences_raw)
                all_targets.append(targets)
            processed_tracks += 1
        # --- 루프 종료 ---

        print(f"Finished processing tracks. Generated sequences from {processed_tracks} tracks.")

        if not all_sequences:
            self._handle_empty_dataset("No valid sequences generated.")
            return

        self.X_raw = np.concatenate(all_sequences, axis=0)
        self.y = np.concatenate(all_targets, axis=0)
        print(f"Total sequences generated: {len(self.X_raw)}")

        # --- 4. 특징 정규화 ---
        if is_train:
             print("Fitting normalizer...")
             # Normalizer 초기화 시 cfg에서 스케일러 타입 로드 가능
             pos_s = cfg.get('pos_scaler_type', 'minmax')
             dyn_s = cfg.get('dyn_scaler_type', 'standard')
             size_s= cfg.get('size_scaler_type', 'minmax')
             ctx_s = cfg.get('ctx_scaler_type', 'minmax')
             self.normalizer = FeatureNormalizer(self.feature_cols, pos_s, dyn_s, size_s, ctx_s)
             self.X = self.normalizer.fit_transform(self.X_raw, self.img_width, self.img_height)
             print(f"Normalizer fitted with scalers: pos={pos_s}, dyn={dyn_s}, size={size_s}, ctx={ctx_s}")
        elif normalizer is not None and normalizer.is_fitted:
             print("Applying provided normalizer...")
             self.normalizer = normalizer
             # --- 중요: 제공된 normalizer의 feature_cols와 현재 데이터셋의 feature_cols 일치 확인 ---
             if self.normalizer.feature_cols != self.feature_cols:
                  warnings.warn(f"Feature mismatch! Provided normalizer features: {self.normalizer.feature_cols}, Current dataset features: {self.feature_cols}. Results may be unpredictable.")
             self.X = self.normalizer.transform(self.X_raw)
        else:
             raise ValueError("Normalizer must be provided and fitted for validation/test dataset.")

        # --- 5. 데이터 증강기 초기화 (학습 시) ---
        self.augmenter = None
        if self.is_train and cfg.get('use_augmentation', True):
             aug_cfg = {k: cfg.get(k) for k in cfg if k.startswith('aug_')} # 증강 관련 설정 추출
             if use_enhanced and cfg.get('use_physics_augmentation', True):
                 self.augmenter = TrajectoryAugmenterPhysicsAware(self.feature_cols, **aug_cfg)
                 print("Using Physics-Aware Augmenter.")
             else:
                 self.augmenter = TrajectoryAugmenterOriginal(**aug_cfg)
                 print("Using Original Augmenter.")
        else:
             print("Data augmentation disabled for this dataset.")

        print(f"--- Dataset Initialized: {len(self.X)} samples ---")
        print(f"Input shape (X): {self.X.shape}")
        print(f"Target shape (y): {self.y.shape}")


    def _handle_empty_dataset(self, reason: str):
        """빈 데이터셋 처리"""
        print(f"ERROR: {reason}. Dataset will be empty.")
        self.X = np.empty((0, self.cfg.get('window_size', 10), self.num_features)) # window_size 접근
        self.y = np.empty((0, 4)) # target shape
        self.normalizer = None
        self.augmenter = None
        self.feature_cols = []
        self.num_features = 0


    def __len__(self):
        return len(self.X) if hasattr(self, 'X') and self.X is not None else 0

    def __getitem__(self, idx):
        if not hasattr(self, 'X') or self.X is None: raise IndexError("Dataset is empty")

        x_seq = self.X[idx].copy() # 증강 위해 복사
        y_tgt = self.y[idx]

        if self.is_train and self.augmenter:
            x_seq = self.augmenter.augment(x_seq) # 각 augmenter의 augment 메소드 호출

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_tgt, dtype=torch.float32)

        return x_tensor, y_tensor