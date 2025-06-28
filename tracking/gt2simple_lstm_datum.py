# 직접 사용 X data processing 재료로..
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_mot_data(file_path, img_width=640, img_height=480):
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'active', 'class', 'visibility']
    df = pd.read_csv(file_path, header=None, names=columns)
    
    # 좌표 정규화
    df['x_center'] = (df['x'] + df['w']/2) / img_width
    df['y_center'] = (df['y'] + df['h']/2) / img_height
    df['w_norm'] = df['w'] / img_width
    df['h_norm'] = df['h'] / img_height
    
    return df.sort_values(['id', 'frame'])

def handle_frame_gaps(track, max_gap=5):
    """객체 궤적의 프레임 간격 처리"""
    frames = track['frame'].values
    diffs = np.diff(frames)
    
    # 간격이 max_gap 이상인 지점에서 세그먼트 분할
    split_indices = np.where(diffs > max_gap)[0] + 1
    segments = np.split(track, split_indices)
    
    # 각 세그먼트에 대한 보간 수행
    processed = []
    for seg in segments:
        if len(seg) < 2:
            continue
        f = interp1d(seg['frame'], seg[['x_center', 'y_center', 'w_norm', 'h_norm']], 
                     axis=0, kind='linear', fill_value="extrapolate")
        new_frames = np.arange(seg['frame'].min(), seg['frame'].max()+1)
        interpolated = f(new_frames)
        processed.append(pd.DataFrame({
            'frame': new_frames,
            'x_center': interpolated[:,0],
            'y_center': interpolated[:,1],
            'w_norm': interpolated[:,2],
            'h_norm': interpolated[:,3]
        }))
    
    return pd.concat(processed) if processed else None

def calculate_dynamics(track):
    """속도 및 가속도 계산"""
    track = track.sort_values('frame')
    track['vx'] = track['x_center'].diff() * 30  # 30 FPS 가정
    track['vy'] = track['y_center'].diff() * 30
    track['ax'] = track['vx'].diff() * 30
    track['ay'] = track['vy'].diff() * 30
    return track.dropna()

def create_sequences(track, time_steps=10):
    """슬라이딩 윈도우 기반 시퀀스 생성"""
    features = ['x_center', 'y_center', 'w_norm', 'h_norm', 'vx', 'vy', 'ax', 'ay']
    data = track[features].values
    sequences = []
    
    for i in range(len(data) - time_steps):
        seq = data[i:i+time_steps]
        sequences.append(seq)
        
    return np.array(sequences)

from sklearn.preprocessing import MinMaxScaler

class FeatureNormalizer:  # 특징별 데이터 정규화
    def __init__(self):
        self.scalers = {
            'position': MinMaxScaler(feature_range=(-1, 1)),
            'size': MinMaxScaler(feature_range=(0, 1)),
            'dynamics': MinMaxScaler(feature_range=(-1, 1))
        }
    
    def fit_transform(self, data):
        # 위치 특징 (x_center, y_center)
        pos_data = data[..., :2]
        pos_norm = self.scalers['position'].fit_transform(pos_data.reshape(-1, 2)).reshape(data.shape)
        
        # 크기 특징 (w_norm, h_norm)
        size_data = data[..., 2:4]
        size_norm = self.scalers['size'].fit_transform(size_data.reshape(-1, 2)).reshape(data.shape)
        
        # 동적 특징 (vx, vy, ax, ay)
        dyn_data = data[..., 4:]
        dyn_norm = self.scalers['dynamics'].fit_transform(dyn_data.reshape(-1, 4)).reshape(data.shape)
        
        return np.concatenate([pos_norm, size_norm, dyn_norm], axis=-1)

class TrajectoryAugmenter:  # 시공간적 데이터 증강
    def __init__(self, noise_scale=0.01, max_scale=0.1, time_warp_ratio=0.2):
        self.noise_scale = noise_scale
        self.max_scale = max_scale
        self.time_warp_ratio = time_warp_ratio
        
    def jitter(self, sequence):
        noise = np.random.normal(0, self.noise_scale, sequence.shape)
        return sequence + noise
    
    def scale_boxes(self, sequence):
        scale = 1 + np.random.uniform(-self.max_scale, self.max_scale)
        return sequence * scale
    
    def time_warp(self, sequence):
        orig_steps = np.arange(sequence.shape[0])
        new_steps = orig_steps * np.random.uniform(1-self.time_warp_ratio, 1+self.time_warp_ratio)
        warped = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            warped[:,i] = np.interp(new_steps, orig_steps, sequence[:,i])
        return warped
    
    def augment(self, sequence):
        aug_seq = self.jitter(sequence)
        aug_seq = self.scale_boxes(aug_seq)
        return self.time_warp(aug_seq)

def process_mot_to_lstm_input(file_path, time_steps=10):
    # 데이터 로드 및 전처리
    df = load_mot_data(file_path)
    
    all_sequences = []
    for obj_id, group in df.groupby('id'):
        # 프레임 불연속 처리
        processed = handle_frame_gaps(group)
        if processed is None or len(processed) < time_steps+1:
            continue
            
        # 동적 특징 계산
        with_dynamics = calculate_dynamics(processed)
        
        # 시퀀스 생성
        sequences = create_sequences(with_dynamics, time_steps)
        all_sequences.append(sequences)
    
    # 데이터 정규화
    normalizer = FeatureNormalizer()
    X = np.concatenate(all_sequences)
    X_normalized = normalizer.fit_transform(X)
    
    return X_normalized

# 도메인 특성에 따른 동적 max_gap 설정 예시
def dynamic_max_gap(fps):
    return int(fps * 1.5)  # 1.5초 간격 허용

class OnlineNormalizer:
    def __init__(self, initial_data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(initial_data)
    
    def update(self, new_data):
        partial_fit_scaler(self.scaler, new_data)  # 점진적 학습 구현 필요

def physics_constrained_augment(seq):
    # 속도 방향 변경 제한
    last_vx = seq[-1,4]
    augmented = aug.augment(seq)
    if np.sign(augmented[-1,4]) != np.sign(last_vx):
        augmented[:,4] *= -1  # 속도 방향 유지
    return augmented

def validate_sequence(seq):
    # 물리적 가능성 검증
    max_speed = 5.0  # 픽셀/프레임 단위
    if np.any(np.abs(seq[:,4:6]) > max_speed):
        return False
    return True
