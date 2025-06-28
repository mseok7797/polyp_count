import torch
import torch.nn as nn
import warnings
from typing import Dict, Any, List # 타입 힌트 추가

# --- WeightDrop 적용을 위한 Helper 함수, 현재 비활성화 ---
# (출처: SalesForce/PyTorch-QRNN 등에서 영감을 받은 일반적인 구현)
'''def weight_drop(module: nn.Module, weights: List[str], dropout_p: float):
    """
    주어진 모듈의 특정 가중치에 드롭아웃(DropConnect)을 적용합니다.
    모듈을 직접 수정(in-place)합니다.

    Args:
        module (nn.Module): 가중치 드롭아웃을 적용할 모듈 (예: nn.LSTM).
        weights (list[str]): 드롭아웃을 적용할 가중치의 파라미터 이름 리스트
                             (예: ['weight_hh_l0']).
        dropout_p (float): 드롭아웃 확률 (0 ~ 1).
    """
    if dropout_p < 0 or dropout_p > 1:
        raise ValueError("dropout_p must be between 0 and 1")
    if dropout_p == 0: # 드롭아웃 확률이 0이면 적용할 필요 없음
        return module

    # 가중치 이름 확인
    valid_weights = [name for name in weights if hasattr(module, name)]
    if not valid_weights:
         warnings.warn(f"None of the specified weights {weights} found in module {module.__class__.__name__} for WeightDrop.")
         return module

    print(f"Applying WeightDrop (p={dropout_p}) to weights: {valid_weights} in {module.__class__.__name__}")

    for name_w in valid_weights:
        # 원본 가중치 파라미터 가져오고 삭제
        param = getattr(module, name_w)
        del module._parameters[name_w]

        # 드롭아웃 미적용 원본 가중치를 '_raw' 접미사 붙여 새로 등록
        module.register_parameter(name_w + '_raw', nn.Parameter(param.data))

        # 순전파 시 적용될 가중치 (초기에는 원본과 동일)
        # 이 파라미터는 forward_pre_hook에서 계속 업데이트됨
        setattr(module, name_w, param.data.clone())

    # forward_pre_hook: 모듈의 forward 메소드 실행 직전에 호출되는 훅
    def _weight_drop_hook(module, *args):
        # 현재 모드가 training일 때만 드롭아웃 마스크를 새로 생성하고 적용
        # eval 모드에서는 마지막 training 시 생성된 마스크가 적용된 가중치 사용 (또는 드롭아웃 미적용)
        # -> eval 시에도 dropout 적용하려면 이 로직 수정 필요 (일반적으론 training시에만)
        is_training = module.training

        for name_w in valid_weights:
            raw_w = getattr(module, name_w + '_raw')
            # Training 시에만 드롭아웃 마스크를 적용하여 실제 사용할 가중치 계산
            if is_training:
                # 드롭아웃 마스크 생성 및 적용
                # F.dropout은 기본적으로 Bernoulli 분포 사용
                dropped_w = nn.functional.dropout(raw_w, p=dropout_p, training=is_training)
                setattr(module, name_w, dropped_w)
            else:
                # Eval 시에는 드롭아웃 미적용 (원본 가중치 사용)
                # 또는 스케일링된 가중치 사용 (1-p) - nn.Dropout과 동일하게
                # 여기서는 nn.Dropout처럼 동작하도록 스케일링 미적용 (PyTorch 기본값)
                # 필요 시: setattr(module, name_w, raw_w * (1 - dropout_p))
                setattr(module, name_w, raw_w) # Eval 모드에서는 원본 가중치 사용

    # 모듈에 훅 등록
    module.register_forward_pre_hook(_weight_drop_hook)
    return module'''


class TrackLSTM(nn.Module):
    """
    [개선됨] Bounding Box 시퀀스를 입력받아 다음 프레임의 Bounding Box 변화량(delta)을 예측.
    설정에 따라 WeightDrop(DropConnect) 기능 적용 가능.
    """
    def __init__(self, cfg: Dict[str, Any]): # 설정 전체를 받도록 변경
        """
        Args:
            cfg (dict): 설정 딕셔너리. 필요한 키:
                        'feat_dim', 'hidden_dim', 'num_layers'.
                        선택적 키: 'bidirectional', 'dropout_prob',
                                  'use_enhanced_pipeline', 'use_weight_drop',
                                  'weight_drop_prob'.
        """
        super().__init__()
        self.predict_delta = True # 현재는 Delta 예측 고정

        # --- 설정값 추출 ---
        feat_dim = cfg.get('feat_dim')
        if feat_dim is None: raise ValueError("Configuration must contain 'feat_dim'")
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        bidirectional = cfg.get('bidirectional', False)
        output_dim = 4 # dx, dy, dw, dh 예측
        dropout_prob = cfg.get('dropout_prob', 0.0) # Layer 간 Dropout (기본값 0)

        # Layer 간 Dropout 설정
        effective_layer_dropout = dropout_prob if num_layers > 1 else 0.0
        if num_layers <= 1 and dropout_prob > 0:
             warnings.warn("dropout_prob > 0 has no effect when num_layers <= 1.")

        # --- 기본 LSTM 레이어 생성 ---
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,          # 입력 Shape: (batch, seq_len, features)
            bidirectional=bidirectional,
            dropout=effective_layer_dropout # Layer 간 Dropout
        )

        # --- WeightDrop 관련 로직 제거/비활성화 ---
        self.applied_weight_drop = False # 항상 False

        # LSTM 출력 차원 계산
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # 최종 출력을 위한 Fully Connected Layer
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # 모델 객체에 설정 저장 (체크포인트 저장/로드 시 사용)
        self.cfg = cfg

        # 초기화 정보 로그 출력
        print(f"Initialized TrackLSTM:")
        print(f"  Input Features (feat_dim): {feat_dim}")
        print(f"  Hidden Dim: {hidden_dim}, Num Layers: {num_layers}, Bidirectional: {bidirectional}")
        print(f"  Output Dim: {output_dim} (Predicting Delta: {self.predict_delta})")
        print(f"  Layer Dropout: {effective_layer_dropout:.2f}")
        print(f"  WeightDrop: {self.applied_weight_drop} (Disabled)")


    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        """ 순전파 함수 (기존과 동일) """
        lstm_out, (h_n, c_n) = self.lstm(seq_features)
        last_time_step_output = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step_output)
        return prediction