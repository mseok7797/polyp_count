import yaml, subprocess
from lstm_module.patcher import patch_deepocsort
# 1) LSTM checkpoint 로드
# 2) patch_deepocsort(lstm_model, cfg['track_config'])
# 3) subprocess.call(['python','tracking/track.py', …])  # or 직접 API 호출
