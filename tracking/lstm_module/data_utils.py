import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

'''def load_mot_annotations(mot_root: str) -> pd.DataFrame:
    dfs = []
    for seq_dir in Path(mot_root).iterdir():
        gt_path = seq_dir / 'gt' / 'gt.txt'
        if not gt_path.exists():
            continue
        df = pd.read_csv(gt_path, header=None,
                         names=['frame','track_id','x','y','w','h','conf','cls','vis'])
        df = df[['frame','track_id','x','y','w','h']].copy()
        df['seq'] = seq_dir.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class TrackSequenceDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, window_size: int = 5):
        self.samples = []
        grouped = annotations.groupby(['seq','track_id'])
        for (_, _), df in grouped:
            df = df.sort_values('frame')
            boxes = df[['x','y','w','h']].values
            if len(boxes) < window_size+1:
                continue
            for i in range(len(boxes)-window_size):
                x_seq = boxes[i:i+window_size]
                y_tgt = boxes[i+window_size]
                self.samples.append((x_seq, y_tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y_tgt = self.samples[idx]
        import torch
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_tgt, dtype=torch.float32)'''