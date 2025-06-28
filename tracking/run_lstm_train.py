#!/usr/bin/env python
import yaml
import torch
import logging
from pathlib import Path

from lstm_module.data_utils import load_mot_annotations, TrackSequenceDataset
from lstm_module.models     import TrackLSTM
from lstm_module.trainer    import train_lstm

def main():
    # 1) load config
    cfg_path = Path(__file__).resolve().parent / 'lstm.yaml'
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('run_lstm_train')
    log.info(f"Loaded config from {cfg_path}")

    # 2) load annotations & build dataset
    ann = load_mot_annotations(cfg['mot_root'])
    if ann.empty:
        log.error("No annotations found – check mot_root path")
        return
    ds = TrackSequenceDataset(ann, window_size=cfg['window_size'])
    log.info(f"Dataset: {len(ds)} samples")

    # 3) instantiate LSTM, optimizer, loss
    device = torch.device(cfg['device'])
    model = TrackLSTM(
        feat_dim      = cfg['feat_dim'],
        hidden_dim    = cfg['hidden_dim'],
        num_layers    = cfg['num_layers'],
        bidirectional = cfg['bidirectional']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = torch.nn.MSELoss()

    # 4) train
    train_lstm(
        dataset   = ds,
        model     = model,
        optimizer = optimizer,
        criterion = criterion,
        epochs    = cfg['epochs'],
        device    = device
    )

    # 5) save weights
    torch.save(model.state_dict(), cfg['lstm_weights'])
    log.info(f"Saved LSTM weights to {cfg['lstm_weights']}")

if __name__ == '__main__':
    main()