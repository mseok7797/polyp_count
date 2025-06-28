import pandas as pd
import motmetrics as mm
from pathlib import Path

def compute_mot_metrics(gt_root: str, res_root: str, seqs: list = None, output_csv: str = 'mot_metrics_summary.csv'):
    """
    Compute MOT metrics (MOTA, IDF1, ID switches) for each sequence and overall summary.
    
    - gt_root: path to MOT-format ground truth root (each sequence has gt/gt.txt)
    - res_root: path to tracker results (each file named <seq>.txt in MOT format)
    - seqs: optional list of sequence names to evaluate
    - output_csv: filename for summary CSV
    
    Returns:
        pandas.DataFrame with metrics per sequence and overall.
    """
    metrics_list = []
    accs = []

    for seq_dir in Path(gt_root).iterdir():
        seq = seq_dir.name
        if seqs and seq not in seqs:
            continue
        
        gt_path = seq_dir / 'gt' / 'gt.txt'
        res_path = Path(res_root) / f'{seq}.txt'
        if not gt_path.exists() or not res_path.exists():
            print(f"Skipping {seq}: missing gt or result file")
            continue
        
        # Load
        gt = pd.read_csv(gt_path, header=None, names=['frame','id','x','y','w','h','conf','cls','vis'])
        res = pd.read_csv(res_path, header=None, names=['frame','id','x','y','w','h','conf','cls','vis'])
        
        # Initialize accumulator
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Per-frame updates
        frames = sorted(gt['frame'].unique())
        for f in frames:
            gt_f = gt[gt['frame'] == f]
            res_f = res[res['frame'] == f]
            # IDs
            gt_ids = gt_f['id'].values
            res_ids = res_f['id'].values
            # IoU distance matrix
            dists = mm.distances.iou_matrix(
                gt_f[['x','y','w','h']].values,
                res_f[['x','y','w','h']].values,
                max_iou=0.5
            )
            acc.update(gt_ids, res_ids, dists)
        
        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames','mota','idf1','id_switches'], name=seq)
        metrics_list.append(summary)
        accs.append(acc)

    # Overall summary
    if accs:
        overall = mm.metrics.compute_many(
            accs, names=[a.name for a in metrics_list],
            metrics=['num_frames','mota','idf1','id_switches'],
            generate_overall=True
        )
        metrics_list.append(overall)

    # Combine and save
    metrics_df = pd.concat(metrics_list)
    metrics_df[['mota','idf1']] *= 100  # convert to percentages
    metrics_df.to_csv(output_csv)
    print(f"Saved summary to {output_csv}")
    return metrics_df

# usage (adjust paths as needed):
gt_root = '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train'
res_root = './runs/track'
df = compute_mot_metrics(gt_root, res_root)
# df


''' count id 추가
import pandas as pd
import motmetrics as mm
from pathlib import Path


def compute_mot_metrics(
    gt_root: str,
    res_root: str,
    seqs: list = None,
    output_csv: str = 'mot_metrics_summary.csv'
) -> pd.DataFrame:
    """
    Compute MOT metrics (MOTA, IDF1, ID switches) for each sequence and overall summary,
    adding GT_IDs (number of unique ground truth IDs) and Count_IDs (number of predicted IDs).
    """
    metrics_list = []
    accs = []
    info = []  # track GT_ID and Count_ID per sequence

    for seq_dir in Path(gt_root).iterdir():
        seq = seq_dir.name
        if seqs and seq not in seqs:
            continue

        gt_path = seq_dir / 'gt' / 'gt.txt'
        res_path = Path(res_root) / f'{seq}.txt'
        if not gt_path.exists() or not res_path.exists():
            print(f"Skipping {seq}: missing gt or result file")
            continue

        # Load annotations
        gt = pd.read_csv(
            gt_path, header=None,
            names=['frame','id','x','y','w','h','conf','cls','vis']
        )
        res = pd.read_csv(
            res_path, header=None,
            names=['frame','id','x','y','w','h','conf','cls','vis']
        )

        # Count unique IDs
        n_gt_ids = gt['id'].nunique()
        n_res_ids = res['id'].nunique()
        info.append({'seq': seq, 'GT_IDs': n_gt_ids, 'Count_IDs': n_res_ids})

        # Accumulator for metrics
        acc = mm.MOTAccumulator(auto_id=True)
        for frame in sorted(gt['frame'].unique()):
            gt_f = gt[gt['frame'] == frame]
            res_f = res[res['frame'] == frame]
            gt_ids = gt_f['id'].values
            res_ids = res_f['id'].values
            dists = mm.distances.iou_matrix(
                gt_f[['x','y','w','h']].values,
                res_f[['x','y','w','h']].values,
                max_iou=0.5
            )
            acc.update(gt_ids, res_ids, dists)

        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=['num_frames','mota','idf1','id_switches'],
            name=seq
        )
        metrics_list.append(summary)
        accs.append(acc)

    # Overall metrics
    if accs:
        overall = mm.metrics.compute_many(
            accs,
            names=[m.name for m in metrics_list],
            metrics=['num_frames','mota','idf1','id_switches'],
            generate_overall=True
        )
        overall_name = 'OVERALL'
        overall_row = pd.DataFrame([{ 'num_frames': overall.loc['OVERALL','num_frames'],
                                      'mota': overall.loc['OVERALL','mota'],
                                      'idf1': overall.loc['OVERALL','idf1'],
                                      'id_switches': overall.loc['OVERALL','id_switches'],
                                      'seq': overall_name,
                                      'GT_IDs': sum(i['GT_IDs'] for i in info),
                                      'Count_IDs': sum(i['Count_IDs'] for i in info) }])
        overall_row.set_index('seq', inplace=True)
        overall.name = overall_name
        metrics_list.append(overall)
        info.append({'seq': overall_name, 'GT_IDs': sum(i['GT_IDs'] for i in info), 'Count_IDs': sum(i['Count_IDs'] for i in info)})

    # Combine metrics and info
    metrics_df = pd.concat(metrics_list)
    info_df = pd.DataFrame(info).set_index('seq')
    # Join on sequence name
    df = metrics_df.join(info_df, how='left')
    df[['mota','idf1']] = df[['mota','idf1']] * 100  # to percentages

    df.to_csv(output_csv)
    print(f"Saved summary to {output_csv}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, required=True)
    parser.add_argument('--res_root', type=str, required=True)
    parser.add_argument('--output', type=str, default='mot_metrics_summary.csv')
    args = parser.parse_args()
    compute_mot_metrics(args.gt_root, args.res_root, output_csv=args.output)

    '''

'''
# Command
python metrics.py --gt_root /home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train \
                  --res_root ./runs/track \
                  --output mot_metrics_summary.csv
'''