# positive, negative labeling
import csv
from collections import defaultdict
import random

def parse_gt_file(gt_filepath):
    """
    MOT Challenge 형식의 gt.txt 파일을 파싱합니다.
    conf=0인 항목은 무시합니다.

    Args:
        gt_filepath (str): gt.txt 파일 경로

    Returns:
        defaultdict: 프레임 번호를 키로, {id: bbox} 딕셔너리를 값으로 갖는 딕셔너리
                     bbox는 [bb_left, bb_top, bb_width, bb_height] 리스트입니다.
    """
    frame_data = defaultdict(lambda: defaultdict(list))
    try:
        with open(gt_filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # MOT Challenge 형식: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                # 필요한 정보만 추출하고 정수/실수형으로 변환
                try:
                    frame = int(row)
                    obj_id = int(row[1])
                    bb_left = float(row[2])
                    bb_top = float(row[3])
                    bb_width = float(row[4])
                    bb_height = float(row[5])
                    conf = int(float(row[6])) # conf는 정수형 플래그로 사용될 수 있음 [7]

                    # conf 값이 0이 아닌 경우 (무시하지 않는 경우)에만 데이터 추가 [7]
                    if conf!= 0:
                        bbox = [bb_left, bb_top, bb_width, bb_height]
                        frame_data[frame][obj_id] = bbox
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid row: {row} due to error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_filepath}")
        return None
    return frame_data

def group_by_id(frame_data):
    """
    프레임별 데이터를 객체 ID별 궤적으로 재구성합니다.

    Args:
        frame_data (defaultdict): parse_gt_file의 반환값

    Returns:
        defaultdict: 객체 ID를 키로, [(frame, bbox)] 리스트(프레임 순 정렬)를 값으로 갖는 딕셔너리
    """
    id_trajectories = defaultdict(list)
    if not frame_data:
        return id_trajectories

    # 프레임 번호 순서대로 순회
    sorted_frames = sorted(frame_data.keys())
    for frame in sorted_frames:
        for obj_id, bbox in frame_data[frame].items():
            id_trajectories[obj_id].append((frame, bbox))

    # 각 ID의 궤적을 프레임 번호 순으로 정렬 (이미 정렬되어 있을 가능성이 높지만 확인 차원)
    for obj_id in id_trajectories:
        id_trajectories[obj_id].sort(key=lambda x: x)

    return id_trajectories

def create_lstm_dataset(gt_filepath, time_steps=10, neg_sample_ratio=1.0):
    """
    gt.txt 파일로부터 LSTM 학습용 데이터셋을 생성합니다.

    Args:
        gt_filepath (str): gt.txt 파일 경로
        time_steps (int): LSTM 입력 시퀀스의 길이 (과거 프레임 수)
        neg_sample_ratio (float): 긍정 샘플 대비 부정 샘플 비율 (1.0이면 1:1)

    Returns:
        list: [(trajectory_sequence, candidate_bbox, label)] 튜플의 리스트
              trajectory_sequence: time_steps 길이의 bbox 리스트
              candidate_bbox: 단일 bbox 리스트
              label: 0 또는 1
        None: 파일 처리 중 오류 발생 시
    """
    frame_data = parse_gt_file(gt_filepath)
    if frame_data is None:
        return None

    id_trajectories = group_by_id(frame_data)
    dataset = 

    print(f"Total unique object IDs found: {len(id_trajectories)}")

    processed_ids = 0
    for obj_id, trajectory in id_trajectories.items():
        if processed_ids % 10 == 0:
             print(f"Processing ID {processed_ids}/{len(id_trajectories)}...")
        processed_ids += 1

        # 각 궤적에 대해 슬라이딩 윈도우 적용 [8]
        for i in range(len(trajectory) - time_steps):
            # 입력 시퀀스 (과거 time_steps 프레임)
            sequence_window = trajectory[i : i + time_steps]
            # 타겟 프레임 (다음 프레임)
            target_frame_info = trajectory[i + time_steps]

            # 시퀀스 윈도우와 타겟 프레임이 연속적인지 확인
            # (프레임 번호가 1씩 증가하는지 확인)
            is_continuous = True
            for k in range(time_steps - 1):
                if sequence_window[k+1]!= sequence_window[k] + 1:
                    is_continuous = False
                    break
            if not is_continuous or target_frame_info!= sequence_window[-1] + 1:
                continue # 연속적이지 않으면 이 윈도우 건너뛰기

            trajectory_sequence = [bbox for frame, bbox in sequence_window]
            target_frame_num = target_frame_info
            target_bbox = target_frame_info[1]

            # 1. 긍정 샘플 생성 (label=1)
            dataset.append((trajectory_sequence, target_bbox, 1))

            # 2. 부정 샘플 생성 (label=0)
            # 타겟 프레임에 있는 다른 객체들의 bbox 가져오기
            negative_candidates =
            if target_frame_num in frame_data:
                for other_id, other_bbox in frame_data[target_frame_num].items():
                    if other_id!= obj_id:
                        negative_candidates.append(other_bbox)

            # 부정 샘플 비율에 맞춰 샘플링 (선택 사항)
            num_positive = 1
            num_negative_to_sample = int(num_positive * neg_sample_ratio * len(negative_candidates))
            sampled_negatives = random.sample(negative_candidates, min(len(negative_candidates), num_negative_to_sample))


            for neg_bbox in sampled_negatives: # 샘플링된 부정 후보 사용
                 dataset.append((trajectory_sequence, neg_bbox, 0))


    print(f"Finished processing. Total samples generated: {len(dataset)}")
    return dataset

# --- 사용 예시 ---
gt_file = '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train' # 실제 gt.txt 파일 경로로 변경. ############ sub folder 순회할 수 있도록?????
sequence_length = 15 # LSTM 입력 시퀀스 길이 (조정 가능)
negative_ratio = 1.0 # 긍정 샘플 1개당 부정 샘플 비율 (조정 가능)

lstm_data = create_lstm_dataset(gt_file, time_steps=sequence_length, neg_sample_ratio=negative_ratio)

if lstm_data:
    print(f"\nSuccessfully created dataset with {len(lstm_data)} samples.")
    # 예시: 첫 5개 샘플 출력
    for idx, (seq, cand, lbl) in enumerate(lstm_data[:5]):
        print(f"\nSample {idx + 1}:")
        print(f"  Label: {lbl}")
        print(f"  Trajectory Sequence (length {len(seq)}):")
        # for frame_idx, bbox in enumerate(seq):
        #     print(f"    Frame {frame_idx}: {bbox}")
        print(f"    First BBox: {seq}")
        print(f"    Last BBox: {seq[-1]}")
        print(f"  Candidate BBox: {cand}")

    # 생성된 데이터셋(lstm_data)을 파일로 저장하거나 LSTM 모델 학습에 사용합니다.
    # 예: pickle로 저장
    # import pickle
    # with open('lstm_mot_dataset.pkl', 'wb') as f:
    #     pickle.dump(lstm_data, f)
    # print("\nDataset saved to lstm_mot_dataset.pkl")
else:
    print("Failed to create dataset.")