# polyp_test_1,2,3,6,7,9,10 train dataset split , label.txt 그대로 move
# gt.txt의 frame number / frame_XXXXX -> frame_000001부터로로 change. ex> (133, 3633) -> (1, 3500)

import os
import re
import shutil

# --- 이미지 복사 및 재번호 매기기 함수 ---
def copy_and_renumber_images(src_img_folder, dest_img_folder, frame_range):
    start_frame = int(re.search(r"\d+", frame_range[0]).group())
    end_frame   = int(re.search(r"\d+", frame_range[1]).group())
    pattern     = re.compile(r"frame_(\d{6})(\..+)")
    os.makedirs(dest_img_folder, exist_ok=True)
    counter = 0
    for fn in sorted(os.listdir(src_img_folder)):
        m = pattern.match(fn)
        if m:
            num = int(m.group(1))
            ext = m.group(2)
            if start_frame <= num <= end_frame:
                counter += 1
                new_name = f"frame_{counter:06d}{ext}"
                shutil.copy(
                    os.path.join(src_img_folder, fn),
                    os.path.join(dest_img_folder, new_name)
                )
    return counter

# --- GT 필터링 및 재번호 매기기 함수 ---
def filter_and_renumber_txt(src_txt_folder, dest_txt_folder, gt_range):
    start_frame, end_frame = gt_range
    offset = start_frame - 1
    os.makedirs(dest_txt_folder, exist_ok=True)

    # gt.txt 처리: 필터링 후 재번호 매기기
    src_gt = os.path.join(src_txt_folder, 'gt.txt')
    dst_gt = os.path.join(dest_txt_folder, 'gt.txt')
    with open(src_gt, 'r') as fin, open(dst_gt, 'w') as fout:
        for line in fin:
            parts = line.strip().split(',')
            try:
                old_frame = int(parts[0])
            except ValueError:
                continue
            if start_frame <= old_frame <= end_frame:
                new_frame = old_frame - offset
                parts[0] = str(new_frame)
                fout.write(','.join(parts) + '\n')

    # labels.txt 처리: 내용 그대로 복사
    src_labels = os.path.join(src_txt_folder, 'labels.txt')
    dst_labels = os.path.join(dest_txt_folder, 'labels.txt')
    if os.path.exists(src_labels):
        shutil.copyfile(src_labels, dst_labels)

# --- seqinfo.txt 생성 함수 ---
def write_seqinfo(dest_sequence, name, imDir, fps, length, w, h, ext):
    content = f"""[Sequence]
name={name}
imDir={imDir}
frameRate={fps}
seqLength={length}
imWidth={w}
imHeight={h}
imExt={ext}
"""
    os.makedirs(dest_sequence, exist_ok=True)
    with open(os.path.join(dest_sequence, 'seqinfo.txt'), 'w') as f:
        f.write(content)

# --- 한 시퀀스 처리 함수 ---
def process_sequence(src_img_folder, src_txt_folder, dest_img_folder, dest_txt_folder,
                     frame_range, gt_range, seq_name,
                     frame_rate=30, imWidth=640, imHeight=480, imExt='.PNG'):
    # 이미지 복사 및 재번호
    num_imgs = copy_and_renumber_images(src_img_folder, dest_img_folder, frame_range)
    # GT 필터링 및 재번호 매기기, labels 복사
    filter_and_renumber_txt(src_txt_folder, dest_txt_folder, gt_range)
    # seqinfo 생성
    dest_seq = os.path.dirname(dest_img_folder)
    write_seqinfo(dest_seq, seq_name, os.path.basename(dest_img_folder),
                  frame_rate, num_imgs, imWidth, imHeight, imExt)
    print(f"Processed {seq_name}: {num_imgs} images and GT frames renumbered from {gt_range[0]}")

# --- 시퀀스별 정보 목록 ---

sequences_info_train = [
# --- polyp_test_1_mot ---
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01/gt',
        'frame_range': ('frame_000132', 'frame_008910'),
        'gt_range': (133, 8911),
        'seq_name': 'polyp_test_1_gt_01'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-1/gt',
        'frame_range': ('frame_000132', 'frame_003632'),
        'gt_range': (133, 3633),
        'seq_name': 'polyp_test_1_gt_01-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-2/gt',
        'frame_range': ('frame_005751', 'frame_006499'),
        'gt_range': (5752, 6450),
        'seq_name': 'polyp_test_1_gt_01-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_1_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_1_gt_01-3/gt',
        'frame_range': ('frame_007845', 'frame_008910'),
        'gt_range': (7846, 8911),
        'seq_name': 'polyp_test_1_gt_01-3'
    },
# --- polyp_test_2_mot ---
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02/gt',
        'frame_range': ('frame_000000', 'frame_011987'),
        'gt_range': (571, 9755),
        'seq_name': 'polyp_test_2_gt_02'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-1/gt',
        'frame_range': ('frame_000570', 'frame_000634'),
        'gt_range': (571, 635),
        'seq_name': 'polyp_test_2_gt_02-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-2/gt',
        'frame_range': ('frame_001800', 'frame_005164'),
        'gt_range': (1801, 5165),
        'seq_name': 'polyp_test_2_gt_02-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-3/gt',
        'frame_range': ('frame_007005', 'frame_008115'),
        'gt_range': (7006, 8116),
        'seq_name': 'polyp_test_2_gt_02-3'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_2_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-4/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_2_gt_02-4/gt',
        'frame_range': ('frame_009740', 'frame_009754'),
        'gt_range': (9741, 9755),
        'seq_name': 'polyp_test_2_gt_02-4'
    },
    # polyp_test_3_mot
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03/gt',
        'frame_range': ('frame_000000', 'frame_012341'),
        'gt_range': (1136, 10567),
        'seq_name': 'polyp_test_3_gt_03'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-1/gt',
        'frame_range': ('frame_001135', 'frame_003334'),
        'gt_range': (1136, 3335),
        'seq_name': 'polyp_test_3_gt_03-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-2/gt',
        'frame_range': ('frame_005316', 'frame_009091'),
        'gt_range': (5317, 9092),
        'seq_name': 'polyp_test_3_gt_03-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_3_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_3_gt_03-3/gt',
        'frame_range': ('frame_010505', 'frame_010566'),
        'gt_range': (10506, 10567),
        'seq_name': 'polyp_test_3_gt_03-3'
    },
    # polyp_test_6_mot
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06/gt',
        'frame_range': ('frame_000000', 'frame_009008'),
        'gt_range': (211, 8332),
        'seq_name': 'polyp_test_6_gt_06'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-1/gt',
        'frame_range': ('frame_000210', 'frame_000532'),
        'gt_range': (211, 533),
        'seq_name': 'polyp_test_6_gt_06-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-2/gt',
        'frame_range': ('frame_002039', 'frame_003553'),
        'gt_range': (2040, 3554),
        'seq_name': 'polyp_test_6_gt_06-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-3/gt',
        'frame_range': ('frame_006510', 'frame_006759'),
        'gt_range': (6511, 6760),
        'seq_name': 'polyp_test_6_gt_06-3'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_6_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-4/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_6_gt_06-4/gt',
        'frame_range': ('frame_008278', 'frame_008331'),
        'gt_range': (8279, 8332),
        'seq_name': 'polyp_test_6_gt_06-4'
    },

    # --- polyp_test_7_mot ---
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07/gt',
        'frame_range': ('frame_000000', 'frame_007978'),
        'gt_range': (121, 6309),
        'seq_name': 'polyp_test_7_gt_07'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-1/gt',
        'frame_range': ('frame_000120', 'frame_000489'),
        'gt_range': (121, 490),
        'seq_name': 'polyp_test_7_gt_07-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-2/gt',
        'frame_range': ('frame_001844', 'frame_003081'),
        'gt_range': (1845, 3082),
        'seq_name': 'polyp_test_7_gt_07-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_7_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_7_gt_07-3/gt',
        'frame_range': ('frame_005155', 'frame_006308'),
        'gt_range': (5156, 6309),
        'seq_name': 'polyp_test_7_gt_07-3'
    },
    # --- polyp_test_9_mot ---
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09/gt',
        'frame_range': ('frame_000000', 'frame_008055'),
        'gt_range': (647, 5865),
        'seq_name': 'polyp_test_9_gt_09'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09-1/gt',
        'frame_range': ('frame_000646', 'frame_003187'),
        'gt_range': (645, 3188),
        'seq_name': 'polyp_test_9_gt_09-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_9_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_9_gt_09-2/gt',
        'frame_range': ('frame_005672', 'frame_005864'),
        'gt_range': (5673, 5865),
        'seq_name': 'polyp_test_9_gt_09-2'
    },
    # --- polyp_test_10_mot ---
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10/gt',
        'frame_range': ('frame_000000', 'frame_008257'),
        'gt_range': (168, 6510),
        'seq_name': 'polyp_test_10_gt_10'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-1/gt',
        'frame_range': ('frame_000167', 'frame_000439'),
        'gt_range': (168, 440),
        'seq_name': 'polyp_test_10_gt_10-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-2/gt',
        'frame_range': ('frame_001667', 'frame_002917'),
        'gt_range': (1668, 2918),
        'seq_name': 'polyp_test_10_gt_10-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_10_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/train/polyp_test_10_gt_10-3/gt',
        'frame_range': ('frame_005083', 'frame_006500'),
        'gt_range': (5084, 6501),
        'seq_name': 'polyp_test_10_gt_10-3'
    },
]
# --- 모든 시퀀스 처리 실행 ---
for seq in sequences_info_train:
    process_sequence(
        seq['src_img_folder'], seq['src_txt_folder'],
        seq['dest_img_folder'], seq['dest_txt_folder'],
        seq['frame_range'], seq['gt_range'], seq['seq_name']
    )