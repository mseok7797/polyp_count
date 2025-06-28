import os
import re
import shutil

# --- 이미지 복사 및 프레임 번호 재지정 ---
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

# --- GT 필터링 및 프레임 번호 재지정, labels.txt 복사 ---
def filter_and_renumber_txt(src_txt_folder, dest_txt_folder, gt_range):
    start_frame, end_frame = gt_range
    offset = start_frame - 1
    os.makedirs(dest_txt_folder, exist_ok=True)
    src_gt     = os.path.join(src_txt_folder, 'gt.txt')
    dst_gt     = os.path.join(dest_txt_folder, 'gt.txt')
    src_labels = os.path.join(src_txt_folder, 'labels.txt')
    dst_labels = os.path.join(dest_txt_folder, 'labels.txt')

    # gt.txt 처리
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
    # labels.txt 복사
    if os.path.exists(src_labels):
        shutil.copyfile(src_labels, dst_labels)

# --- seqinfo.ini 생성 ---
def write_seqinfo(dest_sequence, name, imDir, fps, length, w, h, ext, filename='seqinfo.ini'):
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
    path = os.path.join(dest_sequence, filename)
    with open(path, 'w') as f:
        f.write(content)

# --- 한 시퀀스 다중 목적지 처리 ---
def process_sequence(src_img_folder, src_txt_folder, frame_range, gt_range, seq_name,
                     destinations, frame_rate=30, imWidth=640, imHeight=480, imExt='.PNG'):
    for dest in destinations:
        dest_img = dest['img']
        dest_txt = dest['txt']
        # 이미지 복사 및 재지정
        num_imgs = copy_and_renumber_images(src_img_folder, dest_img, frame_range)
        # GT/labels 처리
        filter_and_renumber_txt(src_txt_folder, dest_txt, gt_range)
        # seqinfo.ini 생성
        dest_seq = os.path.dirname(dest_img)
        write_seqinfo(dest_seq, seq_name, os.path.basename(dest_img),
                      frame_rate, num_imgs, imWidth, imHeight, imExt)
        print(f"Processed {seq_name} -> {dest_seq}: {num_imgs} frames")

# --- 시퀀스별 정보 목록 ---
sequences_info = [
    # [polyp_test_4_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'frame_range': ('frame_000000', 'frame_011529'),
        'gt_range': (462, 10398),
        'seq_name': 'polyp_test_4_gt_04',
        'destinations': [
            {
                'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04/gt',
            },
            {
                'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04/gt',
            }
        ]
    },    
     {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'frame_range': ('frame_000461', 'frame_003638'),
        'gt_range': (462, 3639),
        'seq_name': 'polyp_test_4_gt_04-1-2',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1-2/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-1-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-1-2/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'frame_range': ('frame_004744', 'frame_005046'),
        'gt_range': (4744, 5047),
        'seq_name': 'polyp_test_4_gt_04-2',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-2/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-2/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'frame_range': ('frame_007188', 'frame_008530'),
        'gt_range': (7189, 8531),
        'seq_name': 'polyp_test_4_gt_04-3',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-3/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-3/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-3/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-3/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'frame_range': ('frame_009902', 'frame_010397'),
        'gt_range': (9903, 10398),
        'seq_name': 'polyp_test_4_gt_04-4',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-4/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-4/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-4/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_4_gt_04-4/gt'}
        ]
    },
    # [polyp_test_5_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'frame_range': ('frame_000000', 'frame_009527'),
        'gt_range': (121, 7497),
        'seq_name': 'polyp_test_5_gt_05',
        'destinations': [
            {
                'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05/gt'
            },
            {
                'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05/gt'
            }
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'frame_range': ('frame_000120', 'frame_000450'),
        'gt_range': (121, 451),
        'seq_name': 'polyp_test_5_gt_05-1',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-1/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-1/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-1/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-1/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'frame_range': ('frame_002444', 'frame_002607'),
        'gt_range': (2445, 2608),
        'seq_name': 'polyp_test_5_gt_05-2',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-2/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-2/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-2/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'frame_range': ('frame_004281', 'frame_004319'),
        'gt_range': (4282, 4320),
        'seq_name': 'polyp_test_5_gt_05-3',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-3/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-3/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-3/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-3/gt'}
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'frame_range': ('frame_007270', 'frame_007496'),
        'gt_range': (7271, 7497),
        'seq_name': 'polyp_test_5_gt_05-4',
        'destinations': [
            {'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-4/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-4/gt'},
            {'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-4/img1',
             'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_5_gt_05-4/gt'}
        ]
    },
    # [polyp_test_8_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'frame_range': ('frame_000000', 'frame_009229'),
        'gt_range': (1051, 7101),
        'seq_name': 'polyp_test_8_gt_08',
        'destinations': [
            {
                'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08/gt'
            },
            {
                'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08/gt'
            }
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'frame_range': ('frame_001050', 'frame_005128'),
        'gt_range': (1051, 5129),
        'seq_name': 'polyp_test_8_gt_08-1',
        'destinations': [
            {
                'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-1/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-1/gt'
            },
            {
                'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08-1/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08-1/gt'
            }
        ]
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'frame_range': ('frame_006125', 'frame_007100'),
        'gt_range': (6126, 7101),
        'seq_name': 'polyp_test_8_gt_08-2',
        'destinations': [
            {
                'img': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-2/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-2/gt'
            },
            {
                'img': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08-2/img1',
                'txt': '/home/kms2069/Projects/yolo_tracking/tracking/val_utils/data/polyp_test_mot/test/polyp_test_8_gt_08-2/gt'
            }
        ]
    }
]

# --- 실행 루프 ---
for seq in sequences_info:
    process_sequence(
        seq['src_img_folder'], seq['src_txt_folder'],
        seq['frame_range'], seq['gt_range'], seq['seq_name'],
        seq['destinations']
    )
