# frame number no change.

import os
import re
import shutil
'''
def copy_images(src_img_folder, dest_img_folder, frame_range):
    """
    이미지 파일을 src_img_folder에서 dest_img_folder로 복사합니다.
    frame_range: ('frame_XXXXXX', 'frame_XXXXXX') 형태로 제공되며, 파일명에서 숫자를 추출해 비교합니다.
    """
    # frame_range의 시작/끝 숫자 추출
    start_frame = int(re.search(r'\d+', frame_range[0]).group())
    end_frame = int(re.search(r'\d+', frame_range[1]).group())
    
    pattern = re.compile(r'frame_(\d{6})')
    copied_files = 0

    os.makedirs(dest_img_folder, exist_ok=True)

    for filename in sorted(os.listdir(src_img_folder)):
        match = pattern.search(filename)
        if match:
            frame_num = int(match.group(1))
            if start_frame <= frame_num <= end_frame:
                src_path = os.path.join(src_img_folder, filename)
                dest_path = os.path.join(dest_img_folder, filename)
                shutil.copy(src_path, dest_path)
                copied_files += 1
    return copied_files

def filter_txt(src_txt_folder, dest_txt_folder, gt_range):
    """
    GT와 labels 파일을 src_txt_folder에서 dest_txt_folder로 복사하면서,
    각 행의 첫 번째 숫자(프레임 번호)가 gt_range 내에 해당하는 행만 복사합니다.
    gt_range: (start, end) (예: (462, 10398))
    """
    os.makedirs(dest_txt_folder, exist_ok=True)
    for fname in ['gt.txt', 'labels.txt']:
        src_file = os.path.join(src_txt_folder, fname)
        dest_file = os.path.join(dest_txt_folder, fname)
        with open(src_file, 'r') as fin, open(dest_file, 'w') as fout:
            for line in fin:
                parts = line.strip().split(',')
                try:
                    frame_num = int(parts[0])
                except ValueError:
                    continue  # 숫자 파싱 실패 시 건너뜀
                if gt_range[0] <= frame_num <= gt_range[1]:
                    fout.write(line)

def write_seqinfo(dest_sequence, seq_name, img_folder_name, frame_rate, seq_length, imWidth, imHeight, imExt):
    """
    dest_sequence 폴더에 seqinfo.txt 파일을 생성합니다.
    """
    content = f"""[Sequence]
name={seq_name}
imDir={img_folder_name}
frameRate={frame_rate}
seqLength={seq_length}
imWidth={imWidth}
imHeight={imHeight}
imExt={imExt}
"""
    with open(os.path.join(dest_sequence, 'seqinfo.txt'), 'w') as f:
        f.write(content)

def process_sequence(src_img_folder, src_txt_folder, dest_img_folder, dest_txt_folder, 
                     frame_range, gt_range, seq_name, frame_rate=30, imWidth=640, imHeight=480, imExt='.PNG'):
    """
    주어진 경로 및 범위 정보를 이용해 하나의 시퀀스를 처리합니다.
    이미지 복사, GT/labels 파일 필터링, 그리고 seqinfo.txt 파일 생성을 진행합니다.
    """
    num_images = copy_images(src_img_folder, dest_img_folder, frame_range)
    filter_txt(src_txt_folder, dest_txt_folder, gt_range)
    dest_sequence = os.path.dirname(dest_img_folder)
    write_seqinfo(dest_sequence, seq_name, os.path.basename(dest_img_folder), frame_rate, num_images, imWidth, imHeight, imExt)
    print(f"Processed sequence '{seq_name}' with {num_images} images.")
'''
# frame number change ver.

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
sequences_info = [
    # [polyp_test_4_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04/gt',
        'frame_range': ('frame_000000', 'frame_011529'),
        'gt_range': (462, 10398),
        'seq_name': 'polyp_test_4_gt_04'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1/gt',
        'frame_range': ('frame_000461', 'frame_001361'),
        'gt_range': (462, 907),
        'seq_name': 'polyp_test_4_gt_04-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-1-2/gt',
        'frame_range': ('frame_000461', 'frame_003638'),
        'gt_range': (462, 3639),
        'seq_name': 'polyp_test_4_gt_04-1-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-2/gt',
        'frame_range': ('frame_004744', 'frame_005046'),
        'gt_range': (4744, 5047),
        'seq_name': 'polyp_test_4_gt_04-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-3/gt',
        'frame_range': ('frame_007188', 'frame_008530'),
        'gt_range': (7189, 8531),
        'seq_name': 'polyp_test_4_gt_04-3'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_4_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-4/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_4_gt_04-4/gt',
        'frame_range': ('frame_009902', 'frame_010397'),
        'gt_range': (9903, 10398),
        'seq_name': 'polyp_test_4_gt_04-4'
    },
    # [polyp_test_5_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05/gt',
        'frame_range': ('frame_000000', 'frame_009527'),
        'gt_range': (121, 7497),
        'seq_name': 'polyp_test_5_gt_05'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-1/gt',
        'frame_range': ('frame_000120', 'frame_000450'),
        'gt_range': (121, 451),
        'seq_name': 'polyp_test_5_gt_05-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-2/gt',
        'frame_range': ('frame_002444', 'frame_002607'),
        'gt_range': (2445, 2608),
        'seq_name': 'polyp_test_5_gt_05-2'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-3/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-3/gt',
        'frame_range': ('frame_004281', 'frame_004319'),
        'gt_range': (4282, 4320),
        'seq_name': 'polyp_test_5_gt_05-3'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_5_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-4/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_5_gt_05-4/gt',
        'frame_range': ('frame_007270', 'frame_007496'),
        'gt_range': (7271, 7497),
        'seq_name': 'polyp_test_5_gt_05-4'
    },
    # [polyp_test_8_mot] (대상: test)
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08/gt',
        'frame_range': ('frame_000000', 'frame_009229'),
        'gt_range': (1051, 7101),
        'seq_name': 'polyp_test_8_gt_08'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-1/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-1/gt',
        'frame_range': ('frame_001050', 'frame_005128'),
        'gt_range': (1051, 5129),
        'seq_name': 'polyp_test_8_gt_08-1'
    },
    {
        'src_img_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/img1',
        'src_txt_folder': '/home/kms2069/Projects/datasets/polyp_test_mot/polyp_test_8_mot/gt',
        'dest_img_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-2/img1',
        'dest_txt_folder': '/home/kms2069/Projects/yolo_tracking/assets/polyp_test_mot/test/polyp_test_8_gt_08-2/gt',
        'frame_range': ('frame_006125', 'frame_007100'),
        'gt_range': (6126, 7101),
        'seq_name': 'polyp_test_8_gt_08-2'
    }
]

# --- 모든 시퀀스 처리 ---
for seq in sequences_info:
    process_sequence(
        src_img_folder=seq['src_img_folder'],
        src_txt_folder=seq['src_txt_folder'],
        dest_img_folder=seq['dest_img_folder'],
        dest_txt_folder=seq['dest_txt_folder'],
        frame_range=seq['frame_range'],
        gt_range=seq['gt_range'],
        seq_name=seq['seq_name']
    )
