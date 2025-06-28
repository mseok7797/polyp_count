import os
import pandas as pd
from pathlib import Path
import warnings # warnings 추가

# --- trackeval 라이브러리 import ---
try:
    import trackeval
    from trackeval import Evaluator
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics import CLEAR, Identity, HOTA # 필요시 HOTA 주석 해제
    print("Successfully imported trackeval library.")
except ImportError as e:
    print(f"ERROR: Failed to import trackeval or its components: {e}.")
    print("Please install it (`pip install trackeval`). Metrics calculation will be skipped.")
    # 라이브러리 부재 시 사용할 플레이스홀더
    Evaluator = None
    MotChallenge2DBox = None
    CLEAR = None
    Identity = None
    HOTA = None

def compute_mot_metrics(
    gt_root: str,           # GT 데이터 루트 디렉토리 (예: .../MOT17/train)
    res_root: str,          # 추적 결과 루트 디렉토리 (tracker 이름 폴더 상위, 예: ./mot_results/run_xyz)
    output_csv: str = 'mot_metrics_summary.csv', # 결과 CSV 파일 경로
    benchmark_name: str = 'MOTChallenge', # 평가 벤치마크 이름
    split_to_eval: str = 'train', # 평가할 스플릿 (GT 폴더 구조와 일치해야 함)
    metrics_list: list = None, # 계산할 메트릭 객체 리스트 (None이면 기본값 사용)
    seqmap_filename: str = None # 사용할 seqmap 파일 경로 (None이면 자동 탐색/생성 시도)
) -> pd.DataFrame:
    """
    TrackEval을 이용해 MOT 지표를 계산하고 결과를 반환/저장합니다.

    Args:
        gt_root: Ground Truth 데이터가 있는 루트 폴더 경로.
                 내부에 split_to_eval 이름의 폴더가 있고, 그 안에 시퀀스 폴더(예: MOT17-02)들이 있어야 함.
                 각 시퀀스 폴더 안에는 'gt/gt.txt'와 'seqinfo.ini'가 있어야 함.
        res_root: 추적 결과가 있는 루트 폴더 경로. 이 폴더 바로 아래에 tracker 이름의 폴더가 있고,
                  그 안에 시퀀스 이름(예: MOT17-02.txt)의 결과 파일이 있어야 함.
                  (예: res_root/my_tracker_run/MOT17-02.txt)
        output_csv: 결과를 저장할 CSV 파일의 전체 경로.
        benchmark_name: 평가에 사용할 벤치마크 이름 (데이터셋 클래스 선택에 영향).
        split_to_eval: 평가할 데이터 스플릿 이름 (예: 'train', 'test').
        metrics_list: 사용할 trackeval.metrics 객체 리스트. None이면 CLEAR, Identity 기본 사용.
        seqmap_filename: 특정 시퀀스만 평가할 때 사용할 seqmap 파일 경로. None이면 GT 폴더 내 모든 시퀀스 평가.

    Returns:
        pd.DataFrame: 계산된 메트릭 요약 결과. 실패 시 빈 DataFrame.
    """

    if Evaluator is None:
        warnings.warn("Trackeval library not available. Skipping metrics calculation.")
        return pd.DataFrame()

    gt_root_path = Path(gt_root)
    res_root_path = Path(res_root)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True) # 출력 디렉토리 생성

    print(f"\n--- Starting TrackEval Evaluation ---")
    print(f"  GT Root: {gt_root_path.resolve()}")
    print(f"  Results Root: {res_root_path.resolve()}")
    print(f"  Output CSV: {output_csv_path.resolve()}")
    print(f"  Benchmark: {benchmark_name}, Split: {split_to_eval}")
    if seqmap_filename: print(f"  Using Seqmap: {Path(seqmap_filename).resolve()}")

    # --- 1. Evaluator 설정 ---
    eval_config = Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True # 로그 줄이기
    # eval_config['OUTPUT_SUMMARY_ONLY'] = True # 요약만 출력 원할 시

    # --- 2. Dataset 설정 ---
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    dataset_config['BENCHMARK'] = benchmark_name
    dataset_config['SPLIT_TO_EVAL'] = split_to_eval
    dataset_config['GT_FOLDER'] = str(gt_root_path)      # GT 루트
    dataset_config['TRACKERS_FOLDER'] = str(res_root_path) # 결과 루트 (tracker 폴더 상위)
    # tracker 이름은 자동으로 res_root 아래의 폴더 이름으로 인식됨
    dataset_config['TRACKER_SUB_FOLDER'] = '' # TRACKERS_FOLDER 바로 아래에 결과 파일이 있는 경우
    dataset_config['OUTPUT_FOLDER'] = None # trackeval 내부 결과 저장 안 함

    # Seqmap 설정
    if seqmap_filename:
        seqmap_file_path = Path(seqmap_filename)
        if seqmap_file_path.exists():
             dataset_config['SEQMAP_FILE'] = str(seqmap_file_path)
             dataset_config['SEQ_INFO'] = None # SEQMAP_FILE 사용 시 SEQ_INFO는 None
             print(f"  Using provided seqmap file: {seqmap_file_path.resolve()}")
        else:
             warnings.warn(f"Seqmap file not found at {seqmap_filename}. Evaluating all sequences in GT/{split_to_eval}.")
             dataset_config['SEQMAP_FILE'] = None
    else:
        dataset_config['SEQMAP_FILE'] = None # 모든 시퀀스 평가
        print("  No seqmap provided. Evaluating all sequences found in GT folder.")


    # --- 3. Metrics 설정 및 리스트 생성 ---
    if metrics_list is None:
        metric_config = {'METRICS': ['CLEAR', 'Identity'], 'THRESHOLD': 0.5}
        metrics_list = [CLEAR(metric_config), Identity(metric_config)]
        # 필요시 HOTA 추가: metrics_list.append(HOTA(metric_config))
        print(f"  Using default metrics: {[m.get_name() for m in metrics_list]}")
    else:
        print(f"  Using provided metrics: {[m.get_name() for m in metrics_list]}")


    # --- 4. Evaluator 및 Dataset 인스턴스 생성 ---
    try:
        evaluator = Evaluator(eval_config)
        dataset_list = [MotChallenge2DBox(dataset_config)]
    except Exception as e:
        print(f"Error initializing TrackEval Dataset/Evaluator: {e}")
        return pd.DataFrame()

    # --- 5. 평가 실행 ---
    print("Running evaluation...")
    try:
        results, messages = evaluator.evaluate(dataset_list, metrics_list)
        print("Evaluation finished.")
    except Exception as e:
        print(f"Error during TrackEval evaluation: {e}")
        import traceback
        traceback.print_exc() # 디버깅 위해 traceback 출력
        return pd.DataFrame()

    # --- 6. 결과 처리 및 저장 ---
    if results is None:
         warnings.warn("TrackEval evaluation returned None results.")
         return pd.DataFrame()

    try:
        # 결과는 딕셔너리 형태: results[dataset_name][tracker_name]
        dataset_name = dataset_list[0].get_name() # 예: MotChallenge2DBox
        # tracker 이름은 res_root 아래 폴더 이름으로 자동 결정됨
        if not results or dataset_name not in results or not results[dataset_name]:
             warnings.warn(f"No evaluation results found for dataset '{dataset_name}'.")
             return pd.DataFrame()

        tracker_name = list(results[dataset_name].keys())[0] # 첫 번째 tracker 결과 사용
        print(f"Processing results for tracker: {tracker_name}")

        # Combined(요약) 결과 또는 개별 시퀀스 결과 추출
        if 'COMBINED_SEQ' in results[dataset_name][tracker_name]:
             summary = results[dataset_name][tracker_name]['COMBINED_SEQ']
             # metrics_list에 정의된 이름으로 결과 딕셔너리 생성
             summary_dict = {m.get_name(): summary.get(m.get_name(), None) for m in metrics_list}
             df = pd.DataFrame([summary_dict])
             df.index = [f'{tracker_name}_COMBINED'] # 인덱스 설정
        else:
             # COMBINED 없으면 개별 시퀀스 결과 취합
             warnings.warn("'COMBINED_SEQ' not found. Aggregating individual sequence results.")
             all_seq_results = []
             for seq, res in results[dataset_name][tracker_name].items():
                 seq_res_filtered = {m.get_name(): res.get(m.get_name(), None) for m in metrics_list}
                 seq_res_filtered['SEQUENCE'] = seq
                 all_seq_results.append(seq_res_filtered)
             if not all_seq_results:
                  print("Warning: No individual sequence results found either.")
                  return pd.DataFrame()
             df = pd.DataFrame(all_seq_results)
             df.set_index('SEQUENCE', inplace=True) # 시퀀스 이름을 인덱스로

        df.to_csv(output_csv_path, index=True)
        print(f"Saved MOT metrics summary to {output_csv_path}")
        print("\n--- Evaluation Summary ---")
        # 콘솔 출력 보기 좋게 포맷팅 (옵션)
        try: # pandas 스타일링 옵션
             with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                 print(df)
        except Exception: # 실패 시 기본 출력
            print(df)

    except Exception as e:
        print(f"Error processing results or saving CSV: {e}")
        print("Raw results structure:", type(results)) # 결과 구조 디버깅
        return pd.DataFrame()

    return df