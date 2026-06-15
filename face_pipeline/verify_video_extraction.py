"""
verify_video_extraction.py
--------------------------
소규모 검증: 원본 영상에서 process_video로 추출한 분포가
기존 face-crop 분포 / 학습 목표 분포와 어떻게 다른지 측정.

목적 (팀원 B 캡스톤2 사전 검증)
- "이미지(face-crop)로 하니 성능이 안 좋다" → 영상 직접 추출이 분포 갭을 줄이는가?
- 전체 데이터셋 재추출 + 재학습 결정 전, 소규모로 먼저 확인

사용 전제
- Colab + Drive 마운트 + face_pipeline.py가 sys.path에 있음
- 검증용 영상 몇 개를 한 폴더에 모아둘 것 (CelebDF 권장 — 14주차 face-crop 0.27 단일 비교)

CelebDF를 권장하는 이유
- CelebDF face-crop 학습 분포 = 0.27 (단일)
- FF++는 raw frame(0.07)과 Kaggle face-crop(0.41) 두 분포가 섞여 비교 해석 복잡

Colab 실행 예시
---------------
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/capstone_deepfake/capstone_deepfake/face_pipeline')
    from verify_video_extraction import verify_folder

    # 영상만 검증 (분포 측정)
    verify_folder(
        video_dir='/content/drive/MyDrive/.../celebdf_videos_sample',
        dataset_name='celebdf',
        n_frames=30,
    )

    # 모델 성능까지 검증하려면 with_model=True (Drive 체크포인트 + GPU 필요)
    verify_folder(..., with_model=True, expected_label=1)  # CelebDF Real이면 label=1
"""

import glob
import json
import os
from typing import Optional

import numpy as np

from face_pipeline import process_video


# 14주차 §3 학습 데이터 face-crop 분포 (RetinaFace, threshold=0.5)
TRAIN_DIST = {
    'ff_c23_raw':        {'area_ratio': 0.07, 'cy': 0.35, 'note': 'Drive raw frame (실제 학습 분포)'},
    'ff_c23_facecrop':   {'area_ratio': 0.41, 'cy': 0.50, 'note': 'Kaggle face-crop'},
    'celebdf':           {'area_ratio': 0.27, 'cy': 0.46, 'note': ''},
    'dfdc':              {'area_ratio': 0.34, 'cy': 0.51, 'note': '검출 성공률 17.1%'},
    'ciplab':            {'area_ratio': 0.41, 'cy': 0.58, 'note': 'GAN 생성 (영상 원본 없음)'},
    'sd':                {'area_ratio': 0.37, 'cy': 0.44, 'note': 'GAN 생성 (영상 원본 없음)'},
    '140k':              {'area_ratio': 0.41, 'cy': 0.55, 'note': 'GAN 생성 (영상 원본 없음)'},
}

V1_CKPT = "/content/drive/MyDrive/capstone_deepfake/capstone_deepfake/checkpoints/main_unified/best_model_unified.pt"
V2_CKPT = "/content/drive/MyDrive/capstone_deepfake/capstone_deepfake/checkpoints/main_unified_v2/best_model.pt"


def verify_folder(
    video_dir: str,
    dataset_name: str = 'celebdf',
    n_frames: int = 30,
    margin: float = 0.15,
    video_exts=('*.mp4', '*.avi', '*.mov', '*.mkv'),
    with_model: bool = False,
    expected_label: Optional[int] = None,   # 0=Fake, 1=Real (성능 검증 시)
    save_json: str = '/content/video_extraction_verify.json',
):
    """
    video_dir 안 모든 영상을 process_video로 추출 → 분포 측정 → 학습 분포와 비교.

    Parameters
    ----------
    video_dir : 검증용 영상이 들어있는 폴더
    dataset_name : 'celebdf' / 'ff_c23_facecrop' / 'ff_c23_raw' / 'dfdc' 중 하나
                   (TRAIN_DIST의 비교 기준 선택)
    n_frames : 영상당 추출 frame 수 (소규모 검증이라 30이면 충분)
    margin : padding 비율
    with_model : True면 v1/v2 모델 forward까지 (성능 검증)
    expected_label : with_model 시 정답 라벨 (0=Fake, 1=Real)
    """
    # 영상 파일 수집
    video_paths = []
    for ext in video_exts:
        video_paths.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
    video_paths = sorted(set(video_paths))

    if not video_paths:
        print(f"[검증] 영상 파일 없음: {video_dir}")
        print(f"       찾은 확장자: {video_exts}")
        return None

    print(f"[검증] {len(video_paths)}개 영상 발견. dataset_name={dataset_name}, n_frames={n_frames}, margin={margin}")
    print("-" * 70)

    per_video = []
    all_area_raw = []
    all_cy = []
    all_det_rate = []

    for i, vp in enumerate(video_paths):
        try:
            r = process_video(
                video_path=vp,
                margin=margin,
                n_frames=n_frames,
                frame_sampling='uniform',
                extract_audio=False,
                save_outputs=False,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(video_paths)}] {os.path.basename(vp)} — 처리 실패: {e}")
            continue

        m = r['meta']
        per_video.append({
            'video': os.path.basename(vp),
            'detection_rate': m['detection_rate'],
            'det_area_ratio_mean': m['det_area_ratio_mean'],
            'area_ratio_padded_mean': m['area_ratio_mean'],
            'cy_mean': m['cy_mean'],
            'detected_frames': m['detected_frames'],
        })
        all_area_raw.append(m['det_area_ratio_mean'])
        all_cy.append(m['cy_mean'])
        all_det_rate.append(m['detection_rate'])
        print(f"  [{i+1}/{len(video_paths)}] {os.path.basename(vp):40s} "
              f"det {m['detection_rate']:.0%}  area_raw {m['det_area_ratio_mean']:.4f}  cy {m['cy_mean']:.3f}")

    if not all_area_raw:
        print("[검증] 처리 성공한 영상이 없음")
        return None

    # 집계
    agg_area = float(np.mean(all_area_raw))
    agg_area_std = float(np.std(all_area_raw))
    agg_cy = float(np.mean(all_cy))
    agg_det = float(np.mean(all_det_rate))

    print("-" * 70)
    print(f"[영상 추출 집계] {len(all_area_raw)}개 영상")
    print(f"  검출률 평균        : {agg_det:.1%}")
    print(f"  area_ratio (raw)   : {agg_area:.4f} ± {agg_area_std:.4f}")
    print(f"  cy                 : {agg_cy:.4f}")

    # 학습 분포와 비교
    print("\n[학습 분포 비교]")
    if dataset_name in TRAIN_DIST:
        ref = TRAIN_DIST[dataset_name]
        print(f"  {dataset_name} 기존 face-crop 학습 분포: area_ratio {ref['area_ratio']}, cy {ref['cy']}  {ref['note']}")
        print(f"  → 영상 추출 area_ratio {agg_area:.4f} vs 기존 {ref['area_ratio']}  (차이 {agg_area - ref['area_ratio']:+.4f})")
        print(f"  → 영상 추출 cy {agg_cy:.4f} vs 기존 {ref['cy']}  (차이 {agg_cy - ref['cy']:+.4f})")
    else:
        print(f"  주의: dataset_name='{dataset_name}'이 TRAIN_DIST에 없음. 직접 비교 필요")
        print(f"  사용 가능: {list(TRAIN_DIST.keys())}")

    print("\n[전체 학습 분포 컨텍스트]")
    print(f"  {'출처':>18} | {'area_ratio':>10} | {'cy':>5}")
    print(f"  {'영상 추출 (본 검증)':>18} | {agg_area:>10.4f} | {agg_cy:>5.3f}")
    for k, v in TRAIN_DIST.items():
        print(f"  {k:>18} | {v['area_ratio']:>10.2f} | {v['cy']:>5.2f}")

    result = {
        'dataset_name': dataset_name,
        'n_videos': len(all_area_raw),
        'n_frames_per_video': n_frames,
        'margin': margin,
        'aggregate': {
            'detection_rate': agg_det,
            'area_ratio_raw_mean': agg_area,
            'area_ratio_raw_std': agg_area_std,
            'cy_mean': agg_cy,
        },
        'per_video': per_video,
        'train_dist_ref': TRAIN_DIST.get(dataset_name),
    }

    # 모델 성능 검증 (옵션)
    if with_model:
        model_result = _verify_with_model(video_paths, n_frames, margin, expected_label)
        result['model'] = model_result

    if save_json:
        with open(save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n→ {save_json} 저장")

    return result


def _verify_with_model(video_paths, n_frames, margin, expected_label):
    """v1, v2 모델 forward로 영상 추출 입력의 성능 측정."""
    try:
        import torch
        import timm
    except ImportError:
        print("\n[모델 검증] torch/timm 미설치 — 생략")
        return None

    if not (os.path.exists(V1_CKPT) and os.path.exists(V2_CKPT)):
        print(f"\n[모델 검증] 체크포인트 없음 — 생략")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[모델 검증] device={device}, expected_label={expected_label} (0=Fake, 1=Real)")

    models = {}
    for tag, ckpt in [('v1', V1_CKPT), ('v2', V2_CKPT)]:
        mdl = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k',
                                pretrained=False, num_classes=2)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict):
            for kk in ['model', 'state_dict', 'model_state_dict']:
                if kk in state and isinstance(state[kk], dict):
                    state = state[kk]
                    break
        mdl.load_state_dict(state, strict=False)
        mdl.eval().to(device)
        models[tag] = mdl

    out = {}
    for tag, mdl in models.items():
        p_reals = []
        for vp in video_paths:
            try:
                r = process_video(video_path=vp, margin=margin, n_frames=n_frames,
                                  extract_audio=False, save_outputs=False)
                if r['tensor_batch'].shape[0] == 0:
                    continue
                with torch.no_grad():
                    probs = torch.softmax(mdl(r['tensor_batch'].to(device)), dim=1)[:, 1]
                    p_reals.append(float(probs.mean().item()))
            except Exception:
                continue

        if not p_reals:
            continue
        p_real_arr = np.array(p_reals)
        out[tag] = {
            'video_p_real_mean': float(p_real_arr.mean()),
            'video_p_real_std': float(p_real_arr.std()),
            'n_videos': len(p_reals),
        }
        line = f"  [{tag}] video P(Real) 평균 {p_real_arr.mean():.4f} ± {p_real_arr.std():.4f} (N={len(p_reals)})"
        if expected_label is not None:
            # threshold 0.5 기준 정확도 (참고용 — 실제로는 13주차 threshold 적용 권장)
            preds = (p_real_arr >= 0.5).astype(int)
            acc = float((preds == expected_label).mean())
            line += f"  | label={expected_label} 기준 acc {acc:.1%} (t=0.5 참고용)"
        print(line)

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=str, help="검증용 영상 폴더")
    parser.add_argument("--dataset", type=str, default="celebdf")
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--with_model", action="store_true")
    parser.add_argument("--label", type=int, default=None, help="0=Fake, 1=Real")
    args = parser.parse_args()

    verify_folder(
        video_dir=args.video_dir,
        dataset_name=args.dataset,
        n_frames=args.n_frames,
        margin=args.margin,
        with_model=args.with_model,
        expected_label=args.label,
    )
