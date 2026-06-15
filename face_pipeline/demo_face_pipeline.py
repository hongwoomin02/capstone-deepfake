"""
demo_face_pipeline.py
---------------------
face_pipeline.py 동작 확인 및 §7 A 산출물용 데모.

내용:
1) test.mp4 1개로 process_video 기본 호출
2) margin 0.0 / 0.15 / 0.30 비교 (§7 A 완료 기준 3)
3) v1, v2 모델 forward 예시 (Drive 체크포인트 경로는 명세서 §2 그대로)

Colab에서 사용 시
-----------------
    # !pip install retina-face opencv-python moviepy torch torchvision pillow numpy timm
    # from google.colab import drive; drive.mount('/content/drive')
    # 이 파일과 face_pipeline.py를 같은 폴더에 두고 import

본 데모는 face_pipeline.py와 같은 디렉토리에 둘 것.
"""

import os
import json
import torch
from face_pipeline import process_video, EVAL_TF  # noqa: F401


# =========================
# 경로 설정 (명세서 §2와 100% 일치)
# =========================
# 로컬 또는 Colab 어디서든 동작하도록 video_path는 인자로
TEST_VIDEO = "test.mp4"

# Drive 마운트 후 사용 시 (명세서 §2)
PROJECT_ROOT = "/content/drive/MyDrive/capstone_deepfake/capstone_deepfake"
V1_CKPT = f"{PROJECT_ROOT}/checkpoints/main_unified/best_model_unified.pt"
V2_CKPT = f"{PROJECT_ROOT}/checkpoints/main_unified_v2/best_model.pt"


# =========================
# 1) 기본 호출 (margin=0.15)
# =========================
def demo_basic(video_path: str = TEST_VIDEO, output_dir: str = "output_basic"):
    print("\n[1] 기본 호출 (margin=0.15, n_frames=100, uniform)")
    result = process_video(
        video_path=video_path,
        margin=0.15,
        n_frames=100,
        frame_sampling='uniform',
        extract_audio=True,
        save_outputs=True,
        output_dir=output_dir,
    )
    m = result['meta']
    print(f"  영상 길이               : {m['video_duration']:.2f} s ({m['total_frames']} frame, {m['fps']:.2f} fps)")
    print(f"  샘플링 frame            : {len(m['sampled_frame_indices'])}")
    print(f"  검출 성공               : {m['detected_frames']}")
    print(f"  검출률                  : {m['detection_rate']:.1%}")
    print(f"  실패 frame 수           : {len(m['failed_frame_indices'])}")
    print(f"  area_ratio (padded)     : mean {m['area_ratio_mean']:.4f}, std {m['area_ratio_std']:.4f}")
    print(f"  det_area_ratio (raw)    : mean {m['det_area_ratio_mean']:.4f}, std {m['det_area_ratio_std']:.4f}")
    print(f"  cy_mean                 : {m['cy_mean']:.4f}")
    print(f"  tensor_batch shape      : {tuple(result['tensor_batch'].shape)}")
    print(f"  audio_path              : {result['audio_path']}")
    return result


# =========================
# 2) margin 비교 (§7 A 완료 기준 3)
# =========================
def demo_margin_sweep(video_path: str = TEST_VIDEO):
    print("\n[2] margin 비교 (0.0 / 0.15 / 0.30)")
    print("    학습 분포(14주차 §3 표): ff_c23 0.07 / celebdf 0.27 / dfdc 0.34 /")
    print("                              ciplab 0.41 / sd 0.37 / 140k 0.41 (모두 RetinaFace 기준)")

    rows = []
    for margin in [0.0, 0.15, 0.30]:
        result = process_video(
            video_path=video_path,
            margin=margin,
            n_frames=100,
            frame_sampling='uniform',
            extract_audio=False,
            save_outputs=False,  # 메타만 수집
        )
        m = result['meta']
        rows.append({
            'margin': margin,
            'detection_rate': m['detection_rate'],
            'area_ratio_mean': m['area_ratio_mean'],
            'area_ratio_std': m['area_ratio_std'],
            'det_area_ratio_mean': m['det_area_ratio_mean'],
            'cy_mean': m['cy_mean'],
            'detected_frames': m['detected_frames'],
            'n_sampled': len(m['sampled_frame_indices']),
        })

    print(f"\n  {'margin':>6} | {'det_rate':>9} | {'area(pad)':>10} | {'area(raw)':>10} | {'cy':>6} | {'N_det':>6}")
    print("  " + "-" * 65)
    for r in rows:
        print(f"  {r['margin']:>6.2f} | {r['detection_rate']:>9.1%} | "
              f"{r['area_ratio_mean']:>10.4f} | {r['det_area_ratio_mean']:>10.4f} | "
              f"{r['cy_mean']:>6.3f} | {r['detected_frames']:>6d}")

    # JSON으로도 저장 (보고서 첨부용)
    with open("margin_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n  → margin_sweep.json 저장")
    return rows


# =========================
# 3) v1, v2 모델 forward 예시 (Colab + Drive 환경 전제)
# =========================
def demo_model_forward(video_path: str = TEST_VIDEO, margin: float = 0.15):
    """
    Drive 체크포인트로 v1, v2 모델 로드 후 video-level P(Real) 측정.
    명세서 §2와 §6 사용 예시를 그대로 따름.

    주의: Colab에서 Drive 마운트 + GPU 환경 + timm 설치 필요.
    """
    try:
        import timm
    except ImportError:
        print("[3] timm 미설치 — `pip install timm` 후 재실행")
        return None

    if not (os.path.exists(V1_CKPT) and os.path.exists(V2_CKPT)):
        print(f"[3] 체크포인트 없음 (Drive 마운트 확인). v1={V1_CKPT}, v2={V2_CKPT}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3] 모델 forward 예시 (device={device}, margin={margin})")

    result = process_video(
        video_path=video_path,
        margin=margin,
        n_frames=100,
        frame_sampling='uniform',
        extract_audio=False,
        save_outputs=False,
    )
    batch = result['tensor_batch'].to(device)
    print(f"  tensor_batch shape: {tuple(batch.shape)}")

    out = {}
    for tag, ckpt_path in [('v1', V1_CKPT), ('v2', V2_CKPT)]:
        model = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=False, num_classes=2,
        )
        state = torch.load(ckpt_path, map_location=device)
        # 체크포인트가 dict 안에 'model' 또는 'state_dict' 키로 저장된 경우 대응
        if isinstance(state, dict):
            for k in ['model', 'state_dict', 'model_state_dict']:
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        with torch.no_grad():
            logits = model(batch)
            probs_real = torch.softmax(logits, dim=1)[:, 1]
            video_p_real_mean = float(probs_real.mean().item())
            video_p_real_median = float(probs_real.median().item())
        out[tag] = {
            'p_real_mean': video_p_real_mean,
            'p_real_median': video_p_real_median,
            'frames_used': int(batch.shape[0]),
        }
        print(f"  [{tag}] P(Real) frame 평균 {video_p_real_mean:.4f}, "
              f"median {video_p_real_median:.4f}  (N={int(batch.shape[0])})")

    return out


# =========================
# 메인
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, nargs="?", default=TEST_VIDEO,
                        help="입력 영상 경로 (기본: test.mp4)")
    parser.add_argument("--with_model", action="store_true",
                        help="v1/v2 모델 forward까지 실행 (Drive 마운트 + GPU 필요)")
    args = parser.parse_args()

    demo_basic(args.video)
    demo_margin_sweep(args.video)
    if args.with_model:
        demo_model_forward(args.video)
