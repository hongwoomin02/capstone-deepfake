"""
face_pipeline.py — Fact-Trace AI 영상 전처리 파이프라인 (v2)

캡스톤1 v1 (MediaPipe 기반) 파이프라인을 §6 인터페이스 명세에 맞춰 재작성한 모듈.
검출기는 RetinaFace로 교체 (14주차 학습 분포 측정값과의 일관성을 위해).

주요 함수
---------
- process_video(video_path, ...)  : 영상 → face-crop batch + 메타데이터
- process_image(image_path, ...)  : 단일 이미지 → face-crop dict (없으면 None)
- _detect_and_crop(rgb, ...)      : 공유 내부 함수 (RetinaFace 검출 + crop + 메타)

설치 (Colab 기준)
----------------
    pip install retina-face opencv-python moviepy torch torchvision pillow numpy

작성자: 팀원 B (캡스톤2 진입 전 face_pipeline 통일 작업)
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# RetinaFace (serengil/retina-face). 명세서 §2: insightface, facenet-pytorch 모두 환경 충돌로 제외.
from retinaface import RetinaFace

try:
    from moviepy import VideoFileClip
except ImportError:  # moviepy 구버전 호환
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None


# =========================
# 학습 환경과 일치하는 정규화 (명세서 §2 EVAL_TF)
# =========================
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# =========================
# 내부 공유 함수: 검출 + crop + 메타
# =========================
def _detect_and_crop(
    rgb_image: np.ndarray,
    det_threshold: float = 0.5,
    margin: float = 0.15,
    multi_face: str = 'largest',
) -> Optional[dict]:
    """
    RGB 이미지에서 RetinaFace로 얼굴을 검출하고 (가장 큰 얼굴 1개) margin 적용 후
    224x224 crop과 메타데이터를 반환한다.

    Parameters
    ----------
    rgb_image : np.ndarray
        (H, W, 3) RGB 이미지. cv2.imread의 BGR을 미리 변환해서 넘길 것.
    det_threshold : float
        RetinaFace 검출 confidence threshold. 14주차 통일값 0.5.
    margin : float
        bbox 양쪽에 추가할 padding 비율. 0.0 / 0.15 / 0.30 등.
    multi_face : str
        'largest'만 현재 지원. 'all'은 추후 확장.

    Returns
    -------
    dict 또는 None
        검출 성공 시:
            {
                'crop_224':     (224, 224, 3) np.uint8 RGB, normalize 전,
                'bbox':         [x1, y1, x2, y2] padding 적용 후,
                'det_bbox':     [x1, y1, x2, y2] RetinaFace 원본 (padding 전),
                'det_score':    float,
                'area_ratio':   float,  padding 후 bbox 면적 / 이미지 면적,
                'det_area_ratio': float,  padding 전 bbox 면적 / 이미지 면적 (14주차 비교용),
                'cy':           float,  bbox 중심 y / 이미지 높이,
                'aspect':       float,  bbox w / bbox h,
            }
        검출 실패 시: None
    """
    if rgb_image is None or rgb_image.size == 0:
        return None

    H, W, _ = rgb_image.shape

    # RetinaFace.detect_faces 반환은 dict (성공) 또는 빈 tuple (실패).
    # 명세서 §2 경고: isinstance(dets, dict) 체크 필수.
    dets = RetinaFace.detect_faces(rgb_image, threshold=det_threshold)
    if not isinstance(dets, dict) or len(dets) == 0:
        return None

    # 가장 큰 얼굴 1개 선택 (padding 전 면적 기준)
    best_key = None
    best_area = 0
    for k, v in dets.items():
        x1, y1, x2, y2 = v['facial_area']
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        if area > best_area:
            best_area = area
            best_key = k

    if best_key is None or best_area == 0:
        return None

    det = dets[best_key]
    dx1, dy1, dx2, dy2 = det['facial_area']
    det_score = float(det.get('score', 0.0))

    # padding 적용
    bw = dx2 - dx1
    bh = dy2 - dy1
    pad_w = int(bw * margin)
    pad_h = int(bh * margin)
    x1 = max(0, int(dx1) - pad_w)
    y1 = max(0, int(dy1) - pad_h)
    x2 = min(W, int(dx2) + pad_w)
    y2 = min(H, int(dy2) + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_rgb = rgb_image[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None

    crop_224 = cv2.resize(crop_rgb, (224, 224))

    # 메타 (padding 전/후 둘 다 보관 — 14주차 비교용)
    padded_area = (x2 - x1) * (y2 - y1)
    det_area = bw * bh
    frame_area = H * W

    return {
        'crop_224': crop_224,
        'bbox': [int(x1), int(y1), int(x2), int(y2)],
        'det_bbox': [int(dx1), int(dy1), int(dx2), int(dy2)],
        'det_score': det_score,
        'area_ratio': float(padded_area) / float(frame_area),
        'det_area_ratio': float(det_area) / float(frame_area),
        'cy': float((y1 + y2) / 2.0) / float(H),
        'aspect': float(x2 - x1) / float(max(1, y2 - y1)),
    }


# =========================
# 단일 이미지용
# =========================
def process_image(
    image_path: str,
    det_threshold: float = 0.5,
    margin: float = 0.15,
    multi_face: str = 'largest',
) -> Optional[dict]:
    """
    단일 이미지(ff_c23 raw frame, face-crop 데이터셋 등) 1장을 처리.

    Returns
    -------
    dict 또는 None
        process_video의 crops 리스트 안 dict와 동일 형식.
        검출 실패 시 None. (frame_idx는 없음)
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 읽지 못함: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _detect_and_crop(rgb, det_threshold=det_threshold,
                            margin=margin, multi_face=multi_face)


# =========================
# 영상용 (메인)
# =========================
def process_video(
    video_path: str,
    det_threshold: float = 0.5,
    margin: float = 0.15,
    frame_sampling: str = 'uniform',
    n_frames: int = 100,
    fps_target: float = 1.0,
    multi_face: str = 'largest',
    extract_audio: bool = True,
    save_outputs: bool = True,
    output_dir: str = 'output',
) -> dict:
    """
    영상 1개를 처리해 face-crop batch + tensor + 메타데이터를 반환한다.
    명세서 §6 인터페이스 명세를 100% 따른다.

    Parameters
    ----------
    video_path : str
        입력 영상 경로.
    det_threshold : float
        RetinaFace confidence threshold. 14주차 통일값 0.5 (변경 비권장).
    margin : float
        face bbox에 추가할 padding 비율. 0.0 / 0.15 / 0.30 등.
    frame_sampling : {'uniform', 'fps_based', 'all'}
        - 'uniform'  : n_frames 장을 영상 전체 길이에 균등 분포로 선택.
        - 'fps_based': 영상 fps를 기준으로 초당 fps_target 장씩 선택.
        - 'all'      : 모든 frame 사용 (긴 영상은 메모리/시간 주의).
    n_frames : int
        'uniform' 모드에서 균등 추출할 frame 수.
    fps_target : float
        'fps_based' 모드에서 초당 추출할 frame 수.
    multi_face : str
        'largest'만 현재 지원.
    extract_audio : bool
        True면 MoviePy로 오디오 추출.
    save_outputs : bool
        True면 output_dir에 crops/, vis/, audio, metadata.json 저장.
    output_dir : str
        save_outputs=True일 때 저장 디렉토리.

    Returns
    -------
    dict
        명세서 §6에 정의된 반환 구조와 동일.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일 없음: {video_path}")

    # ----- 1) 영상 메타만 빠르게 읽기 (frame은 안 로드, OOM 방지) -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열지 못함: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if total_frames_meta <= 0:
        raise ValueError(
            f"영상 메타에서 frame 수를 읽지 못함 (CAP_PROP_FRAME_COUNT=0). "
            f"코덱 호환성 문제 가능성. ffmpeg로 재인코딩 후 재시도 권장: {video_path}"
        )

    # 우선 메타 기반 추정. 실제 read 끝나면 actual_total로 갱신.
    total_frames = total_frames_meta
    video_duration = float(total_frames) / float(fps) if fps > 0 else 0.0

    # ----- 2) frame 샘플링 indices 결정 -----
    sampled_indices = _sample_frame_indices(
        total_frames=total_frames,
        fps=fps,
        strategy=frame_sampling,
        n_frames=n_frames,
        fps_target=fps_target,
    )
    sampled_set = set(sampled_indices)

    # ----- 3) 출력 디렉토리 + key frame 인덱스 -----
    crops_dir = os.path.join(output_dir, "crops")
    vis_dir = os.path.join(output_dir, "vis")
    keyframes_dir = os.path.join(output_dir, "keyframes")
    audio_path_out = os.path.join(output_dir, "extracted_audio.wav")
    metadata_path = os.path.join(output_dir, "metadata.json")

    if save_outputs:
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(keyframes_dir, exist_ok=True)
        key_frame_indices = [0, total_frames // 2] if total_frames >= 2 else [0]
    else:
        key_frame_indices = []
    key_set = set(key_frame_indices)

    # ----- 4) streaming read + 즉시 처리 (frame 1장만 메모리에) -----
    processed_results: dict = {}   # frame_idx -> info
    failed_frame_indices: List[int] = []
    key_frames_saved = 0

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    actual_total = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            actual_total = frame_idx + 1

            # key frame 저장 (이 frame이 key 대상이면)
            if frame_idx in key_set:
                kf_path = os.path.join(keyframes_dir, f"keyframe_{key_frames_saved + 1}.jpg")
                cv2.imwrite(kf_path, frame_bgr)
                key_frames_saved += 1

            # sampled 대상이면 검출
            if frame_idx in sampled_set:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                info = _detect_and_crop(
                    rgb_image=frame_rgb,
                    det_threshold=det_threshold,
                    margin=margin,
                    multi_face=multi_face,
                )
                if info is None:
                    # 명세서 §5 6번: skip + 실패 frame_idx 기록
                    failed_frame_indices.append(int(frame_idx))
                else:
                    info['frame_idx'] = int(frame_idx)
                    processed_results[frame_idx] = info

            frame_idx += 1
            # frame_bgr/frame_rgb는 다음 iteration에서 덮어써지므로 자동 GC됨
    finally:
        cap.release()

    if actual_total == 0:
        raise ValueError(f"영상에서 frame을 한 장도 읽지 못함: {video_path}")

    # 실제 read한 frame 수가 메타와 다르면 실제값 신뢰
    total_frames = actual_total
    if fps > 0:
        video_duration = float(total_frames) / float(fps)

    # ----- 5) sampled_indices 순서대로 crops/vis_imgs/tensors 구성 -----
    crops: List[dict] = []
    vis_imgs: List[np.ndarray] = []
    tensors: List[torch.Tensor] = []
    failed_set = set(failed_frame_indices)

    for sampled_pos, idx in enumerate(sampled_indices):
        if idx not in processed_results:
            # 메타 부정확으로 영상 끝 넘어간 idx, 또는 검출 실패 (이미 failed에 들어감)
            if idx not in failed_set:
                failed_frame_indices.append(int(idx))
                failed_set.add(int(idx))
            continue

        info = processed_results[idx]
        crops.append(info)
        vis_imgs.append(info['crop_224'])

        pil = Image.fromarray(info['crop_224'])
        tensors.append(EVAL_TF(pil))

        if save_outputs:
            crop_bgr = cv2.cvtColor(info['crop_224'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(crops_dir, f"crop_{sampled_pos:04d}.jpg"), crop_bgr)
            cv2.imwrite(os.path.join(vis_dir, f"vis_{sampled_pos:04d}.jpg"), crop_bgr)

    detected_frames = len(crops)
    detection_rate = (detected_frames / len(sampled_indices)) if sampled_indices else 0.0

    # ----- 5) tensor batch -----
    if tensors:
        tensor_batch = torch.stack(tensors, dim=0)  # (N, 3, 224, 224)
    else:
        tensor_batch = torch.zeros((0, 3, 224, 224), dtype=torch.float32)

    # ----- 6) 분포 통계 (학습 분포 비교용) -----
    if crops:
        area_arr = np.array([c['area_ratio'] for c in crops], dtype=np.float64)
        det_area_arr = np.array([c['det_area_ratio'] for c in crops], dtype=np.float64)
        cy_arr = np.array([c['cy'] for c in crops], dtype=np.float64)
        area_ratio_mean = float(area_arr.mean())
        area_ratio_std = float(area_arr.std())
        det_area_ratio_mean = float(det_area_arr.mean())
        det_area_ratio_std = float(det_area_arr.std())
        cy_mean = float(cy_arr.mean())
    else:
        area_ratio_mean = 0.0
        area_ratio_std = 0.0
        det_area_ratio_mean = 0.0
        det_area_ratio_std = 0.0
        cy_mean = 0.0

    # ----- 7) 오디오 추출 -----
    audio_path: Optional[str] = None
    if extract_audio and save_outputs:
        if VideoFileClip is None:
            print("[face_pipeline] moviepy import 실패로 오디오 추출 생략")
        else:
            try:
                clip = VideoFileClip(video_path)
                if clip.audio is not None:
                    clip.audio.write_audiofile(audio_path_out, logger=None)
                    audio_path = audio_path_out
                else:
                    print("[face_pipeline] 영상에 오디오 트랙 없음")
                clip.close()
            except Exception as e:
                print(f"[face_pipeline] 오디오 추출 실패: {e}")

    # ----- 8) 메타데이터 -----
    meta = {
        'video_path': video_path,
        'video_duration': video_duration,
        'total_frames': total_frames,
        'total_frames_meta': total_frames_meta,  # cv2 메타가 알려준 frame 수 (참고)
        'fps': fps,
        'sampled_frame_indices': [int(i) for i in sampled_indices],
        'detected_frames': detected_frames,
        'failed_frame_indices': failed_frame_indices,
        'detection_rate': detection_rate,
        'area_ratio_mean': area_ratio_mean,           # padding 후 (§6 명세 기준)
        'area_ratio_std': area_ratio_std,
        'det_area_ratio_mean': det_area_ratio_mean,   # padding 전 (14주차 학습 분포 직접 비교용)
        'det_area_ratio_std': det_area_ratio_std,
        'cy_mean': cy_mean,
        'key_frame_indices': key_frame_indices,
        'config': {
            'det_threshold': det_threshold,
            'margin': margin,
            'frame_sampling': frame_sampling,
            'n_frames': n_frames,
            'fps_target': fps_target,
            'multi_face': multi_face,
        },
    }

    if save_outputs:
        # crop_224는 ndarray라 JSON 직렬화 불가 → 메타에서 제외한 사본 저장
        meta_for_json = dict(meta)
        meta_for_json['crops'] = [
            {k: v for k, v in c.items() if k != 'crop_224'}
            for c in crops
        ]
        meta_for_json['audio_path'] = audio_path
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta_for_json, f, ensure_ascii=False, indent=2)

    return {
        'crops': crops,
        'tensor_batch': tensor_batch,
        'vis_imgs': vis_imgs,
        'audio_path': audio_path,
        'meta': meta,
    }


# =========================
# 내부: frame 샘플링 전략
# =========================
def _sample_frame_indices(
    total_frames: int,
    fps: float,
    strategy: str = 'uniform',
    n_frames: int = 100,
    fps_target: float = 1.0,
) -> List[int]:
    """
    명세서 §5 7번: 영상 길이 따라 sample 수 변동하는 v1 방식 (step=len//50) 폐기.
    'uniform'으로 균등 N장 추출이 기본. 영상이 짧으면 총 frame 수 이하로 자동 클램프.
    """
    if total_frames <= 0:
        return []

    if strategy == 'all':
        return list(range(total_frames))

    if strategy == 'uniform':
        n = min(n_frames, total_frames)
        if n == 1:
            return [total_frames // 2]
        # 0 ~ total_frames-1 사이를 n등분 (양 끝 포함)
        return [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]

    if strategy == 'fps_based':
        if fps <= 0:
            # fps를 알 수 없으면 uniform fallback
            return _sample_frame_indices(total_frames, fps, 'uniform', n_frames, fps_target)
        step = max(1, int(round(fps / max(fps_target, 1e-6))))
        return list(range(0, total_frames, step))

    raise ValueError(f"알 수 없는 frame_sampling: {strategy}")


# =========================
# CLI 빠른 동작 확인용
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="face_pipeline 빠른 실행 (test용)")
    parser.add_argument("video", type=str, help="입력 영상 경로 (예: test.mp4)")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    result = process_video(
        video_path=args.video,
        margin=args.margin,
        n_frames=args.n_frames,
        extract_audio=not args.no_audio,
        output_dir=args.output_dir,
    )
    m = result['meta']
    print("\n=== 처리 완료 ===")
    print(f"video_duration       : {m['video_duration']:.2f} s")
    print(f"total_frames         : {m['total_frames']}")
    print(f"sampled              : {len(m['sampled_frame_indices'])}")
    print(f"detected             : {m['detected_frames']}")
    print(f"detection_rate       : {m['detection_rate']:.1%}")
    print(f"area_ratio (padded)  : {m['area_ratio_mean']:.4f}  (std {m['area_ratio_std']:.4f})")
    print(f"det_area_ratio (raw) : {m['det_area_ratio_mean']:.4f}  (std {m['det_area_ratio_std']:.4f})")
    print(f"cy_mean              : {m['cy_mean']:.4f}")
    print(f"tensor_batch shape   : {tuple(result['tensor_batch'].shape)}")
    print(f"audio_path           : {result['audio_path']}")
