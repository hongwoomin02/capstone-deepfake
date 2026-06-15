# face_pipeline.py — 캡스톤2 진입 전 영상 전처리 통합 모듈

캡스톤1에서 작성한 MediaPipe 기반 영상 전처리 코드를 §6 인터페이스 명세에 맞춰 모듈로 재작성한 것이다. 검출기는 RetinaFace로 교체했다. 본 모듈은 작업 5(분포갭 측정), 작업 6(Grad-CAM), 실서비스 backend 모두에서 동일하게 import해서 쓰는 것을 전제로 설계되었다.

## 변경 요약

§5에 정리된 7가지 수정 사항을 모두 반영했다.

| 항목 | v1 (캡스톤1) | v2 (본 모듈) |
|---|---|---|
| 1. `continue흠` SyntaxError | 있음 | 제거 (정상 `continue`) |
| 2. 얼굴 검출기 | MediaPipe (`mp.solutions.face_detection`) | RetinaFace (serengil/retina-face), threshold 0.5 |
| 3. 코드 구조 | top-level 스크립트 | `process_video` / `process_image` / `_detect_and_crop` 함수 모듈 |
| 4. padding | `padding_ratio=0.15` 하드코딩 | `margin` 인자 (0.0 / 0.15 / 0.30 등) |
| 5. tensor 반환 | 첫 검출 1장 (1, 3, 224, 224) | batch (N, 3, 224, 224), 검출 성공한 모든 frame |
| 6. 검출 실패 처리 | 단순 skip | skip + `failed_frame_indices`에 frame_idx 기록 |
| 7. frame 샘플링 | `step = len(frames)//50`, 최대 10장 | `uniform`(N 고정) / `fps_based` / `all` 중 선택 |

검출기를 RetinaFace로 교체한 이유는 14주차 학습 데이터 6 출처 face-crop 분포 측정(area_ratio 0.07~0.41)이 모두 RetinaFace 기준이기 때문이다. 다른 검출기를 쓰면 학습 분포와 실서비스 영상 분포를 직접 비교할 수 없어 작업 5(분포갭 측정)의 측정값 일관성이 깨진다.

## 유지한 점

v1에서 잘 짜여 있던 부분은 그대로 살렸다. 전체 흐름(영상 → frame → 검출 → padding → BGR/RGB → 224 resize → tensor), vis_img를 정규화 전 RGB로 별도 보관(Grad-CAM overlay용), output 디렉토리 구조(crops/ keyframes/ extracted_audio.wav metadata.json), MoviePy 오디오 추출, key frame 별도 저장(시작 + 중간) 모두 동일하게 동작한다.

## 파일

- `face_pipeline.py` — 메인 모듈. import해서 사용
- `demo_face_pipeline.py` — test.mp4 처리 데모 + margin 비교 + v1/v2 모델 forward 예시
- `README_face_pipeline.md` — 본 문서

## 설치

```bash
pip install retina-face opencv-python moviepy torch torchvision pillow numpy timm
```

명세서 §2 경고: `insightface`(numpy/pickle 충돌), `facenet-pytorch`(PIL 충돌) 모두 사용 불가. RetinaFace는 반드시 serengil/retina-face 패키지(`pip install retina-face`)를 사용할 것.

## 사용법

### 영상 1개 처리

```python
from face_pipeline import process_video

result = process_video(
    video_path='test.mp4',
    margin=0.15,
    n_frames=100,
    frame_sampling='uniform',
    extract_audio=True,
    save_outputs=True,
    output_dir='output',
)

print(f"검출률: {result['meta']['detection_rate']:.1%}")
print(f"area_ratio (padded): {result['meta']['area_ratio_mean']:.4f}")
print(f"det_area_ratio (raw): {result['meta']['det_area_ratio_mean']:.4f}")
print(f"tensor_batch shape: {result['tensor_batch'].shape}")  # (N, 3, 224, 224)
```

### 모델 forward (video-level aggregation, 13주차 14절 권고)

```python
import torch, timm

model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k',
                          pretrained=False, num_classes=2)
model.load_state_dict(torch.load(V1_CKPT, map_location='cuda'), strict=False)
model.eval().cuda()

with torch.no_grad():
    probs_real = torch.softmax(model(result['tensor_batch'].cuda()), dim=1)[:, 1]
    video_p_real = probs_real.mean().item()   # frame 평균
    # 또는 probs_real.median().item()
```

### 단일 이미지 (ff_c23 raw frame 등)

```python
from face_pipeline import process_image

info = process_image('/path/to/ff_c23/Original/123_f0.jpg', margin=0.15)
if info is not None:
    crop_224 = info['crop_224']         # (224, 224, 3) RGB
    bbox = info['bbox']                  # padding 적용 후
    det_area_ratio = info['det_area_ratio']  # padding 전 (14주차 비교용)
```

### CLI

```bash
python face_pipeline.py test.mp4 --margin 0.15 --n_frames 100
python demo_face_pipeline.py test.mp4 --with_model   # v1/v2 forward까지
```

## 반환 dict 구조 (§6 명세서 100% 준수)

`process_video`는 다음 dict를 반환한다.

- `crops`: List[dict]. 각 dict는 `frame_idx`, `crop_224`(224×224×3 RGB), `bbox`(padding 후), `det_bbox`(padding 전), `det_score`, `area_ratio`, `det_area_ratio`, `cy`, `aspect`
- `tensor_batch`: torch.Tensor, (N, 3, 224, 224). EVAL_TF 적용 후
- `vis_imgs`: List[np.ndarray]. 정규화 전 (224, 224, 3) RGB. Grad-CAM용
- `audio_path`: Optional[str]
- `meta`: dict. `video_path`, `video_duration`, `total_frames`, `fps`, `sampled_frame_indices`, `detected_frames`, `failed_frame_indices`, `detection_rate`, `area_ratio_mean/std`, `det_area_ratio_mean/std`, `cy_mean`, `config`, `key_frame_indices`

§6 명세에 없는 추가 필드: `det_bbox`, `det_area_ratio`, `det_area_ratio_mean/std`, `total_frames_meta`, `key_frame_indices`. 14주차 학습 분포 측정값이 padding 적용 여부 불명확하므로, padding 전(`det_*`)과 padding 후(`area_*`) 두 가지를 모두 보관한다. **14주차 표(area_ratio 0.07~0.41)와 직접 비교할 때는 `det_area_ratio_mean`을 사용할 것.**

## 미완료 항목 (본인이 실행 검증해야 함)

본 모듈 자체는 RetinaFace 미설치 환경에서 작성됐다. 다음 항목은 본인이 Colab에서 실행 검증이 필요하다.

1. RetinaFace 첫 호출 시 모델 가중치 자동 다운로드 동작 확인 (Colab에서 보통 wget으로 받아짐)
2. test.mp4 처리 결과의 검출률·area_ratio 측정값
3. v1, v2 모델 forward까지 통과해서 video-level P(Real) 출력 확인
4. margin 0.0 / 0.15 / 0.30 sweep 결과를 표로 정리 → 팀원 A 공유
5. v1 MediaPipe 버전 출력과의 검출률·area_ratio 차이 비교 (§7 A 산출물 3)

위 5번까지 완료해야 §7 A의 "완료 기준 4: 팀원 A에게 1회 데모"가 성립한다.

## 인터페이스 변경 시

§8 절차 준수: 본인이 수정 필요를 발견하면 팀원 A에 목적 + 변경 전후를 제안 → 합의 → 양쪽 코드 업데이트 → docstring과 통합 명세서 둘 다 기록.
