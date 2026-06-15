# face_pipeline 인계 문서 — 팀원 A 작업 B용

작성: 팀원 B / 2026-05-27
근거 문서: §6 face_pipeline 인터페이스 명세, 14주차 보고서 v2 §10~§12

본 문서 한 장 + 첨부 코드 3개(face_pipeline.py, demo_face_pipeline.py, README_face_pipeline.md)면 팀원 A의 작업 B(외부 영상 분포 측정 + 학습 분포 비교)를 시작할 수 있다.

---

## 1. 작업 흐름상 위치

```
[완료] 팀원 B: face_pipeline 재작성 (RetinaFace 교체, 함수화, margin 파라미터화)
[완료] 팀원 B: test.mp4로 동작 검증
[완료] 팀원 B: margin 0.0/0.15/0.30 비교 측정 → 1차 추정
[지금] 팀원 B → 팀원 A 인계 (본 문서)
[다음] 팀원 A: face_pipeline으로 외부 영상(셀카 + 뉴스) 분포 측정 → 학습 분포(0.27~0.41) vs 실서비스 분포 비교
```

---

## 2. 설치 + Drive 경로

### Colab 셀 1 (세션 시작 후 1번만)

```python
!pip install retina-face -q

from google.colab import drive
drive.mount('/content/drive')
```

### Colab 셀 2 (sys.path는 매 세션마다 필요)

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/capstone_deepfake/capstone_deepfake/face_pipeline')

from face_pipeline import process_video, process_image
print("import OK")
```

### 환경 통일 (명세서 §2와 일치)

```python
PROJECT_ROOT = "/content/drive/MyDrive/capstone_deepfake/capstone_deepfake"
V1_CKPT = f"{PROJECT_ROOT}/checkpoints/main_unified/best_model_unified.pt"
V2_CKPT = f"{PROJECT_ROOT}/checkpoints/main_unified_v2/best_model.pt"
```

### 주의

- RetinaFace 첫 호출 시 가중치(retinaface.h5, ~119MB)가 `~/.deepface/weights/`에 자동 다운로드 (30초~1분)
- TF 관련 경고가 잔뜩 출력되는데 무시 가능
- `insightface`, `facenet-pytorch`는 환경 충돌로 사용 불가 (명세서 §2)

---

## 3. 사용 예시 — 영상 1개 처리

```python
result = process_video(
    video_path='/content/drive/MyDrive/.../영상.mp4',
    margin=0.15,                  # 0.0 / 0.15 / 0.30 등
    n_frames=100,                 # 균등 N장 추출
    frame_sampling='uniform',     # 'uniform' / 'fps_based' / 'all'
    extract_audio=False,          # 분포 측정만 할 때는 False가 빠름
    save_outputs=False,           # 측정만 할 때는 False (디스크 안 쓰고 속도↑)
)

m = result['meta']
print(f"검출률          : {m['detection_rate']:.1%} ({m['detected_frames']}/{len(m['sampled_frame_indices'])})")
print(f"area_raw (검출)  : {m['det_area_ratio_mean']:.4f} ± {m['det_area_ratio_std']:.4f}")
print(f"area_padded     : {m['area_ratio_mean']:.4f} ± {m['area_ratio_std']:.4f}")
print(f"cy_mean         : {m['cy_mean']:.4f}")
print(f"tensor shape    : {tuple(result['tensor_batch'].shape)}")
```

### 14주차 학습 분포(0.07~0.41)와 직접 비교할 값은 `det_area_ratio_mean`

- `area_ratio`(= padding 적용 후) = padding 후 bbox 면적 / frame 면적
- `det_area_ratio`(= padding 전 = raw 검출) = RetinaFace 원본 bbox 면적 / frame 면적
- 14주차 표는 padding 안 한 raw 측정값이므로 **반드시 `det_area_ratio`로 비교**

---

## 4. 사용 예시 — 단일 이미지 처리 (셀카, 뉴스 캡처)

```python
info = process_image('/path/to/셀카.jpg', margin=0.15)
if info is None:
    print("검출 실패")
else:
    print(f"area_raw         : {info['det_area_ratio']:.4f}")
    print(f"area_padded      : {info['area_ratio']:.4f}")
    print(f"cy               : {info['cy']:.4f}")
    print(f"det_score        : {info['det_score']:.4f}")
    print(f"crop_224 shape   : {info['crop_224'].shape}")
```

여러 이미지 batch 처리:

```python
import os, glob, numpy as np

paths = glob.glob('/path/to/folder/*.jpg')
areas, cys, fails = [], [], 0
for p in paths:
    info = process_image(p, margin=0.15)
    if info is None:
        fails += 1
        continue
    areas.append(info['det_area_ratio'])
    cys.append(info['cy'])

print(f"검출률    : {1 - fails/len(paths):.1%}")
print(f"area_raw  : {np.mean(areas):.4f} ± {np.std(areas):.4f}")
print(f"cy_mean   : {np.mean(cys):.4f}")
```

---

## 5. test.mp4 기준 측정값 (팀원 A 검증 baseline)

### 영상 정보
- 출처: 유튜브 (캡스톤1 노션 기록, 단일 인물, 화면 중앙, 안정 조명)
- 길이: 135.36 초 / 3,384 frame / 25 fps
- 처리 설정: n_frames=100, frame_sampling='uniform'

### n_frames=10 결과 (빠른 확인용)

| 지표 | 값 |
|---|---|
| 검출률 | 9/9 = 100% (실제) ※ 메타 부정확 사유는 §8 참조 |
| det_area_ratio (raw) | 0.0928 |

### margin sweep 결과 (n_frames=100)

| margin | det_rate | area_padded | area_raw | cy |
|---|---|---|---|---|
| 0.00 | 99.0%* | 0.0937 | 0.0937 | 0.395 |
| 0.15 | 99.0%* | 0.1577 | 0.0937 | 0.395 |
| 0.30 | 99.0%* | 0.2384 | 0.0937 | 0.397 |

\* 검출 실패 1건은 메타 부정확으로 영상 범위 밖 idx 1개(§8 참조). 실제 검출률 100%.

팀원 A가 자기 환경에서 test.mp4를 같은 설정으로 돌렸을 때 위 값과 일치하면 환경 일관성 OK.

---

## 6. 14주차 학습 분포와 test.mp4 비교

학습 데이터 6 출처 face-crop 분포 (14주차 §3, RetinaFace, threshold=0.5):

| 출처 | area_ratio (raw) | cy |
|---|---|---|
| **test.mp4 (본 측정)** | **0.0937** | **0.395** |
| ff_c23 (Drive raw frame) | 0.07 | 0.35 |
| celebdf | 0.27 | 0.46 |
| dfdc | 0.34 | 0.51 |
| sd | 0.37 | 0.44 |
| ciplab | 0.41 | 0.58 |
| 140k | 0.41 | 0.55 |
| ff_c23 (Kaggle face-crop) | 0.41 | 0.50 |

### 관찰 (1개 영상 기준 — 일반화는 팀원 A의 추가 측정 필요)

- test.mp4는 area_raw·cy 두 지표 모두에서 **ff_c23 raw frame과 가장 가깝고, 나머지 5 face-crop 출처와 멀다**
- 14주차 가설 C(도메인 갭이 본질)와 정합
- margin 0.30까지 키워도 area_padded 0.2384로 학습 분포 하한(0.27) 미달 → margin만으로는 학습 분포에 못 맞춤

### 팀원 A 작업 B에서 확인 권장

- test.mp4 1개만으로는 일반화 불가. 셀카 N장, 뉴스 영상 N개 측정해서 분포 패턴이 일관되는지 확인
- 분포가 0.07~0.10 부근에 모이면 학습 데이터를 face-crop이 아닌 raw frame 분포로 통일하는 방향이 강화됨
- 분포가 다양하게 흩어지면 시나리오 (가/나/다) 중 어떤 게 더 합리적인지 추가 논의 필요

---

## 7. 반환 구조 요약 (§6 명세서 100% 준수 + 추가 필드 2개)

### process_video 반환 dict

- `crops`: List[dict]
  - 각 dict 필드: `frame_idx`, `crop_224` (224,224,3 RGB), `bbox` (padding 후), **`det_bbox`** (padding 전), `det_score`, `area_ratio` (padded), **`det_area_ratio`** (raw, 14주차 비교용), `cy`, `aspect`
- `tensor_batch`: torch.Tensor (N, 3, 224, 224), EVAL_TF 적용 후 (학습 val transform과 동일)
- `vis_imgs`: List[np.ndarray] (224, 224, 3) RGB, normalize 전 (Grad-CAM 시각화용)
- `audio_path`: Optional[str]
- `meta`: dict — 모든 통계 (`area_ratio_mean`, `det_area_ratio_mean`, `cy_mean`, `detection_rate`, `sampled_frame_indices`, `failed_frame_indices`, `config` 등)

### 명세서 §6에 없는 추가 필드 (사유)

`det_bbox`, `det_area_ratio`, `det_area_ratio_mean/std`

14주차 학습 분포 측정(area_ratio 0.07~0.41)이 padding 적용 여부 불명확. padding 전(`det_*`)과 padding 후(`area_*`) 둘 다 보관해서 14주차 표와 직접 비교 가능하도록.

→ 인터페이스 추가만 한 거라 기존 사용 코드와 호환성 OK. 명세서 §8 절차상 사후 보고로 충분한지, 명세서 §6 본문 갱신 필요한지 회신 부탁.

---

## 8. 알려진 한계 / 미완료 항목

### 한계 1: OpenCV 메타 부정확

`CAP_PROP_FRAME_COUNT`가 1 frame 어긋나는 mp4가 존재. test.mp4도 메타 3385 반환했지만 실제 read는 3384 frame까지. 결과적으로 sampled의 마지막 idx 1개가 영상 범위 밖이 되어 `failed_frame_indices`에 1건 들어감. 측정값에는 영향 없으나(검출 성공한 frame만 통계에 포함) detection_rate 계산이 약간 깎임. 다음 패치에서 `failed_frame_indices`를 "진짜 검출 실패" vs "out_of_range" 분리 예정.

### 한계 2: GPU vs CPU

RetinaFace는 TensorFlow 기반. Colab 무료 CPU 런타임에서 RetinaFace 100회 ≈ 2~4분, GPU(T4) 런타임이면 30초~1분. 외부 영상 다량 처리 시 GPU 권장.

### 한계 3: 메모리

streaming read 패턴이라 영상 한 frame씩만 메모리에 두지만, RetinaFace + TF 로딩 자체가 약 1.5GB 차지. Colab 무료 12.7GB RAM에서 동시 처리 영상 수 주의 (1~2개씩).

### 한계 4: multi_face='all' 미구현

현재 `multi_face='largest'`만 동작. 화면에 여러 명 등장하는 영상에서 모든 얼굴 처리 필요하면 추후 확장 필요. test.mp4는 단일 인물이라 영향 없음.

---

## 9. 인터페이스 변경 요청 시 (명세서 §8 절차)

팀원 A가 작업 B 진행 중 인터페이스 수정 필요를 발견하면:

1. 팀원 B에 목적 + 변경 전후 제안
2. 합의 → 양쪽 코드 업데이트
3. 변경 사항은 face_pipeline.py docstring + 통합 작업 명세서 둘 다 기록

---

## 10. 첨부 파일

- `face_pipeline.py` — 메인 모듈
- `demo_face_pipeline.py` — test.mp4 처리 + margin sweep + v1/v2 forward 예시
- `README_face_pipeline.md` — 변경 사항 (MediaPipe→RetinaFace 등 §5 7가지) + 사용법
- `face_pipeline_handover.md` — 본 문서

모두 Drive `capstone_deepfake/capstone_deepfake/face_pipeline/` 폴더에 업로드 완료.
