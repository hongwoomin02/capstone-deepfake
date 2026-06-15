# Fact-Trace AI — 영상 딥페이크 탐지 모듈

부경대학교 컴퓨터·인공지능공학부 캡스톤디자인 I (2026-1)

**ConvNeXtV2 기반 영상 face-swap 딥페이크 탐지 모델의 학습 및 외부 도메인 평가** 코드입니다. 출처가 다른 face-swap 데이터셋(FaceForensics++, Celeb-DF, KoDF)을 단일 얼굴 검출 파이프라인으로 통일하여 영상 단위 누수 없이 통합 학습하고, 학습 분포 내부(in-domain)와 외부 face-swap 데이터셋(DFD)에서 교차 평가하여 일반화 성능을 측정합니다.

>  데이터셋 원본·crop 이미지와 체크포인트(`*.pt`)는 라이선스·용량 문제로 저장소에 포함되지 않습니다. **코드, 학습셋 메타데이터(CSV), 평가 예측 결과만** 제공합니다. 원본 데이터 획득 방법은 [데이터셋](#데이터셋) 절을 참고하세요.

---

## 주요 결과

| 평가셋 | 유형 | n | fake recall | real recall | macro F1 | AUC |
|---|---|---|---|---|---|---|
| in-domain test | face-swap (학습 분포) | 11,914 프레임 | 0.918 | 0.969 | **0.943** | 0.983 |
| DFD | face-swap (외부 도메인) | 726 영상 | 0.683 | 0.945 | 0.811 | 0.900 |

- **도메인 갭**: 같은 face-swap도 학습 분포(fake recall 0.918) → 외부 DFD(0.683)로 하락. AUC(순위 능력)는 0.900으로 유지되나 고정 임계값 0.5에서 fake를 일부 놓칩니다.
- **임계값 트레이드오프**: 임계값을 올리면 fake recall은 회복(0.683→0.879, t=0.80)되나 real recall이 하락(0.945→0.672)하여, 단일 임계값 조정만으로 종합 성능은 개선되지 않습니다(t=0.65가 균형점).
- **lip-sync 구조적 취약**: 프레임 단위 face-swap 모델은 입모양만 합성하는 lip-sync에 구조적으로 취약합니다(in-domain `audio_driven` recall 0.000). 한국 정치인 lip-sync(KoEBA) 외부 평가는 후속 과제입니다.

---

## 모델 설정

| 항목 | 값 |
|---|---|
| Backbone | ConvNeXtV2-Tiny (`timm: convnextv2_tiny.fcmae_ft_in22k_in1k`, ImageNet 사전학습 fine-tuning) |
| 라벨 | 0 = Fake, 1 = Real, P(Real) = softmax[:, 1] |
| Optimizer / LR | AdamW (weight_decay 0.05), 5e-5, CosineAnnealingLR |
| Loss / Batch | CrossEntropyLoss, batch 64 (AMP) |
| Epochs | 10 (Early Stopping patience 3) → best epoch 9 |
| 얼굴 검출 | RetinaFace (threshold 0.5), margin 0.15, 224×224 |
| SEED | 42 |

---

## 디렉터리 구조

```
capstone-deepfake/
├── 최종모델학습.ipynb         # 전체 파이프라인 (Google Colab)
│                              #  데이터셋 통합·해제 → 모델 학습(resume·early stop)
│                              #  → in-domain 평가 → source·fake_type별 오판 분석
├── requirements.txt          # 의존 패키지
├── LICENSE                   # MIT
│
├── csv/                      # 통합 학습셋 메타데이터 (video_id 단위 split, 누수 0) — 이미지 미포함
│   ├── train_v4.csv          #  학습 50,063 프레임
│   ├── val_v4.csv            #  검증 11,300 프레임
│   └── test_v4.csv           #  in-domain test 11,914 프레임
│                             #  컬럼: filepath, label, source, fake_type, video_id,
│                             #        split, det_area_ratio, cy
│
├── face_pipeline/            # 영상 전처리 모듈 (RetinaFace 기반, 캡스톤2 진입용)
│   ├── face_pipeline.py      #  process_video / process_image
│   │                         #   영상 → 얼굴 검출·정규화(224×224) → tensor + 메타데이터
│   ├── demo_face_pipeline.py #  동작 데모 (margin 0.0 / 0.15 / 0.30 비교 등)
│   ├── verify_video_extraction.py  # 영상 직접 추출 vs face-crop 분포갭 소규모 검증
│   ├── README_face_pipeline.md     # 모듈 설명 (v1→v2 변경 요약)
│   └── face_pipeline_handover.md   # 인계 문서
│
└── results/                  # 평가 예측 결과 (CSV)
    ├── v4_test_predictions.csv  #  in-domain test 프레임별 예측 (p_real 포함)
    └── v4_dfd_predictions.csv   #  DFD 영상 단위 예측 (video_score, pred)
```

학습용 통합 CSV의 모든 이미지·영상·체크포인트(`*.pt`)는 저장소에서 제외됩니다.

---

## 데이터셋

원본 데이터는 각 출처에서 직접 획득해야 하며, 본 저장소는 재배포하지 않습니다.

| 데이터셋 | 용도 | 출처 / 라이선스 |
|---|---|---|
| FaceForensics++ | 학습 | Rössler et al., ICCV 2019 (FaceForensics 서버) |
| Celeb-DF (v2) | 학습 | Li et al., CVPR 2020 |
| KoDF | 학습 | AI-Hub (datasetkey 55), Kwon et al., ICCV 2021 |
| DFD (DeepFakeDetection) | 외부 평가 | Google/FaceForensics (Kaggle: `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset`) |
| KoEBA | 외부 평가 (후속 과제) | DeepBrain AI Research, [github.com/deepbrainai-research/koeba](https://github.com/deepbrainai-research/koeba) — CC BY-NC-SA 4.0, 저작권: 각급 선거관리위원회(공직선거법 §279) |

**라이선스 주의**
- KoEBA / KoDF / FF++ / Celeb-DF / DFD 원본·crop 이미지는 **재배포 금지**입니다.
- KoEBA는 비상업·연구 목적만 허용되며, 영상 저작권은 선거관리위원회에 있습니다. 원본 영상을 재배포하지 마세요.

---

## 한계

- 단일 SEED(42) 1회 학습·평가 기준이며, 반복 측정(평균·표준편차)은 수행하지 않았습니다.
- 외부 평가는 현재 DFD(face-swap) 1종이며, 한국 정치 도메인 lip-sync(KoEBA) 외부 평가는 후속 과제입니다.
- KoDF in-domain test는 인물 수가 적어(기법당 1~2명) 통계적 신뢰가 제한적입니다.

---

## 참고문헌

1. A. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV, 2019.
2. Y. Li et al., "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics," CVPR, 2020.
3. P. Kwon et al., "KoDF: A Large-Scale Korean DeepFake Detection Dataset," ICCV, 2021.
4. S. Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," CVPR, 2023.
5. G. Hwang et al., "DisCoHead: Audio-and-Video-Driven Talking Head Generation," arXiv:2303.07697, 2023. (KoEBA)

---

## 라이선스

코드는 MIT 라이선스를 따릅니다. 데이터셋은 각 원 출처의 라이선스를 따릅니다.
