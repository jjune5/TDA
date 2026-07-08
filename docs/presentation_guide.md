# 발표자료 구성 가이드라인
> 연구 제목 (안): **"이종 그래프에서의 위상적 특징 학습: GTN-PDGNN을 활용한 노드 분류 성능 향상"**
> 
> 대상: 학부/대학원 발표, 15-20분 분량 기준

---

## 1. 서론 (Introduction)

### 1-1. 연구 배경 및 필요성

**[슬라이드 1-2] 이종 그래프(Heterogeneous Graph)란?**
- 실세계의 많은 데이터는 여러 종류의 노드와 엣지로 구성된 이종 그래프
  - 예: 학술 네트워크 (논문 · 저자 · 키워드), 영화 DB (영화 · 배우 · 장르), 지식 그래프
- 동종 그래프(homogeneous)와의 차이: 노드 타입, 관계(relation) 타입이 복수
- 이종 그래프에서의 노드 분류 = 실용적으로 중요한 과제

**[슬라이드 3-4] 기존 방법론과 한계**

> 핵심 주장: 기존 GNN은 **국소(local) 이웃 집계**에 머물러 **전역(global) 위상 구조**를 반영하지 못한다.

- **기존 Het GNN의 작동 방식 (메시지 패싱)**
  - 각 노드는 이웃 노드의 feature를 집계(aggregate)하여 자신의 표현 갱신
  - 대표 방법: HAN (metapath 기반 어텐션), RGCN (relation별 가중치 행렬)
  - 집계 반경 = 레이어 수 × hop — 전형적으로 2-3 hop

- **한계점**
  1. **국소성(Locality)**: 집계 반경 밖의 구조 정보 무시
  2. **위상 맹목성(Topology Blindness)**: 루프, 구멍, 클러스터 연결성 등 전역 위상 구조를 특성화하지 못함
  3. **이종성(Heterogeneity)**: 위상 분석 도구 대부분이 동종 그래프만 지원

- **시각 자료 제안**: 2-hop 집계 범위 vs. 실제 그래프 전역 구조 비교 그림

**[슬라이드 5] 연구 동기: "왜 위상(Topology)인가?"**
- 지속적 호몰로지(Persistent Homology): 연결 성분(H0), 루프(H1) 등 다양한 스케일의 구조를 수치화
- 예시: 같은 이웃을 가져도 전역 위상이 다른 두 노드 → 메시지 패싱은 구분 불가, 위상 descriptor는 구분 가능
- **직관**: 의학 논문 그래프에서 동일 저자 이웃이라도 해당 저자가 "허브 클러스터"에 속하는지 여부는 분류에 중요할 수 있음

---

### 1-2. 배경 지식

**[슬라이드 6] 지속 호몰로지와 Extended Persistence Diagram (EPD)**
- **지속 호몰로지**: 그래프에 필터 함수 f를 적용하여 임계값을 점점 높이면서 위상 특징(연결 성분, 루프)의 생성(birth)·소멸(death) 추적
- **Persistence Diagram (PD)**: birth-death 쌍의 집합 — 오래 지속되는 쌍 = 의미 있는 구조
- **Extended PD (EPD)**: 일반 PD를 확장, up/down 방향 두 filtration을 통합 → 더 풍부한 위상 정보
- **Persistence Image (PI)**: PD를 고정 크기 벡터로 변환 — 신경망 입력에 적합
- **시각 자료 제안**: filtration 진행 과정 애니메이션 또는 단계별 그림

**[슬라이드 7] PDGNN (Yan et al., NeurIPS 2022)**
- **역할**: EPD 계산을 신경망으로 근사 — 직접 계산은 O(n³), PDGNN은 근사로 빠르게 처리
- **입력**: 동종 그래프 + HKS(Heat Kernel Signature) 필터 함수
  - HKS: 각 노드의 "열 확산 패턴"을 측정, 그래프 구조에 민감
- **출력**: 노드별 75차원 persistence image feature
- **핵심 한계**: **동종 그래프(homogeneous graph)에서만 동작**
  - 이종 그래프는 여러 관계 타입이 혼재 → 단일 라플라시안 정의 불가

**[슬라이드 8] 메타패스(Metapath)란?**
- 정의: 이종 그래프에서 동일 타입 노드를 연결하는 관계 시퀀스
  - 예: `P → A → P` (논문-저자-논문) = PAP, `P → S → P` (논문-주제-논문) = PSP
- 역할: 메타패스를 따라 인접 행렬을 구성하면 이종 그래프 → 동종 그래프 변환 가능
- **시각 자료 제안**: ACM 데이터셋의 PAP, PSP 메타패스 그림

**[슬라이드 9] GTN (Yun et al., NeurIPS 2019)**
- **역할**: 메타패스를 사람이 직접 정의하지 않고 **자동으로 학습**
- **동작 원리**: 여러 관계 타입의 인접 행렬에 소프트 가중치를 학습하여 조합 → 채널별 메타패스 그래프 H 생성
  - `H = A_r1 × A_r2 × ... (학습된 가중치로 혼합)`
- **출력**: `num_channels=4`개의 동종 그래프 H₀, H₁, H₂, H₃
- **시각 자료 제안**: 채널별 relation 가중치 히트맵(→ metapath.png 활용)

**[슬라이드 10] 이 연구의 핵심 아이디어 (브리지)**

```
이종 그래프
    ↓ GTN (메타패스 자동 발견)
채널별 동종 그래프 H₀~H₃
    ↓ PDGNN (위상 feature 추출)
노드별 위상 feature (75d × 4채널)
    ↓ SemanticAttentionFusion
위상 feature 융합
    ↓ RGCN/HAN (backbone 분류기)
노드 분류
```

> "GTN이 이종→동종 변환을 담당하고, PDGNN이 동종에서 위상 추출을 담당 — 두 모듈의 결합이 핵심 기여"

---

### 1-3. 연구 목적 및 핵심 질문

**[슬라이드 11]**

> **Q1 (주 질문)**: GTN-PDGNN 위상 feature를 이종 그래프 GNN에 추가하면 노드 분류 성능이 향상되는가?

> **Q2 (기여 분리)**: 성능 향상이 있다면, 그것이 **진짜 위상 신호** 때문인가, 아니면 **feature 차원 증가** 효과인가?

> **Q3 (backbone 의존성)**: 위상 feature의 기여는 backbone 구조(HAN vs. RGCN)에 따라 달라지는가?

---

## 2. 연구 방법 (Methodology)

### 2-1. 전체 파이프라인

**[슬라이드 12]** (파이프라인 다이어그램)
- **3단계 Staged Training** (end-to-end 역전파 없음 — 독립 모듈)
  - Stage 1 → Stage 2 → Stage 3 순차 실행
  - 이유: GTN·PDGNN 결합 시 gradient explosion 위험; 각 모듈 독립 최적화가 더 안정적

| Stage | 모듈 | 입력 | 출력 |
|-------|------|------|------|
| 1 | GTN | 이종 그래프 전체 | H (4, N, N) 채널별 메타패스 그래프 |
| 2 | PDGNN | H₀~H₃ | 노드별 위상 feature (N, 75×4→75) |
| 3 | RGCN / HAN | 원본 feature ⊕ 위상 feature | 노드 분류 |

### 2-2. 실험 매트릭스 (Ablation Design)

**[슬라이드 13]**

|  | topology 없음 | noisy topology | GTN-PDGNN topology |
|--|:---:|:---:|:---:|
| **HAN** | (a) | (b) | (c) |
| **RGCN** | (d) | (e) | **(f)** |

- **(a)(d)**: 기준선 — backbone만 사용
- **(b)(e) noisy baseline**: 같은 차원의 랜덤/혼합 feature 주입 → "feature 차원 효과"만 측정
  - b1/e1: Random Gaussian noise
  - b2/e2: Class-wise mixing (같은 클래스 내 위상 feature 섞기 — 노드 구분력 제거)
- **(c)(f)**: 실제 위상 feature 주입 — 핵심 주장

**noisy baseline이 왜 필요한가?**
> `(f) > (e)` 이어야만 "위상 신호" 효과가 있다고 주장할 수 있음. `(f) ≈ (e)` 이면 차원 증가 효과에 불과.

**백본 선택 이유: RGCN vs. HAN**

| | HAN | RGCN |
|-|-----|------|
| 메타패스 처리 | semantic attention (강) | relation별 W_r 집계 (약) |
| 위상 기여 예상 | 이미 메타패스 주의 → 기여 제한 | 메타패스 주의 없음 → 위상 기여 더 명확 |

> RGCN을 주 실험 백본으로 선택한 이유: 위상 feature의 독립 기여를 더 깨끗하게 측정 가능

### 2-3. 데이터셋

**[슬라이드 14]**

| 데이터셋 | 도메인 | 규모 (N) | 관계 수 | node feature | 라벨 |
|---------|--------|---------|--------|-------------|------|
| ACM | 학술/인용 | 3,025 | 5 | 강 (1902d) | 단일 (3) |
| DBLP | 학술/인용 | ~4,000 | 3 | 약 (334d) | 단일 (4) |
| IMDB | 영화 | ~11,000 | 4 | 강 (3489d) | 다중 (5) |
| Freebase | 지식그래프 | ~40,000 | 36 | 없음 | 단일 |
| MAG | 학술(대규모) | ~12,000 | 4 | 중 (128d) | 단일 (349) |
| AIFB | RDF/연구기관 | ~2,000 | 45 | 없음 | 단일 (4) |
| Yelp | 비즈니스 리뷰 | ~8,000 | 3 | 없음 | 다중 (16) |

**데이터셋 선택 근거 — 다양성 최대화**
- Node feature 강도: 강(ACM, IMDB) / 약(DBLP) / 없음(Freebase, AIFB, Yelp)
- 그래프 규모: 소형(AIFB) ~ 대형(Freebase)
- 라벨 유형: 단일 분류 / 다중 레이블
- **가설**: featureless 데이터셋일수록 위상 feature의 기여가 클 것

### 2-4. 평가 지표 및 재현성 설정

**[슬라이드 15]**
- **주 지표**: Macro-F1 (클래스 불균형에 강인), Test Accuracy
- **집계 방식**: 10 seeds 평균 ± 표준편차
  - Seeds: 312132, 238623, 792965, 15092, 661491, 588722, 825661, 500973, 88015, 251219
  - 동일 seed를 모든 조건(a-f)에 적용 → paired comparison 가능
- **이유**: 단일 seed 결과는 운에 의한 분산이 크므로, 통계적으로 유의미한 결론을 위해 10-fold 반복

### 2-5. 구현 세부사항 (선택 슬라이드)

**[슬라이드 16 — 기술적 청중 대상 선택]**
- **GTN 하이퍼파라미터**: num_channels=4, num_layers=2, hidden=64, epochs=50
- **PDGNN 하이퍼파라미터**: HKS scales K=3, kNN k=20, PI resolution=5×5
- **RGCN 하이퍼파라미터**: hidden=64, 2-layer, lr=0.01, dropout=0.5, epochs=100
- **SemanticAttentionFusion**: 채널별 학습 가능 가중치 β로 위상 feature 가중 평균
- Framework: PyTorch Geometric, GPU 학습 (CUDA)

---

## 3. 연구 결과 (Results) — 가이드라인

> ⚠️ 결과는 실험 완료 후 채워 넣을 것. 아래는 **어떤 내용을 어떤 형식으로 제시해야 하는가**에 대한 가이드라인.

### [필수] 슬라이드 A — 메인 성능 비교 테이블

**포함 내용**
- 조건별 (d) RGCN only / (f) RGCN+GTN-PDGNN × 7 데이터셋
- 형식: `mean ± std` (Macro-F1 기준)
- 강조: Δ = (f) - (d), 통계적으로 유의미한 개선 여부

**해석 포인트**
- featureless 데이터셋(Freebase, AIFB, Yelp)에서 Δ가 큰가?
- Feature가 강한 데이터셋(ACM, IMDB)에서는 Δ가 작은가?

### [필수] 슬라이드 B — Noisy Baseline과의 비교

**포함 내용**
- (d) / (e random noise) / (e class-wise) / (f) 4개 조건 비교 bar chart
- 핵심 확인: `(f) > (e)` 여부 → 위상 신호의 실질적 기여 입증

**해석 포인트**
- `(f) ≈ (e)` : 위상 신호 없음, 차원 효과만
- `(f) > (e) > (d)` : 위상 신호 존재
- `(f) ≈ (d) < (e)` : 위상 feature가 오히려 노이즈 (주의 필요)

### [권장] 슬라이드 C — HAN vs. RGCN backbone 비교

**포함 내용**
- (c) HAN+GTN-PDGNN vs. (f) RGCN+GTN-PDGNN의 Δ 비교
- 데이터셋별로 어느 backbone이 위상 feature를 더 잘 활용하는가

**해석 포인트**
- 가설: RGCN에서 위상 기여가 더 크다 (HAN은 이미 메타패스 주의 내재)

### [권장] 슬라이드 D — 데이터셋별 패턴 분석

**포함 내용**
- x축: node feature 강도(강/약/없음), y축: Δ F1
- 산점도 또는 grouped bar chart

**해석 포인트**
- "Feature가 약할수록 위상 기여가 크다"는 가설 검증

### [선택] 슬라이드 E — 시각화 결과

**포함 내용**

1. **GTN Metapath 시각화** (`metapath.png`)
   - 채널별 relation 가중치 히트맵
   - 해석: 각 채널이 어떤 관계를 주로 학습했는가 (e.g., 채널 0은 PAP 중심)

2. **Persistence Image 시각화** (`epd_pi.png`)
   - 채널별 평균 persistence image
   - 해석: 채널마다 위상 패턴이 다른가 → 채널 다양성 확인

3. (선택) 노드 임베딩 t-SNE
   - 위상 feature 추가 전/후 임베딩 분포 비교

### [선택] 슬라이드 F — 수렴 분석

**포함 내용**
- 대표 데이터셋(ACM)의 학습 곡선: Stage 1 GTN val_f1, Stage 3 RGCN val_f1
- Staged training의 안정성 확인

---

## 4. 결론 및 제언 (Conclusion & Future Work)

### 4-1. 결과 요약

**[슬라이드 17]**
- **(작성 예시 — 결과 나오면 수치 채울 것)**
  - "7개 데이터셋 중 X개에서 위상 feature 추가 시 Macro-F1 유의미 향상"
  - "특히 featureless 데이터셋(Freebase, AIFB)에서 Δ 가장 큰 것으로 확인"
  - "(f) > (e) 결과: 성능 향상이 feature 차원 증가가 아닌 위상 신호에서 비롯됨을 지지"

### 4-2. 연구의 의의 및 시사점

**[슬라이드 18]**
- **방법론적 기여**: PDGNN(동종 전용)을 이종 그래프에 처음 적용한 사례
  - GTN을 브리지로 사용한 설계 — 추가 재학습 없이 이종 그래프에서 위상 분석 가능
- **실용적 시사점**: Node feature가 부족한 이종 그래프(지식 그래프, RDF 등)에서 위상 feature가 보완재로 유효
- **이론적 시사점**: GNN이 놓치는 전역 위상 정보가 노드 분류에 실질적 기여를 함을 실증

### 4-3. 한계 및 향후 연구 과제

**[슬라이드 19]**

**현재 한계**
1. **계산 비용**: Stage 2 PDGNN이 run당 ~16분 소요 (N=3025 기준) — 대규모 그래프에 scalability 문제
2. **Staged Training**: GTN → PDGNN → backbone을 독립 학습, end-to-end 최적화 미지원 → 최적 결합 미달성 가능
3. **고정 채널 수**: GTN num_channels=4 고정 — 데이터셋 특성에 따른 최적 채널 수 탐색 미수행
4. **메타패스 해석 가능성**: GTN의 소프트 가중치는 정확히 어떤 메타패스를 의미하는지 해석 어려움

**향후 연구 방향**
1. **End-to-end 학습**: GTN + PDGNN + backbone의 통합 역전파 (gradient checkpointing 등으로 메모리 관리)
2. **Scalability**: Mini-batch PDGNN, ego-graph 샘플링으로 대규모 그래프 대응
3. **다른 위상 descriptor 탐색**: Betti curve, Mapper graph 등 EPD 외 위상 특징
4. **더 강한 backbone 적용**: HGT, Simple-HGN, MAGNN 등에서의 위상 기여 검증
5. **Link prediction · Graph classification으로 태스크 확장**

---

## 부록: 발표 흐름 체크리스트

```
□ 서론
  □ 이종 그래프 정의 + 실세계 예시  (1 slide)
  □ 기존 방법론 (메시지 패싱) 설명  (1 slide)
  □ 위상 분석 동기: 국소 집계의 한계  (1 slide)
  □ 배경지식: EPD, PDGNN, PDGNN 한계  (2 slides)
  □ 배경지식: 메타패스, GTN  (2 slides)
  □ 핵심 아이디어 (브리지 다이어그램)  (1 slide)
  □ 연구 질문 3개  (1 slide)

□ 연구 방법
  □ 전체 파이프라인 (3 Stage)  (1 slide)
  □ 실험 매트릭스 (a-f) + noisy baseline 설명  (1 slide)
  □ backbone 선택 이유  (1 slide)
  □ 데이터셋 + 선택 근거  (1 slide)
  □ 평가 지표 + 재현성  (1 slide)

□ 연구 결과
  □ [필수] 메인 성능 테이블
  □ [필수] noisy baseline 비교 (위상 신호 검증)
  □ [권장] HAN vs. RGCN 비교
  □ [권장] 데이터셋별 패턴 분석
  □ [선택] GTN metapath 시각화
  □ [선택] Persistence image 시각화

□ 결론
  □ 결과 요약 (3줄 이내)
  □ 의의 및 시사점
  □ 한계 + 향후 과제
```

---

## 부록: 핵심 개념 원페이퍼 요약 (발표 중 질문 대비)

| 개념 | 한 줄 정의 | 논문 |
|------|-----------|------|
| Persistent Homology | 필터 함수 임계값 변화에 따른 위상 특징(연결 성분·루프) 추적 | Edelsbrunner et al. 2002 |
| Extended PD (EPD) | up/down filtration을 통합한 persistence diagram의 확장 | Cohen-Steiner et al. 2009 |
| HKS | 그래프 열 방정식 해 — 각 노드의 구조적 역할을 시간 스케일별로 인코딩 | Sun et al. 2009 |
| PDGNN | EPD를 신경망으로 근사, end-to-end 학습 가능 | Yan et al. NeurIPS 2022 |
| GTN | 관계 인접행렬의 학습된 가중합으로 메타패스 자동 발견 | Yun et al. NeurIPS 2019 |
| HAN | 메타패스 기반 노드-수준 + 의미-수준 dual attention | Wang et al. WWW 2019 |
| RGCN | 관계별 독립 가중치 행렬 W_r, basis decomposition 가능 | Schlichtkrull et al. ESWC 2018 |
| Metapath | 이종 그래프에서 동일 타입 노드를 연결하는 관계 경로 시퀀스 | Sun et al. VLDB 2011 |
