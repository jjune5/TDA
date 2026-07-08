# 실험 (f) 구현 가이드 — RGCN + GTN-PDGNN

> **담당:** 실험 (f) `RGCN + GTN-PDGNN feature (by learned meta-path)`  
> **목적:** 이 문서는 실험 설계 배경, 코드 수정 내용, 실행 방법을 팀원과 공유하기 위해 작성됩니다.

---

## 1. 전체 실험 설계 맥락

### 연구 질문

> "GTN-PDGNN이 만든 위상(topological) feature를 이종 그래프(Het GNN)에 추가하면 성능이 오르는가?  
> 그리고 그 향상이 **진짜 위상 신호** 덕분인가, 아니면 **단순히 feature 차원이 늘어서**인가?"

### 실험 매트릭스 (2 백본 × 3 위상 조건)

|  | topology 없음 | noisy topology | GTN-PDGNN topology |
|---|:---:|:---:|:---:|
| **HAN** | (a) | (b) | (c) |
| **RGCN** | (d) | (e) | **(f) ← 우리 담당** |

- **(a)(d)**: 위상 없이 backbone만 — 기준선
- **(b)(e)**: noisy feature 삽입 대조군 — "feature 추가 효과만" 측정
  - `b1/e1`: random noise (같은 차원의 랜덤 Gaussian feature)
  - `b2/e2`: class-wise mixing (같은 클래스 내 위상 feature shuffle → node-level topology 파괴)
- **(c)(f)**: GTN-PDGNN 실제 위상 feature 추가 — 핵심 주장

**noisy baseline의 역할**: (c)나 (f)의 성능 향상이 `(c) > (b)` 또는 `(f) > (e)`를 만족해야만  
"위상 신호 덕분"이라는 주장이 성립한다.

### 7개 데이터셋

| 데이터셋 | 도메인 | 특징 | node feature |
|---------|--------|------|-------------|
| acm | 학술/인용 | dense, 3 class | 강 (1902d) |
| dblp | 학술/인용 | 4 class | 약 (334d) |
| imdb | 영화 | 멀티라벨(5) | 강 (3489d) |
| freebase | 지식그래프 | 관계 36종 | featureless |
| mag | 학술(초대형) | 349 class | 중 (128d) |
| aifb | RDF/연구기관 | 소형 | featureless |
| yelp | business | 멀티라벨(16), 밀집 | featureless |

> 선택 기준: 도메인 다양성 + feature 강도(강/약/없음) + 라벨 타입(단일/멀티) 최대화.  
> 자세한 선택 근거: [`results/DATASETS.md`](../results/DATASETS.md)

---

## 2. 파이프라인 구조

```
이종 그래프 (HeteroData)
    │
    ▼
[Stage 1] GTN — 메타패스 자동 발견
    → base_relations를 조합하여 채널별 target-target 인접행렬 H (C, N, N) 생성
    │
    ▼
[Stage 2] PDGNN — 위상 feature 추출
    → 채널별 H에 kNN 희소화 → ego 그래프 → HKS 필터 → 확장 persistence diagram
    → PDGNN이 EPD를 근사 → persistence image → 노드별 75차원 feature
    → SemanticAttentionFusion으로 채널 위상 feature 융합
    │
    ▼
[Stage 3] RGCN (backbone) — 분류
    → 입력: [원본 node feature ⊕ 위상 feature]
    → metapath 그래프를 relation type별로 구분하여 집계
    → 노드 분류 (CE 또는 BCE for 멀티라벨)
```

### HAN vs RGCN 핵심 차이

| | HAN | RGCN |
|--|-----|------|
| metapath 처리 | semantic attention (node-level + metapath-level) | relation별 W_r 행렬 집계, attention 없음 |
| 강점 | metapath 의미 이해 높음 | 단순·안정적 |
| topology 기여 예상 | 이미 metapath 처리가 강해서 기여 제한적 | metapath attention 없어서 topology 기여 더 명확 |

---

## 3. 코드 수정 내용

### 3-1. `tda/models/rgcn.py` — RGCN 모델 추가 (신규)

```python
class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_relations,
                 num_bases=None, num_layers=2, dropout=0.5):
        ...
        # 마지막 레이어가 직접 num_classes 출력 (HAN과 달리 별도 Linear 없음)
        # num_bases: basis decomposition으로 관계별 파라미터 공유 (원논문 §2.2)
```

- PyG `RGCNConv` 사용
- `num_bases=None` → 관계별 독립 full weight (기본 설정)
- 마지막 conv 레이어가 직접 `num_classes`를 출력

### 3-2. `tda/train.py` — backbone 분기 추가

**추가된 내용:**

```python
# (1) torch.load 호환성 패치 (PyTorch >= 2.6 대응)
if not getattr(torch.load, '_compat_patched', False):
    torch.load = functools.partial(torch.load, weights_only=False)

# (2) RGCN import 추가
from tda.models.rgcn import RGCN

# (3) viz.py import (없으면 시각화 비활성화)
try:
    from tda.viz import save_run_figures as _save_run_figures
except Exception:
    _save_run_figures = None
```

**`_edge_index_dict_to_rgcn()` 헬퍼 추가:**
```python
def _edge_index_dict_to_rgcn(edge_index_dict, device):
    """메타패스별 edge_index_dict → (edge_index, edge_type) RGCN 포맷 변환.
    각 메타패스에 정수 relation type 0, 1, 2, ... 부여.
    """
    all_ei, all_et = [], []
    for rel_idx, ei in enumerate(edge_index_dict.values()):
        all_ei.append(ei.to(device))
        all_et.append(torch.full((ei.size(1),), rel_idx, dtype=torch.long, device=device))
    return torch.cat(all_ei, dim=1), torch.cat(all_et, dim=0)
```

**`train_rgcn()` 함수 추가** (train_han과 동일 구조):
```python
def train_rgcn(bundle, x_in_builder, in_dim, edge_index_dict, ...):
    # edge_index_dict → (edge_index, edge_type) 변환
    # RGCN 생성 (num_bases 지원)
    # 학습 루프: val_f1 기준 early stopping
    # 반환: {"test_macro_f1": ..., "test_accuracy": ..., "val_macro_f1": ...}
```

**`run()` 함수 수정:**
```python
backbone = config.get("backbone", "han")  # 'han' | 'rgcn'

# Stage 2 완료 후 시각화 저장
if output_dir and _save_run_figures is not None:
    _save_run_figures(record, topo_channels, output_dir, dataset, config)

# Stage 3 backbone 분기
_stage3_fn = train_rgcn if backbone == "rgcn" else train_han
metrics = _stage3_fn(...)
```

**CLI 인자 추가:**
```bash
--backbone [han|rgcn]   # Stage 3 backbone 선택 (기본: han)
```

### 3-3. `tda/viz.py` — 시각화 모듈 (GitHub 동기화)

실험 실행 시 `output_dir`가 지정된 경우 자동으로 저장:

| 파일 | 내용 |
|------|------|
| `metapath.png` | GTN이 발견한 채널별 relation 가중치 히트맵 + fusion beta 막대 |
| `epd_pi.png` | PDGNN이 만든 채널×스케일 평균 persistence image |
| `topo_pi.npy` | 위상 feature 원본 (N, dim) — 나중에 재시각화 가능 |

### 3-4. `experiments/gen_rgcn.py` — config 자동 생성

```python
DATASETS = ["acm", "dblp", "imdb", "freebase", "mag", "aifb", "yelp"]
SETTINGS = [
    ("d_rgcn", {"backbone": "rgcn", "use_topology": False}),       # (d) RGCN only
    ("f_rgcn", {"backbone": "rgcn", "use_topology": True,
                "topology_source": "gtn"}),                         # (f) RGCN+GTN-PDGNN
]
```

→ `configs/campaign/{ds}__{d|f}_rgcn.json` 14개 생성

---

## 4. 실행 방법

### 환경 설정

```bash
conda activate tda   # 또는 해당 환경
pip install gudhi    # PDGNN topology 계산에 필요
```

### 단일 실험

```bash
cd repos/TDA
python -m tda.train \
  --config configs/campaign/acm__f_rgcn.json \
  --dataset acm \
  --backbone rgcn \
  --topology-source gtn \
  --seed 312132 \
  --output-dir runs/campaign/acm__f_rgcn__s312132
```

### 전체 실험 (f) — 7 datasets × 10 seeds

```bash
bash experiments/run_f_rgcn.sh ./data
```

- 이미 완료된 seed는 자동으로 skip
- 결과: `runs/campaign/{ds}__f_rgcn__s{seed}/metrics.json`
- 시각화: `runs/campaign/{ds}__f_rgcn__s{seed}/metapath.png`, `epd_pi.png`

### 결과 집계

```bash
python experiments/collect_f_results.py
# → results/f_rgcn_results.md 생성
```

### 사용 seed (10개, 지정값)

```
312132  238623  792965  15092  661491
588722  825661  500973  88015  251219
```

---

## 5. Config 구조 (`configs/campaign/acm__f_rgcn.json` 예시)

```json
{
  "dataset": "acm",
  "backbone": "rgcn",
  "use_topology": true,
  "topology_source": "gtn",
  "base_relations": { "PAP": [...], "PSP": [...], ... },
  "han_metapaths": ["PAP", "PSP"],
  "gtn":   { "num_channels": 4, "num_layers": 2, "hidden_dim": 64, "epochs": 50, ... },
  "pdgnn": { "hks_K": 3, "knn_k": 20, "pi_resolution": 5, ... },
  "han":   { "hidden_dim": 64, "heads": 4, "epochs": 100, ... },
  "rgcn":  { "hidden_dim": 64, "num_layers": 2, "num_bases": null,
             "lr": 0.01, "weight_decay": 0.0005, "dropout": 0.5, "epochs": 100 }
}
```

> `han_metapaths`는 RGCN 실험에서도 edge_index_dict 구성에 사용됩니다  
> (metapath 그래프 → relation type 정수 매핑).

---

## 6. 출력 파일 구조

```
runs/campaign/
└── acm__f_rgcn__s312132/
    ├── config.json        # 실행에 사용된 config 전체
    ├── metrics.json       # test_macro_f1, test_accuracy, val_macro_f1, seed, ...
    ├── metapath.png       # GTN 메타패스 시각화
    ├── epd_pi.png         # PDGNN persistence image 시각화
    └── topo_pi.npy        # 위상 feature 원본 (nch, N, 75)
```

### `metrics.json` 주요 필드

```json
{
  "dataset": "acm",
  "backbone": "rgcn",
  "use_topology": true,
  "topology_source": "gtn",
  "seed": 312132,
  "gtn_only_test_macro_f1": 0.xxxx,   // GTN 단독 성능 (Stage 1 결과)
  "gtn_attentions": [...],             // GTN relation 가중치 (시각화용)
  "fusion_beta": [...],                // 채널별 중요도 가중치
  "topo_dim": 75,
  "test_macro_f1": 0.xxxx,
  "test_accuracy": 0.xxxx,
  "val_macro_f1": 0.xxxx
}
```

---

## 7. 비교 분석 방법

실험 완료 후 아래 조합으로 비교합니다.

### 핵심 비교 (위상 기여 측정)

```
(d) RGCN only      →  (f) RGCN + GTN-PDGNN
    mean F1 Δ, std Δ 측정
```

### noisy baseline과 비교 (위상 신호 vs 차원 효과)

```
(e) RGCN + noisy   →  (f) RGCN + GTN-PDGNN
    (f) > (e) 이어야 "위상 신호"가 의미 있다는 주장 성립
```

### HAN과 비교 (backbone 효과)

```
(c) HAN + GTN-PDGNN   vs   (f) RGCN + GTN-PDGNN
    데이터셋별로 어느 backbone이 topology를 더 잘 활용하는가
```

### 데이터셋별 해석

| node feature 강도 | 예상 패턴 |
|-----------------|---------|
| 강 (acm, imdb) | Δ 작음 — feature가 이미 충분 |
| 약 (dblp) | Δ 중간 — topology가 부족한 feature 보완 |
| featureless (freebase, aifb, yelp) | Δ 클 가능성 — topology가 유일한 구조 신호 |

---

## 8. 관련 파일 목록

| 파일 | 역할 |
|------|------|
| `tda/models/rgcn.py` | RGCN 모델 구현 |
| `tda/train.py` | 학습 드라이버 (backbone 분기, viz 연동) |
| `tda/viz.py` | metapath + persistence image 시각화 |
| `experiments/gen_rgcn.py` | (d)(f) config 자동 생성 |
| `experiments/run_f_rgcn.sh` | 실험 (f) 전체 실행 스크립트 |
| `experiments/collect_f_results.py` | 결과 집계 → md 저장 |
| `configs/campaign/{ds}__f_rgcn.json` | 데이터셋별 실험 config (14개) |
| `results/DATASETS.md` | 7개 데이터셋 선택 근거 |
