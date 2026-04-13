# SAM3 코드 수정 내역

Ctrl-World 온라인 생성 파이프라인에 SAM3 멀티오브젝트 트래킹을 통합하면서
`/home/dgu/minyoung/sam3/` 내부 파일들을 아래와 같이 수정했습니다.

---

## 1. `sam3/model/sam3_video_inference.py`

### 문제
`add_prompt()`가 호출될 때마다 내부에서 `reset_state()`를 호출 →
두 번째 객체를 같은 세션에 등록하면 첫 번째 객체가 초기화됨.
단일 세션 멀티오브젝트 등록이 불가능했던 근본 원인.

### 수정 1: `_get_visual_prompt()` — `force_new` 파라미터 추가
```python
# 변경 전
def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
    is_new_visual_prompt = (
        inference_state["per_frame_visual_prompt"][frame_idx] is None
        and inference_state["previous_stages_out"][frame_idx] is None
    )

# 변경 후
def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels, force_new=False):
    is_new_visual_prompt = force_new or (
        inference_state["per_frame_visual_prompt"][frame_idx] is None
        and inference_state["previous_stages_out"][frame_idx] is None
    )
```

**이유**: `previous_stages_out[frame_idx]`가 이미 세팅된 상태에서 두 번째 객체를
등록할 때, `is_new_visual_prompt=False`가 되어 detector가 suppressed됨.
`force_new=True`로 강제 우회.

### 수정 2: `add_prompt()` — `skip_reset` 파라미터 추가
```python
# 변경 전
self.reset_state(inference_state)

# 변경 후
if not skip_reset:
    self.reset_state(inference_state)
```

`_get_visual_prompt()` 호출부에도 `force_new=skip_reset` 전달.

### 수정 3: `Sam3VideoInferenceWithInstanceInteractivity.add_prompt()` — `skip_reset` 전파
하위 클래스에서 `super().add_prompt()`로 넘길 때 `skip_reset` 누락 → 추가.

---

## 2. `sam3/model/sam3_base_predictor.py`

### 수정 1: `handle_request()` key 이름 통일
`frame_index` → `frame_idx` (내부 API 호출 시 key 불일치 수정)

### 수정 2: `add_prompt()` — `skip_reset` 파라미터 추가 및 전달
```python
def add_prompt(self, ..., skip_reset: bool = False):
    kwargs = dict(
        ...
        skip_reset=skip_reset,
    )
```

`inspect.signature` 필터링을 통해 모델이 `skip_reset`을 지원하는 경우에만 전달.

---

## 3. `sam3/model/sam3_multiplex_video_predictor.py`

### 수정: bf16 autocast 비활성화 (디버깅 목적)
```python
# 변경 전 (원본)
self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
self.bf16_context.__enter__()

# 변경 후 (주석 처리)
# self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# self.bf16_context.__enter__()
```

**이유**: SAM 3.1 multiplex 시도 시 FlashAttention fp32 에러가 발생.
`sam3_manager.py`에서 직접 `torch.autocast` 컨텍스트를 관리하는 방식으로 전환.

---

## 4. `sam3/model_builder.py`

### 수정 1: `build_sam3_multiplex_video_model()` — `use_rope_real` 기본값 변경
```python
# 변경 전
use_rope_real: bool = False

# 변경 후
use_rope_real: bool = True
```

### 수정 2: `build_sam3_predictor()` — `use_fa3` 기본값 변경
```python
# 변경 전
use_fa3: bool = True

# 변경 후
use_fa3: bool = False
```

**이유**: Ampere GPU (A100 등) 환경에서 FlashAttention 3가 fp32 텐서를 받으면
에러 발생. `use_fa3=False`로 일반 attention fallback.

### 수정 3: bf16 캐스팅 시도 코드 (주석 처리 상태로 보존)
```python
# demo_model = demo_model.cuda().eval().to(torch.bfloat16)
```

---

## 현재 상태 요약

| 파일 | 상태 |
|------|------|
| `sam3_video_inference.py` | `skip_reset` + `force_new` 패치 적용 |
| `sam3_base_predictor.py` | `skip_reset` 전달 + key 이름 수정 |
| `sam3_multiplex_video_predictor.py` | bf16 autocast 비활성화 |
| `sam3_model_builder.py` | `use_fa3=False`, `use_rope_real=True` |

## `sam3_manager.py` 현재 전략
- `build_sam3_video_predictor` (SAM3 기본) 사용
- `initialize()`: 객체마다 독립 세션 + text prompt → anchor box 탐지
- `update_chunk()`: 객체별 독립 세션 + anchor box prompt → propagate → merge
- `torch.autocast(bfloat16)` 래퍼를 각 모델 호출부에 명시적으로 적용
