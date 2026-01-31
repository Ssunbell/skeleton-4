# vLLM과 PagedAttention

vLLM은 LLM(Large Language Model) 추론을 위한 고성능 라이브러리로, PagedAttention 알고리즘을 통해 메모리 효율성을 극대화하고 처리량을 향상시킵니다. 이 실습에서는 vLLM의 핵심 개념을 이해하고, LoRA adapter를 통합하여 Text-to-SQL task 추론을 수행하며, 일반 Transformers 라이브러리 대비 성능을 비교 분석합니다.

## 프로젝트 목표

1. 실습명 : vLLM을 이용한 효율적인 LLM 추론과 성능 벤치마킹
2. 핵심 주제:
    1. vLLM의 PagedAttention과 KV Cache 개념 이해
    2. LoRA adapter 통합 방법 (Runtime LoRA vs Merged Model)
    3. Transformers vs vLLM 성능 비교 (속도, 메모리, 처리량)
    4. Text-to-SQL task를 위한 실전 추론 시스템 구축
3. 학습 목표 :
    1. vLLM의 PagedAttention 메커니즘을 이해하고 메모리 효율성의 원리를 설명할 수 있다.
    2. LoRA adapter를 vLLM에 통합하여 추론을 수행할 수 있다.
    3. 성능 벤치마크를 통해 vLLM의 장점을 정량적으로 분석할 수 있다.
4. 학습 개념: 키워드명 :
    1. PagedAttention
    2. KV Cache
    3. vLLM
    4. LoRA adapter
    5. Performance Benchmarking
5. 학습 방향 :
  - vLLM의 핵심 기술인 PagedAttention이 어떻게 메모리를 효율적으로 관리하는지 시뮬레이터를 통해 체험합니다.
  - 실습 코드는 단계별로 구성되어 있으며, Jupyter notebook을 통해 이론과 실습을 병행합니다.
  - 성능 벤치마킹을 통해 일반 추론 방식과 vLLM의 차이를 직접 확인하고 분석합니다.

## 사전 요구사항

-   **Python**: 3.13 이상
-   **GPU**: CUDA 12.9.1 GPU (최소 6GB VRAM 권장)
-   **RAM**: 최소 8GB
-   **저장공간**: 최소 20GB (모델 다운로드 포함)
-   **운영체제**: Linux(Windows는 wsl)

## 프로젝트 구조

```text
.
├── vllm.ipynb                      # PagedAttention/KV Cache 개념 학습
├── pyproject.toml                  # 의존 패키지 버전 명시
└── README.md                       # 프로젝트 문서
```

## 시작하기

### 1. 환경 설정

1. 윈도우에서 WSL을 설치해야 합니다. 설치 과정이 매우 복잡하므로 [링크](https://www.notion.so/ssunbell/WSL-2ec1806f5bc1804b8648d6b30450613a?source=copy_link)를 참고해서 차근차근 진행해주세요. 

2. uv가 설치되어 있어야 합니다. uv를 설치하는 방법은 아래를 참고하세요.
   **Linux/macOS:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. `uv sync` 명령어로 가상환경을 설치 및 라이브러리 설치를 진행합니다. uv sync 명령어 이후 `.venv` 폴더가 생성된 것을 확인하실 수 있습니다.
```bash
uv sync
```

- 만약 이슈가 있다면 아래 명령어를 참고해서 라이브러리 설치를 진행해주세요.
```bash
uv add numpy==2.1.0 pandas==2.2.3 peft==0.17.1 torch==2.8.0 transformers==4.56.0 accelerate==1.10.1 datasets==4.0.0 trl==0.22.2 juypter==1.1.1 ipykernel==7.1.0 vllm==0.11.0
```

4. `uv run python vllm_offline_inference.py` 명령어 등 uv의 명령어를 사용하는 것이 익숙하지 않을 수 있습니다. 가상환경을 활성화하면 일반적인 `python vllm_offline_inference.py` 명령어로 코드를 동작시킬 수 있습니다.

   **WSL에서 가상환경 활성화 방법:**
   ```bash
   source .venv/bin/activate
   ```

### 2. 학습 및 실습

#### 2.1 Jupyter Notebook 실습

학습은 `vllm.ipynb` 파일을 이용하여 진행합니다. 각 셀마다 상세한 설명이 포함되어 있으며, 다음 내용을 다룹니다:

- **Part 1**: vLLM 기초 개념 (PagedAttention, KV Cache)
- **Part 1.5**: PagedAttention 시뮬레이터 실습
- **Part 2**: vLLM 기본 사용법
- **Part 2.5**: KV Cache 성능 비교 시뮬레이터
- **Part 3**: Chat 형식 프롬프트 처리
- **Part 4**: LoRA Adapter 사용하기
- **Part 5**: 성능 벤치마크 (Transformers vs vLLM)
- **Part 6**: 종합 실습 (Text-to-SQL 시스템 구축)


### 3. LoRA Adapter 통합 방법

vLLM에서 LoRA adapter를 사용하는 두 가지 방법을 제공합니다:

#### 방법 1: Runtime LoRA (동적 적용)

vLLM이 추론 시점에 LoRA를 동적으로 적용합니다.

**장점:**
- 여러 LoRA adapter를 동시에 사용 가능
- Adapter 교체가 유연함

**사용 예시:**
```python
from vllm_offline_inference import VLLMInferenceWithLoRA

inferencer = VLLMInferenceWithLoRA(
    base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    lora_adapter_path="./checkpoint-100",  # LoRA adapter 경로
)

messages = [
    {'content': 'You are a helpful assistant.', 'role': 'system'},
    {'content': 'Hello!', 'role': 'user'}
]

result = inferencer.generate(messages)
```

#### 방법 2: Merged Model (사전 통합) - 권장

LoRA adapter를 base model에 미리 merge한 후 추론합니다.

**장점:**
- 추론 속도가 더 빠름
- 메모리 오버헤드 없음
- 배포에 용이함

**사용 예시:**
```python
from vllm_offline_inference import merge_lora_to_base_model, VLLMInferenceWithLoRA

# Step 1: LoRA를 base model에 merge (한 번만 실행)
merged_path = merge_lora_to_base_model(
    base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    lora_adapter_path="./checkpoint-100",
    output_path="./merged_model",
)

# Step 2: Merge된 모델로 추론
inferencer = VLLMInferenceWithLoRA(
    base_model_name=merged_path,
    lora_adapter_path=None,  # 이미 merge되었으므로 None
)
```

### 4. 성능 벤치마크

Transformers 라이브러리와 vLLM의 성능을 비교합니다.

**측정 항목:**
- First Token Latency (첫 토큰 생성 시간)
- Tokens per Second (초당 생성 토큰 수)
- Total Inference Time (전체 추론 시간)
- Peak GPU Memory (최대 GPU 메모리 사용량)
- Throughput (처리량)


**예상 결과:**
- vLLM이 Transformers 대비 **2-5배 빠른 추론 속도**
- **메모리 효율성 향상** (PagedAttention 덕분)
- **높은 처리량** (배치 처리 최적화)

## 주요 기능

- **PagedAttention**: OS의 가상 메모리처럼 KV Cache를 블록 단위로 관리
- **LoRA 지원**: Runtime LoRA 및 Merged Model 방식 모두 지원
- **Chat 포맷팅**: 자동 chat template 적용 (없으면 수동 포맷팅)
- **배치 추론**: 여러 프롬프트를 동시에 처리
- **성능 벤치마킹**: Transformers vs vLLM 정량적 비교

## 파라미터 설정

### Sampling 파라미터
- `max_tokens`: 최대 생성 토큰 수 (기본: 512)
- `temperature`: 샘플링 온도, 낮을수록 결정적 (기본: 0.7)
- `top_p`: Nucleus sampling (기본: 0.9)
- `top_k`: Top-k sampling (기본: 50)
- `repetition_penalty`: 반복 방지 (기본: 1.0)

### GPU 메모리 설정
```python
inferencer = VLLMInferenceWithLoRA(
    base_model_name="...",
    gpu_memory_utilization=0.75,  # GPU 메모리 75% 사용
    tensor_parallel_size=1,      # 단일 GPU
)
```

멀티 GPU 사용 시 `tensor_parallel_size`를 GPU 개수로 설정하세요.

## 문제 해결

-   **CUDA 메모리 부족**: `gpu_memory_utilization` 값을 낮추거나, 더 작은 모델을 사용하세요.

-   **uv 명령어 인식 안 됨**: ubuntu22.04에서 uv를 설치하세요.:
    
    ubuntu에서 bash를 사용할 경우 `curl -LsSf https://astral.sh/uv/install.sh | sh`를 사용해주세요.

-   **vLLM 버전 호환성**: vLLM 0.13.0 버전을 사용하세요. 다른 버전에서는 API가 다를 수 있습니다.

-   **LoRA adapter 로드 실패**: adapter 폴더 구조를 확인하세요:
    ```
    checkpoint-100/
    ├── adapter_config.json
    └── adapter_model.safetensors (또는 .bin)
    ```

## 참고 문헌
-   **[vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)** – vLLM GitHub.
-   **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** - vLLM 논문.
-   **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** - LoRA 논문.
-   **[PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)** – Hugging Face.
