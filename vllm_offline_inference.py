"""
vLLM Offline Inference with LoRA Adapter
Model: naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
"""

import os
import time
import torch
from typing import List, Dict, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def merge_lora_to_base_model(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
    save_tokenizer: bool = True,
) -> str:
    """
    LoRA adapter를 base model에 merge하여 통합 모델 생성

    Args:
        base_model_name: 기본 모델 경로 또는 이름
        lora_adapter_path: LoRA adapter 경로
        output_path: merge된 모델 저장 경로
        save_tokenizer: tokenizer도 함께 저장할지 여부

    Returns:
        저장된 모델 경로
    """
    print(f"🔄 LoRA merge 시작...")
    print(f"  Base Model: {base_model_name}")
    print(f"  LoRA Adapter: {lora_adapter_path}")

    # Base model 로드
    print("  1) Base model 로딩 중...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    # LoRA adapter 로드
    print("  2) LoRA adapter 로딩 중...")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge 수행
    print("  3) LoRA weights를 base model에 merge 중...")
    merged_model = model_with_lora.merge_and_unload()

    # 저장
    print(f"  4) Merged model 저장 중: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    if save_tokenizer:
        print("  5) Tokenizer 저장 중...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(output_path)

    print(f"✅ LoRA merge 완료! 저장 위치: {output_path}\n")
    return output_path


class VLLMInferenceWithLoRA:
    def __init__(
        self,
        base_model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        lora_adapter_path: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        vLLM 오프라인 추론 초기화

        Args:
            base_model_name: 기본 모델 이름
            lora_adapter_path: LoRA adapter 경로 (merge할 경우)
            tensor_parallel_size: GPU 병렬화 크기
            gpu_memory_utilization: GPU 메모리 사용률
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # vLLM 모델 로드
        if lora_adapter_path:
            # LoRA adapter와 함께 로드 (vLLM이 자동으로 merge)
            self.llm = LLM(
                model=base_model_name,
                enable_lora=True,
                max_lora_rank=64,  # LoRA rank 설정 (adapter에 맞게 조정)
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
        else:
            # 기본 모델만 로드
            self.llm = LLM(
                model=base_model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )

    def format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat 메시지를 모델이 이해할 수 있는 형식으로 변환

        Args:
            messages: [{'role': 'system/user/assistant', 'content': '...'}, ...]

        Returns:
            포맷팅된 프롬프트 문자열
        """
        if self.tokenizer.chat_template:
            # Tokenizer에 chat_template이 있으면 사용
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # 없으면 수동으로 포맷팅
            formatted_prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    formatted_prompt += f"System: {content}\n\n"
                elif role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted_prompt += f"Assistant: {content}\n\n"

            # 새로운 응답 생성을 위한 프롬프트 추가
            if messages[-1]["role"] != "assistant":
                formatted_prompt += "Assistant: "

        return formatted_prompt

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: List[str] = None,
    ) -> str:
        """
        텍스트 생성

        Args:
            messages: Chat 형식의 메시지 리스트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: Nucleus sampling 파라미터
            top_k: Top-k sampling 파라미터
            repetition_penalty: 반복 페널티
            stop_tokens: 생성 중단 토큰 리스트

        Returns:
            생성된 텍스트
        """
        # 메시지 포맷팅
        prompt = self.format_chat_messages(messages)

        # Sampling 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop_tokens if stop_tokens else [],
        )

        # LoRA adapter 사용 시
        if self.lora_adapter_path:
            # vLLM에서 LoRA request 생성
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("lora_adapter", 1, self.lora_adapter_path)

            outputs = self.llm.generate(
                [prompt], sampling_params, lora_request=lora_request
            )
        else:
            outputs = self.llm.generate([prompt], sampling_params)

        # 결과 추출
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def batch_generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> List[str]:
        """
        배치 추론

        Args:
            messages_list: Chat 메시지 리스트의 리스트
            기타 파라미터: generate()와 동일

        Returns:
            생성된 텍스트 리스트
        """
        prompts = [self.format_chat_messages(msgs) for msgs in messages_list]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if self.lora_adapter_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("lora_adapter", 1, self.lora_adapter_path)
            outputs = self.llm.generate(
                prompts, sampling_params, lora_request=lora_request
            )
        else:
            outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]


class PerformanceBenchmark:
    """Transformers vs vLLM 성능 비교 벤치마크"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_gpu_memory_mb(self) -> float:
        """현재 GPU 메모리 사용량 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def benchmark_transformers(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
    ) -> Dict[str, float]:
        """
        Transformers 라이브러리로 추론 성능 측정

        Returns:
            성능 메트릭 딕셔너리
        """
        print("\n" + "=" * 60)
        print("🔵 Transformers 벤치마크")
        print("=" * 60)

        # 모델 로드
        print("  모델 로딩 중...")
        start_load = time.time()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        load_time = time.time() - start_load
        memory_after_load = self.get_gpu_memory_mb()

        print(f"  ✅ 로딩 완료: {load_time:.2f}s, {memory_after_load:.0f} MB")

        # 추론 측정
        print(f"\n  추론 시작 ({len(prompts)}개 프롬프트)...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_inference_time = 0
        total_tokens = 0
        first_token_latencies = []

        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs.input_ids

            # First token 측정
            start_first = time.time()

            with torch.no_grad():
                # 첫 토큰 생성
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            first_token_latency = time.time() - start_first
            first_token_latencies.append(first_token_latency)

            # 전체 생성 (autoregressive)
            start_full = time.time()

            generated_ids = input_ids
            generated_count = 0

            for _ in range(max_new_tokens - 1):
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                generated_count += 1

                with torch.no_grad():
                    outputs = model(generated_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            inference_time = time.time() - start_full
            total_inference_time += inference_time
            total_tokens += generated_count

            if i == 0:
                decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"\n  [샘플 출력] {decoded[:100]}...")

        peak_memory = self.get_gpu_memory_mb()
        avg_first_token_latency = sum(first_token_latencies) / len(
            first_token_latencies
        )
        tokens_per_sec = (
            total_tokens / total_inference_time if total_inference_time > 0 else 0
        )

        # 결과 출력
        print(f"\n  ✅ 추론 완료")
        print(f"  📊 First Token Latency: {avg_first_token_latency*1000:.2f}ms (평균)")
        print(f"  📊 Token/sec: {tokens_per_sec:.2f}")
        print(f"  📊 총 추론 시간: {total_inference_time:.2f}s")
        print(f"  📊 피크 GPU 메모리: {peak_memory:.0f} MB")

        # 정리
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "load_time": load_time,
            "total_inference_time": total_inference_time,
            "avg_first_token_latency_ms": avg_first_token_latency * 1000,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": total_tokens,
            "peak_memory_mb": peak_memory,
            "throughput_prompts_per_sec": len(prompts) / total_inference_time,
        }

    def benchmark_vllm(
        self,
        prompts: List[str],
        max_tokens: int = 128,
    ) -> Dict[str, float]:
        """
        vLLM으로 추론 성능 측정

        Returns:
            성능 메트릭 딕셔너리
        """
        print("\n" + "=" * 60)
        print("🟢 vLLM 벤치마크")
        print("=" * 60)

        # 모델 로드
        print("  모델 로딩 중...")
        start_load = time.time()

        llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

        load_time = time.time() - start_load
        memory_after_load = self.get_gpu_memory_mb()

        print(f"  ✅ 로딩 완료: {load_time:.2f}s, {memory_after_load:.0f} MB")

        # 추론 측정
        print(f"\n  추론 시작 ({len(prompts)}개 프롬프트)...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # greedy decoding for fair comparison
        )

        # First token latency 측정을 위해 단일 프롬프트 먼저 실행
        start_first = time.time()
        single_output = llm.generate([prompts[0]], sampling_params)
        first_token_latency = time.time() - start_first

        # 전체 배치 추론
        start_inference = time.time()
        outputs = llm.generate(prompts, sampling_params)
        total_inference_time = time.time() - start_inference

        peak_memory = self.get_gpu_memory_mb()

        # 토큰 통계
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_sec = (
            total_tokens / total_inference_time if total_inference_time > 0 else 0
        )

        # 샘플 출력
        if outputs:
            print(f"\n  [샘플 출력] {outputs[0].outputs[0].text[:100]}...")

        # 결과 출력
        print(f"\n  ✅ 추론 완료")
        print(f"  📊 First Token Latency: {first_token_latency*1000:.2f}ms")
        print(f"  📊 Token/sec: {tokens_per_sec:.2f}")
        print(f"  📊 총 추론 시간: {total_inference_time:.2f}s")
        print(f"  📊 피크 GPU 메모리: {peak_memory:.0f} MB")

        return {
            "load_time": load_time,
            "total_inference_time": total_inference_time,
            "first_token_latency_ms": first_token_latency * 1000,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": total_tokens,
            "peak_memory_mb": peak_memory,
            "throughput_prompts_per_sec": len(prompts) / total_inference_time,
        }

    def compare(self, prompts: List[str], max_tokens: int = 128):
        """
        Transformers와 vLLM 성능 비교
        """
        print("\n" + "=" * 60)
        print("🚀 Transformers vs vLLM 성능 비교")
        print("=" * 60)
        print(f"모델: {self.model_name}")
        print(f"프롬프트 수: {len(prompts)}")
        print(f"Max Tokens: {max_tokens}")

        # Transformers 벤치마크
        tf_results = self.benchmark_transformers(prompts, max_tokens)

        # 메모리 정리 대기
        time.sleep(3)

        # vLLM 벤치마크
        vllm_results = self.benchmark_vllm(prompts, max_tokens)

        # 비교 결과 출력
        print("\n" + "=" * 60)
        print("📊 최종 비교 결과")
        print("=" * 60)

        print("\n⚡ First Token Latency (낮을수록 좋음):")
        tf_ttft = tf_results.get("avg_first_token_latency_ms", 0)
        vllm_ttft = vllm_results.get("first_token_latency_ms", 0)
        print(f"  Transformers: {tf_ttft:.2f}ms")
        print(f"  vLLM:         {vllm_ttft:.2f}ms")
        if vllm_ttft > 0:
            improvement = ((tf_ttft - vllm_ttft) / tf_ttft) * 100
            print(
                f"  {'🚀 개선' if improvement > 0 else '⚠️ 차이'}: {abs(improvement):.1f}%"
            )

        print("\n🔥 Token/sec (높을수록 좋음):")
        print(f"  Transformers: {tf_results['tokens_per_sec']:.2f} tokens/sec")
        print(f"  vLLM:         {vllm_results['tokens_per_sec']:.2f} tokens/sec")
        speedup = (
            vllm_results["tokens_per_sec"] / tf_results["tokens_per_sec"]
            if tf_results["tokens_per_sec"] > 0
            else 0
        )
        print(f"  🚀 vLLM 향상: {speedup:.2f}x")

        print("\n⏱️ 총 추론 시간:")
        print(f"  Transformers: {tf_results['total_inference_time']:.2f}s")
        print(f"  vLLM:         {vllm_results['total_inference_time']:.2f}s")
        time_speedup = (
            tf_results["total_inference_time"] / vllm_results["total_inference_time"]
        )
        print(f"  🚀 속도 향상: {time_speedup:.2f}x")

        print("\n💾 피크 GPU 메모리:")
        print(f"  Transformers: {tf_results['peak_memory_mb']:.0f} MB")
        print(f"  vLLM:         {vllm_results['peak_memory_mb']:.0f} MB")
        memory_diff = tf_results["peak_memory_mb"] - vllm_results["peak_memory_mb"]
        memory_saving_pct = (
            (memory_diff / tf_results["peak_memory_mb"]) * 100
            if tf_results["peak_memory_mb"] > 0
            else 0
        )
        print(
            f"  💡 메모리 {'절감' if memory_diff > 0 else '증가'}: {abs(memory_diff):.0f} MB ({abs(memory_saving_pct):.1f}%)"
        )

        print("\n🎯 처리량 (Throughput):")
        print(
            f"  Transformers: {tf_results['throughput_prompts_per_sec']:.2f} prompts/sec"
        )
        print(
            f"  vLLM:         {vllm_results['throughput_prompts_per_sec']:.2f} prompts/sec"
        )

        print("\n" + "=" * 60)

        return {
            "transformers": tf_results,
            "vllm": vllm_results,
        }


def run_benchmark():
    """성능 벤치마크 실행"""
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    # 테스트 프롬프트
    prompts = [
        "You are a SQL expert. Convert this to SQL: Find all users with age greater than 25",
        "You are a SQL expert. Convert this to SQL: Count total employees in sales department",
        "You are a SQL expert. Convert this to SQL: Show top 10 products by revenue",
        "You are a SQL expert. Convert this to SQL: Delete inactive user accounts",
        "You are a SQL expert. Convert this to SQL: Update email addresses for all admins",
    ]

    benchmark = PerformanceBenchmark(model_name)
    results = benchmark.compare(prompts, max_tokens=128)

    return results


def main_with_merged_model():
    """
    방법 1: LoRA를 미리 merge한 모델로 추론
    - LoRA adapter를 base model에 merge하여 통합 모델 생성
    - vLLM에서 merge된 모델을 직접 로드
    """
    base_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    lora_adapter_path = "./lora_adapter"
    merged_model_path = "./merged_model"

    # Step 1: LoRA를 base model에 merge (한 번만 실행)
    if not os.path.exists(merged_model_path):
        if os.path.exists(lora_adapter_path):
            print("=" * 60)
            print("Step 1: LoRA를 Base Model에 Merge")
            print("=" * 60)
            merge_lora_to_base_model(
                base_model_name=base_model_name,
                lora_adapter_path=lora_adapter_path,
                output_path=merged_model_path,
                save_tokenizer=True,
            )
        else:
            print(f"⚠️  LoRA adapter를 찾을 수 없습니다: {lora_adapter_path}")
            print("   Base model만 사용합니다.\n")
            merged_model_path = base_model_name
    else:
        print(f"✅ 이미 merge된 모델이 존재합니다: {merged_model_path}\n")

    # Step 2: Merge된 모델로 vLLM 추론
    print("=" * 60)
    print("Step 2: Merged Model로 vLLM 추론")
    print("=" * 60)
    print("🚀 vLLM 모델 로딩 중...")

    # Merge된 모델을 base model처럼 직접 로드 (LoRA path 없이)
    inferencer = VLLMInferenceWithLoRA(
        base_model_name=merged_model_path,  # merge된 모델 경로 사용
        lora_adapter_path=None,  # LoRA는 이미 merge되었으므로 None
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    print("✅ 모델 로딩 완료!\n")

    # 추론 실행
    messages = [
        {
            "content": "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query.",
            "role": "system",
        },
        {
            "content": "Given the <USER_QUERY>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<USER_QUERY>\nHow many heads of the departments are older than 56 ?\n</USER_QUERY>",
            "role": "user",
        },
    ]

    print("📝 입력 프롬프트:")
    for msg in messages:
        print(f"  [{msg['role'].upper()}] {msg['content'][:100]}...")
    print()

    print("🤖 추론 시작...")
    generated_text = inferencer.generate(
        messages=messages,
        max_tokens=256,
        temperature=0.1,
        top_p=0.95,
    )

    print("✨ 생성 결과:")
    print(generated_text)
    print()


def main():
    """
    방법 2: vLLM의 런타임 LoRA 기능 사용
    - vLLM이 추론 시점에 LoRA를 동적으로 적용
    """
    # LoRA adapter 경로 (실제 경로로 변경 필요)
    # None으로 설정하면 기본 모델만 사용
    lora_adapter_path = "./lora_adapter"  # 또는 None

    # 모델 초기화
    print("=" * 60)
    print("방법 2: vLLM Runtime LoRA")
    print("=" * 60)
    print("🚀 vLLM 모델 로딩 중...")
    inferencer = VLLMInferenceWithLoRA(
        base_model_name="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        lora_adapter_path=(
            lora_adapter_path if os.path.exists(lora_adapter_path or "") else None
        ),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    print("✅ 모델 로딩 완료!\n")

    # 예제 프롬프트 (Text-to-SQL)
    messages = [
        {
            "content": "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query.",
            "role": "system",
        },
        {
            "content": "Given the <USER_QUERY>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<USER_QUERY>\nHow many heads of the departments are older than 56 ?\n</USER_QUERY>",
            "role": "user",
        },
        {"content": "SELECT count(*) FROM head WHERE age  >  56", "role": "assistant"},
    ]

    print("📝 입력 프롬프트:")
    for msg in messages:
        print(f"  [{msg['role'].upper()}] {msg['content'][:100]}...")
    print()

    # 추론 실행
    print("🤖 추론 시작...")
    generated_text = inferencer.generate(
        messages=messages,
        max_tokens=256,
        temperature=0.1,  # SQL 생성이므로 낮은 temperature 사용
        top_p=0.95,
    )

    print("✨ 생성 결과:")
    print(generated_text)
    print()

    # 추가 예제: 새로운 질의
    new_query_messages = [
        {
            "content": "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query.",
            "role": "system",
        },
        {
            "content": "Given the <USER_QUERY>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<USER_QUERY>\nWhat are the names of all employees in the Sales department?\n</USER_QUERY>",
            "role": "user",
        },
    ]

    print("📝 새로운 질의:")
    print(f"  [USER] {new_query_messages[1]['content'][:100]}...")
    print()

    print("🤖 추론 시작...")
    new_generated_text = inferencer.generate(
        messages=new_query_messages,
        max_tokens=256,
        temperature=0.1,
    )

    print("✨ 생성 결과:")
    print(new_generated_text)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("vLLM Offline Inference with LoRA")
    print("=" * 60)
    print("\n실행 모드를 선택하세요:")
    print("  1) Merged Model 방식 - LoRA를 미리 merge한 모델 사용")
    print("  2) Runtime LoRA 방식 - vLLM이 추론 시 LoRA 동적 적용")
    print("  3) 🚀 성능 벤치마크 - Transformers vs vLLM 비교 (권장)")
    print("\n")

    # 커맨드 라인 인자가 있으면 사용
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("선택 (1, 2, 또는 3, 기본값=3): ").strip() or "3"

    print("\n")

    if choice == "1":
        main_with_merged_model()
    elif choice == "2":
        main()
    elif choice == "3":
        run_benchmark()
    else:
        print("⚠️  잘못된 선택입니다. 벤치마크를 실행합니다.")
        run_benchmark()
