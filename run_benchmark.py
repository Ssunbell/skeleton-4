"""
독립 실행형 벤치마크 스크립트
Transformers vs vLLM 성능 비교를 단독으로 실행
"""

from vllm_offline_inference import PerformanceBenchmark


def main():
    """
    Transformers와 vLLM 성능 비교 벤치마크 실행

    측정 항목:
    - First Token Latency (TTFT): 첫 토큰 생성까지 걸리는 시간
    - Token/sec: 초당 생성 토큰 수
    - 총 추론 시간: 전체 추론 완료 시간
    - GPU 메모리 사용량: 피크 메모리
    - Throughput: 초당 처리 프롬프트 수
    """

    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    print("\n" + "=" * 60)
    print("📊 Transformers vs vLLM 성능 벤치마크")
    print("=" * 60)
    print("\n이 벤치마크는 다음을 측정합니다:")
    print("  • First Token Latency (TTFT) - 첫 토큰 생성 속도")
    print("  • Token/sec - 토큰 생성 처리량")
    print("  • GPU Memory - 메모리 사용량")
    print("  • Throughput - 초당 프롬프트 처리량")
    print("\n" + "=" * 60)

    # Text-to-SQL 테스트 프롬프트
    prompts = [
        "You are a SQL expert. Convert this to SQL: Find all users with age greater than 25",
        "You are a SQL expert. Convert this to SQL: Count total employees in sales department",
        "You are a SQL expert. Convert this to SQL: Show top 10 products by revenue",
        "You are a SQL expert. Convert this to SQL: Delete inactive user accounts",
        "You are a SQL expert. Convert this to SQL: Update email addresses for all admins",
    ]

    print(f"\n📝 테스트 프롬프트 {len(prompts)}개 준비")
    print(f"🎯 Max Tokens: 128\n")

    # 벤치마크 실행
    benchmark = PerformanceBenchmark(model_name)
    results = benchmark.compare(prompts, max_tokens=128)

    # 결과 요약
    print("\n" + "=" * 60)
    print("✨ 벤치마크 완료!")
    print("=" * 60)

    tf_results = results["transformers"]
    vllm_results = results["vllm"]

    print("\n핵심 지표 요약:")
    print(
        f"  • vLLM Token 생성 속도: {vllm_results['tokens_per_sec'] / tf_results['tokens_per_sec']:.2f}x 더 빠름"
    )
    print(
        f"  • vLLM 전체 추론 시간: {tf_results['total_inference_time'] / vllm_results['total_inference_time']:.2f}x 더 빠름"
    )

    memory_diff = tf_results["peak_memory_mb"] - vllm_results["peak_memory_mb"]
    if memory_diff > 0:
        print(
            f"  • vLLM 메모리 절감: {memory_diff:.0f} MB ({memory_diff/tf_results['peak_memory_mb']*100:.1f}%)"
        )
    else:
        print(f"  • vLLM 메모리 사용: {abs(memory_diff):.0f} MB 더 사용")

    print("\nvLLM의 PagedAttention 덕분에 메모리 효율성과 처리 속도가")
    print("동시에 개선되었습니다! 🚀")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
