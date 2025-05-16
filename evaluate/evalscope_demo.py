from evalscope.perf.main import run_perf_benchmark

task_cfg = {"url": "http://my:8000/v1/chat/completions",
            "parallel": 10,
            "model": "Qwen2.5-7B-Instruct",
            "number": 100,
            "api": "openai",
            "dataset": "openqa",
            "stream": True}
run_perf_benchmark(task_cfg)
