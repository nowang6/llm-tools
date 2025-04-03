from evalscope.perf.main import run_perf_benchmark

task_cfg = {"url": "http://my:30000/v1/chat/completions",
            "parallel": 10,
            "model": "qwen2.5",
            "number": 100,
            "api": "openai",
            "dataset": "openqa",
            "stream": True}
run_perf_benchmark(task_cfg)