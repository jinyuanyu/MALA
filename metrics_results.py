"""兼容转发入口，实际实现位于 legacy/metrics_results.py。"""

if __name__ == "__main__":
    from legacy.metrics_results import parse_args_and_run

    parse_args_and_run()
else:
    from legacy.metrics_results import *  # noqa: F401,F403
