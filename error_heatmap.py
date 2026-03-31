"""兼容转发入口，实际实现位于 legacy/error_heatmap.py。"""

if __name__ == "__main__":
    from legacy.error_heatmap import parse_args_and_run

    parse_args_and_run()
else:
    from legacy.error_heatmap import *  # noqa: F401,F403
