"""兼容入口：与规范中的 test.py 等价，便于沿用 baseline 文档中的 predict.py 命令。"""

from test import *  # noqa: F401,F403
from test import main

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
