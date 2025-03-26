#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片方向矫正与PDF转换工具
启动脚本
"""

import os
import sys
import warnings
import traceback

# 将项目根目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 显示一些环境信息
print("Python版本:", sys.version)
print("当前工作目录:", os.getcwd())
print("脚本目录:", current_dir)

# 忽略不重要的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 导入主函数
try:
    from src.main import main
    main()
except Exception as e:
    print("\n程序启动时出错:", e)
    print("\n详细错误信息:")
    traceback.print_exc()
    input("\n按Enter键退出...") 