#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import shutil
import logging
from datetime import datetime
import time
import random


# 临时目录路径
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'temp')


def get_temp_path(prefix="temp_", suffix=".jpg"):
    """生成临时文件路径"""
    # 使用时间戳和随机数生成唯一文件名
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳
    random_suffix = str(random.randint(1000, 9999))
    
    # 使用英文文件名
    temp_filename = f"{prefix}{timestamp}{random_suffix}{suffix}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    # 确保临时目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    return temp_path


def cleanup_temp_files():
    """清理临时文件"""
    try:
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                if filename.startswith("temp_"):
                    file_path = os.path.join(TEMP_DIR, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"清理临时文件失败: {str(e)}")
    except Exception as e:
        print(f"清理临时目录失败: {str(e)}") 