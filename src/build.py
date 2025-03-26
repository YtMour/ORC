#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打包脚本，用于将应用打包为独立可执行文件
"""

import os
import sys
import shutil
import PyInstaller.__main__
import paddle
import paddleocr


def get_paddle_model_path():
    """获取PaddleOCR模型路径"""
    # paddleocr package的路径
    package_path = os.path.dirname(paddleocr.__file__)
    # 模型存储路径
    model_path = os.path.join(os.path.expanduser('~'), '.paddleocr')
    
    return model_path


def get_paddle_path():
    """获取Paddle库路径"""
    # paddle package的路径
    package_path = os.path.dirname(paddle.__file__)
    return package_path


def main():
    """主打包函数"""
    # 源码路径
    src_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(src_path)
    
    # 输出路径
    dist_path = os.path.join(base_path, 'dist')
    build_path = os.path.join(base_path, 'build')
    
    # 清理之前的构建
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    
    # 获取模型路径
    paddle_path = get_paddle_path()
    model_path = get_paddle_model_path()
    
    print(f"Paddle路径: {paddle_path}")
    print(f"模型路径: {model_path}")
    
    # 构建打包参数
    paddle_data = []
    
    # 添加OCR模型文件
    if os.path.exists(model_path):
        paddle_data.append(f"{model_path}:inference")
    
    # 打包命令
    package_args = [
        os.path.join(src_path, 'main.py'),  # 主程序路径
        '--name=图片方向矫正与PDF转换工具',  # 应用名称
        '--noconsole',  # 不显示控制台
        '--onefile',  # 打包为单个文件
        f'--distpath={dist_path}',  # 输出目录
        f'--workpath={build_path}',  # 工作目录
        '--clean',  # 清理临时文件
    ]
    
    # 添加数据文件
    for data in paddle_data:
        package_args.append(f'--add-data={data}')
    
    # 执行打包
    print("开始打包...")
    PyInstaller.__main__.run(package_args)
    print(f"打包完成，输出文件: {dist_path}")


if __name__ == '__main__':
    main() 