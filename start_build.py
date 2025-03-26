#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用PyInstaller直接打包应用程序的Python脚本
运行方式: python start_build.py
"""

import os
import sys
import subprocess

def main():
    print("开始使用PyInstaller打包应用程序...")
    
    # 基本路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(base_path, "start_build.spec")
    
    # 检查图标文件
    icon_path = os.path.join(base_path, "icon.ico")
    if not os.path.exists(icon_path):
        print(f"错误: 图标文件 {icon_path} 不存在!")
        return
    
    if os.path.exists(spec_path):
        # 如果spec文件存在，直接使用它
        cmd = ["pyinstaller", "--clean", spec_path]
    else:
        # 构建命令
        cmd = [
            "pyinstaller",
            "--name=图片方向矫正与PDF转换工具",  # 使用正确的名称
            "--windowed",  # 无控制台
            "--clean",     # 清理临时文件
            "--log-level=DEBUG",
            "--noupx",     # 不使用UPX压缩
            "--icon=" + icon_path,  # 添加图标
        ]
        
        # 添加数据文件
        data_files = [
            "--add-data=src" + os.pathsep + "src",
            "--add-data=temp" + os.pathsep + "temp",
            "--add-data=debug_images" + os.pathsep + "debug_images"
        ]
        cmd.extend(data_files)
        
        # 添加隐含导入 - 不包含paddle相关
        hidden_imports = [
            "--hidden-import=PyQt5",
            "--hidden-import=PyQt5.QtCore",
            "--hidden-import=PyQt5.QtGui",
            "--hidden-import=PyQt5.QtWidgets",
            "--hidden-import=cv2",
            "--hidden-import=PIL",
            "--hidden-import=numpy"
        ]
        cmd.extend(hidden_imports)
        
        # 排除模块 - 明确排除paddle相关
        excludes = [
            "--exclude-module=paddleocr",
            "--exclude-module=paddle",
            "--exclude-module=matplotlib",
            "--exclude-module=scipy",
            "--exclude-module=tensorflow",
            "--exclude-module=torch",
            "--exclude-module=pandas"
        ]
        cmd.extend(excludes)
        
        # 添加主脚本
        cmd.append(os.path.join(base_path, "start.py"))
    
    # 显示命令
    print("执行命令: " + " ".join(cmd))
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        print("打包完成，请检查dist目录。")
        
        # 如果生成了新的spec文件，重命名它
        if not os.path.exists(spec_path):
            generated_spec = os.path.join(base_path, "start_build.spec")
            if os.path.exists(generated_spec):
                print("重命名spec文件为start_build.spec...")
                os.rename(generated_spec, spec_path)
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")

if __name__ == "__main__":
    main() 