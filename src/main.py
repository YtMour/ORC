#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片文字方向检测、矫正与PDF转换工具
主要功能：
1. 选择多张图片
2. 通过OCR识别中文文字并判断方向
3. 自动矫正图片方向
4. 将矫正后的图片转换为PDF
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from src.gui.main_window import MainWindow


def main():
    """主函数"""
    # 确保临时目录存在
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    # 启动应用
    app = QApplication(sys.argv)
    app.setApplicationName('图片方向矫正与PDF转换工具')
    
    # 设置应用程序图标
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icon.ico')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 