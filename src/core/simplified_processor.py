#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import logging
from PIL import Image

from src.utils.file_utils import get_temp_path


class SimpleImageProcessor:
    """
    简化版图像处理器类，不使用OCR
    仅提供基本的图像旋转功能作为替代方案
    """
    
    def __init__(self):
        """初始化处理器"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SimpleImageProcessor")
        self.logger.info("简化版图像处理器初始化完成")
    
    def process_image(self, image_path):
        """
        处理单张图片 - 由于不使用OCR，这里只是简单地复制图片
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            处理后图片的路径
        """
        self.logger.info(f"开始处理图片: {os.path.basename(image_path)}")
        
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 将图片直接保存到输出路径
            output_path = get_temp_path(os.path.basename(image_path))
            cv2.imwrite(output_path, image)
            
            self.logger.info(f"图片处理完成，保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            raise
    
    def rotate_image(self, image_path, angle=0):
        """
        手动旋转图像
        
        Args:
            image_path: 图片文件路径
            angle: 旋转角度 (0, 90, 180, 270)
            
        Returns:
            旋转后的图片路径
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 根据角度旋转图片
            if angle == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated = image
            
            # 保存旋转后的图片
            output_path = get_temp_path(os.path.basename(image_path))
            cv2.imwrite(output_path, rotated)
            
            return output_path
        except Exception as e:
            self.logger.error(f"旋转图片时出错: {str(e)}")
            raise 