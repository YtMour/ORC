#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from PIL import Image
import time
import tempfile


def convert_to_pdf(image_paths, output_path=None):
    """
    将图片转换为PDF文件，优先使用原始图片（正确旋转后）
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出PDF路径，默认为临时文件
        
    Returns:
        输出PDF路径
    """
    try:
        logging.info(f"准备转换 {len(image_paths)} 张图片为PDF")
        
        if not image_paths:
            raise ValueError("没有提供图片路径")
        
        # 准备图片列表
        images = []
        
        # 记录处理过的原始图片路径，确保不重复处理
        processed_originals = set()
        
        for path in image_paths:
            # 检查是否存在元数据文件，提取原始图片路径和旋转角度
            meta_path = path + ".meta"
            original_path = None
            rotation_angle = 0
            
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("original_path="):
                                original_path = line.strip().split("=", 1)[1]
                            elif line.startswith("rotation_angle="):
                                try:
                                    rotation_angle = int(line.strip().split("=", 1)[1])
                                except:
                                    rotation_angle = 0
            except Exception as e:
                logging.warning(f"读取元数据文件出错: {str(e)}")
            
            # 如果找到原始图片路径且文件存在，使用原始图片
            if original_path and os.path.exists(original_path) and original_path not in processed_originals:
                logging.info(f"使用原始图片进行PDF转换: {original_path}")
                try:
                    img = Image.open(original_path)
                    processed_originals.add(original_path)  # 添加到已处理列表
                    
                    # 根据旋转角度调整图片方向
                    if rotation_angle != 0:
                        # 修正PIL和OpenCV旋转方向的差异
                        logging.info(f"旋转原始图片 {rotation_angle} 度")
                        # PIL Image旋转方式：ROTATE_90 = 逆时针90度，ROTATE_270 = 顺时针90度
                        # 注意：这与OpenCV的定义是相反的，需要进行转换
                        if rotation_angle == 90:
                            # OpenCV顺时针90度 -> PIL逆时针270度
                            img = img.transpose(Image.ROTATE_270)
                        elif rotation_angle == 180:
                            img = img.transpose(Image.ROTATE_180)
                        elif rotation_angle == 270:
                            # OpenCV逆时针90度 -> PIL逆时针90度
                            img = img.transpose(Image.ROTATE_90)
                    
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')  # 确保图片模式兼容PDF
                    images.append(img)
                except Exception as e:
                    logging.error(f"无法打开原始图片 {original_path}: {str(e)}")
                    # 如果原始图片打开失败，尝试使用处理后的图片（已经包含旋转）
                    img = Image.open(path)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    images.append(img)
            else:
                # 如果没有元数据或原始图片不存在，使用处理后的图片
                logging.info(f"使用处理后的图片进行PDF转换: {path}")
                img = Image.open(path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                images.append(img)
                
        # 设置输出路径
        if not output_path:
            # 生成临时PDF文件路径
            output_dir = os.path.join(tempfile.gettempdir(), 'image_converter')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'output_{int(time.time())}.pdf')
        
        # 将图片转换为PDF
        if images:
            # 取第一张图片作为基准保存其他图片
            images[0].save(
                output_path, 
                save_all=True, 
                append_images=images[1:] if len(images) > 1 else [],
                resolution=100.0,  # 设置分辨率为100dpi
                quality=95        # 设置JPEG压缩质量
            )
            
            logging.info(f"PDF转换完成: {output_path}")
            return output_path
        else:
            raise ValueError("没有可用的图片进行转换")
            
    except Exception as e:
        logging.error(f"PDF转换过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise 