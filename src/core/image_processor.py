#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import logging
import traceback
from PIL import Image
import shutil
import sys
import time

from src.utils.file_utils import get_temp_path

class ImageProcessor:
    """
    图像处理器类，用于:
    1. 检测图片中的文字并判断方向
    2. 根据文字方向矫正图片
    """
    
    def __init__(self):
        """初始化图像处理器和相关参数"""
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ImageProcessor")
        
        # 设置CPU性能参数
        self.use_multi_core = True  # 启用多核处理
        
        # 图像处理参数
        self.min_text_area = 100    # 最小文本区域面积
        self.text_density_threshold = 0.1  # 文本密度阈值
        self.line_length_threshold = 50    # 最小线条长度
        self.angle_threshold = 15          # 角度判断阈值
        
        # 批处理相关参数
        self.batch_mode = False  # 是否处于批处理模式
        self.batch_rotation_angles = []  # 记录批处理中的旋转角度
        self.batch_consistent_rotation = None  # 批处理中确定的一致旋转角度
        
    def get_consistent_rotation(self):
        """获取一致的旋转角度"""
        if not self.batch_rotation_angles:
            return None
            
        # 计算每个角度的出现次数
        from collections import Counter
        angle_counts = Counter(self.batch_rotation_angles)
        
        # 找到出现次数最多的角度
        most_common = angle_counts.most_common(1)[0]
        angle, count = most_common
        
        # 如果这个角度的出现次数超过总数的50%，则使用这个角度
        if count / len(self.batch_rotation_angles) > 0.5:
            self.logger.info(f"找到主要旋转角度: {angle}度 (占比: {count/len(self.batch_rotation_angles):.2%})")
            return angle
            
        # 否则返回None，表示没有找到一致的角度
        self.logger.info("未找到主要旋转角度，各角度分布较为分散")
        return None

    def start_batch_processing(self):
        """启动批量处理模式"""
        self.batch_mode = True
        self.batch_rotation_angles = []
        self.batch_consistent_rotation = None
        self.logger.info("启动批量处理模式")
        
    def end_batch_processing(self):
        """结束批量处理模式"""
        self.batch_mode = False
        if self.batch_rotation_angles:
    # 获取一致的旋转角度
            self.batch_consistent_rotation = self.get_consistent_rotation()
            if self.batch_consistent_rotation is not None:
                self.logger.info(f"批处理结束，确定统一旋转角度为: {self.batch_consistent_rotation}度")
            else:
                self.logger.info("批处理结束，未能确定统一的旋转角度")
        self.batch_rotation_angles = []
    
    def process_image(self, image_path):
        """
        处理单张图片，增强中文文字特征，并保留原始图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理后的图片路径（但PDF处理将使用原始图片）
        """
        try:
            self.logger.info(f"开始处理图片: {os.path.basename(image_path)}")
            
            # 使用新的临时文件路径生成方式
            temp_path = get_temp_path()
            
            # 生成一个存储原始图片的路径（用于PDF生成）
            orig_temp_path = temp_path.replace(".jpg", "_original.jpg")
            
            # 复制原始图片到临时目录（这个将用于最终PDF生成）
            shutil.copy2(image_path, orig_temp_path)
            self.logger.info(f"保存原始图片用于PDF生成: {orig_temp_path}")
            
            # 复制原始图片到临时目录用于处理（这个将在处理中修改）
            shutil.copy2(image_path, temp_path)
            
            # 读取图片 - 添加对中文路径的支持
            img = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                # 尝试直接读取原始路径
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")
            else:
                # 如果能够读取原始图片但无法读取临时路径的图片，重新保存
                cv2.imencode('.jpg', img)[1].tofile(temp_path)
                self.logger.info(f"直接从原路径读取图片并重新保存到临时路径")
            
            # 获取图片原始尺寸
            orig_height, orig_width = img.shape[:2]
            self.logger.info(f"图片原始尺寸: {orig_width}x{orig_height}")
            
            # 确定旋转角度
            if self.batch_mode:
                # 如果在批处理模式下，先尝试使用已有的一致角度
                consistent_angle = self.get_consistent_rotation()
                if consistent_angle is not None:
                    angle = consistent_angle
                    self.logger.info(f"使用批处理一致角度: {angle}度")
                else:
                    # 如果没有一致角度，进行检测
                    angle = self._determine_rotation_angle(img)
                    self.logger.info(f"检测到的旋转角度: {angle}度")
                    # 记录检测到的角度
                self.batch_rotation_angles.append(angle)
            else:
                # 非批处理模式，直接检测角度
                angle = self._determine_rotation_angle(img)
                self.logger.info(f"检测到的旋转角度: {angle}度")
            
            # 处理特殊情况：如果没有确定旋转角度，使用默认的猜测策略
            if angle == 0:
                # 为竖向图片设置默认旋转角度
                if orig_height > orig_width * 1.3:  # 降低竖向图片的判断阈值
                    angle = 270
                    self.logger.info(f"竖向图片但未检测到旋转需求，使用默认角度: {angle}度")
            
            # 根据检测结果旋转图片
            if angle != 0:
                # 旋转处理图片
                img = self._rotate_image(img, angle)
                self.logger.info(f"已将处理图片旋转 {angle} 度")
                
                # 获取旋转后的尺寸
                rot_height, rot_width = img.shape[:2]
                self.logger.info(f"旋转后尺寸: {rot_width}x{rot_height}")
            
            # 获取图片尺寸
            height, width = img.shape[:2]
            
            # 计算新的尺寸（保持宽高比）
            max_size = 2000
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = cv2.resize(img, (new_width, new_height))
                self.logger.info(f"调整图片尺寸为: {new_width}x{new_height}")
            
            # 保存处理后的图片
            cv2.imencode('.jpg', img)[1].tofile(temp_path)
            
            # 记录元数据
            with open(temp_path + ".meta", "w", encoding="utf-8") as f:
                f.write(f"original_path={orig_temp_path}\n")
                f.write(f"processed_path={temp_path}\n")
                f.write(f"rotation_angle={angle}\n")
                f.write(f"processing_time={time.time()}\n")
                f.write(f"batch_mode={self.batch_mode}\n")
                if self.batch_mode:
                    consistent_angle = self.get_consistent_rotation()
                    if consistent_angle is not None:
                        f.write(f"batch_consistent_rotation={consistent_angle}\n")
            
            self.logger.info(f"图片处理完成，保存至: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _determine_rotation_angle(self, image):
        """
        确定图片旋转角度，使用多种方法综合分析
        
        Args:
            image: OpenCV图像对象
            
        Returns:
            旋转角度 (0, 90, 180, 270)
        """
        try:
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 1. 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            
            # 2. 图像预处理 - 增强文字特征
            # 2.1 自适应直方图均衡化 - 增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 2.2 高斯模糊去噪
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 2.3 Otsu二值化 - 更好地分离文字
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 进行形态学操作增强文本特征
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            # 保存分析过程中的图像以便调试
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"binary_{time.time()}.jpg")
            cv2.imencode('.jpg', binary)[1].tofile(debug_path)
            self.logger.info(f"保存二值化图像用于调试: {debug_path}")
            
            # 3. 分析图像特征
            scores = {0: 0, 90: 0, 180: 0, 270: 0}
            evidence = []
            
            # 初始默认得分 - 降低0度的初始得分，降低算法对0度的偏好
            scores[0] = -0.5  # 略微降低0度的初始得分
            
            # 3.1 分析文本区域分布和密度
            text_regions, region_evidence = self._analyze_text_regions(binary)
            if text_regions is not None:
                scores[text_regions] += 3  # 增加权重
                evidence.append(f"文本区域分析指向 {text_regions}度: {region_evidence}")
            
            # 3.2 分析线条特征
            line_angle = self._analyze_lines(binary)
            if line_angle is not None:
                scores[line_angle] += 2
                evidence.append(f"线条特征指向 {line_angle}度")
            
            # 3.3 分析投影特征
            proj_angle, proj_evidence = self._analyze_projections(binary)
            if proj_angle is not None:
                scores[proj_angle] += 2.5  # 提高投影特征权重
                evidence.append(f"投影特征指向 {proj_angle}度: {proj_evidence}")
            
            # 3.4 分析文本行和文本特征
            text_angle, text_evidence = self._analyze_text_layout(binary)
            if text_angle is not None:
                scores[text_angle] += 3  # 提高文本布局分析权重
                evidence.append(f"文本布局分析指向 {text_angle}度: {text_evidence}")
            
            # 3.5 分析图像尺寸特征
            if height > width * 1.2:  # 明显的竖向图片
                scores[270] += 2.5  # 增加竖向图片270度的权重
                evidence.append("图片明显为竖向，倾向270度")
            elif width > height * 1.2:  # 明显的横向图片
                # 移除对0度的偏好，平等对待0度和180度
                evidence.append("图片明显为横向，可能为0度或180度")
            
            # 3.6 特别处理原始方向(0度)和180度旋转
            # 分析上下对称性来区分0度和180度
            upside_down_score = self._detect_upside_down(binary)
            
            # 调整倒置检测的阈值和得分，提高180度判断的准确性
            if upside_down_score > 0.5:  # 降低倒置判断阈值
                # 阈值越高，得分越高
                upside_down_boost = (upside_down_score - 0.5) * 5.0
                scores[180] += 2.5 + upside_down_boost  # 动态增强180度得分
                evidence.append(f"上下部分分析指向倒置图像(180度)，置信度: {upside_down_score:.2f}, 加分: {upside_down_boost:.2f}")
            elif upside_down_score < 0.48:  # 提高正向判断要求
                scores[0] += 1.5  # 降低正向判断的权重
                evidence.append(f"上下部分分析指向正向图像(0度)，置信度: {1-upside_down_score:.2f}")
            
            # 4. 确定最终角度
            # 4.1 找出得分最高的角度
            max_score = max(scores.values())
            top_angles = [angle for angle, score in scores.items() if score == max_score]
            
            # 记录所有角度得分情况
            self.logger.info(f"各角度得分: 0度={scores[0]}, 90度={scores[90]}, 180度={scores[180]}, 270度={scores[270]}")
            
            # 4.2 如果有多个相同得分的角度，根据图像特性选择
            if len(top_angles) > 1:
                if height > width * 1.2 and 270 in top_angles:
                    # 竖向图片优先选择270度
                    final_angle = 270
                    evidence.append("多个角度同分，竖向图片优先选择270度")
                elif 180 in top_angles:
                    # 优先选择180度，增强对倒置图片的识别
                    final_angle = 180
                    evidence.append("多个角度同分，优先选择180度")
                elif 90 in top_angles:
                    # 其次选择90度
                    final_angle = 90
                    evidence.append("多个角度同分，优先选择90度")
                else:
                    # 最后选择0度
                    final_angle = 0
                    evidence.append("多个角度同分，选择0度")
            else:
                final_angle = top_angles[0]
            
            # 4.3 特殊处理：如果是竖向图片且得分最高的是90度，考虑改为270度
            if height > width * 1.2 and final_angle == 90:
                # 进一步分析顶部区域的特征来确认是否需要调整
                top_region = binary[:height//3, :]
                top_density = np.sum(top_region) / (top_region.shape[0] * top_region.shape[1])
                bottom_region = binary[2*height//3:, :]
                bottom_density = np.sum(bottom_region) / (bottom_region.shape[0] * bottom_region.shape[1])
                
                # 降低阈值，更容易选择270度
                if top_density > bottom_density:
                    final_angle = 270
                    evidence.append(f"竖向图片顶部文字密度较大({top_density:.4f}>{bottom_density:.4f})，调整为270度")
            
            # 5. 根据图像特征强制处理特殊情况
            # 强制模式更积极地应用于图像处理
            original_angle = final_angle
            
            # 5.1 竖向图片强制处理
            if height > width * 1.3:  # 降低竖向图片判断阈值
                if final_angle == 0 or final_angle == 180:  # 如果判断为横向
                    # 分析文本分布来确定是90度还是270度
                    if scores[270] > scores[90]:
                        final_angle = 270
                        evidence.append("强制模式: 竖向图片强制旋转270度")
                    else:
                        final_angle = 90
                        evidence.append("强制模式: 竖向图片强制旋转90度")
            
            # 5.2 增强倒置图片强制处理
            # 如果明显偏向180度但未选择，强制应用
            if final_angle == 0 and upside_down_score > 0.52:
                final_angle = 180
                evidence.append(f"强制模式: 倒置可能性较高({upside_down_score:.2f})，强制旋转180度")
            # 检查是否倒置得分接近但不够高
            elif final_angle == 0 and scores[180] > scores[0] * 0.8 and upside_down_score > 0.45:
                final_angle = 180
                evidence.append(f"强制模式: 倒置可能性中等({upside_down_score:.2f})但得分接近({scores[180]:.2f}>{scores[0]:.2f}*0.8)，强制旋转180度")
            
            # 如果强制模式改变了角度，记录下来
            if original_angle != final_angle:
                self.logger.info(f"强制模式将旋转角度从 {original_angle}度 调整为 {final_angle}度")
            
            # 记录详细的分析结果
            self.logger.info(f"角度分析得分: {scores}")
            self.logger.info(f"分析依据: {', '.join(evidence)}")
            self.logger.info(f"最终选择: {final_angle}度")
            
            return final_angle
            
        except Exception as e:
            self.logger.error(f"确定旋转角度时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0
    
    def _analyze_text_regions(self, binary):
        """
        分析文本区域分布特征
        
        Args:
            binary: 二值化图像
            
        Returns:
            建议的旋转角度和分析依据
        """
        try:
            # 1. 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 2. 过滤和分析轮廓
            valid_contours = []
            angles = []
            total_area = binary.shape[0] * binary.shape[1]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_text_area:  # 过滤太小的区域
                    continue
                if area > total_area * 0.5:  # 过滤太大的区域
                    continue
                    
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                w = rect[1][0]
                h = rect[1][1]
                
                # 计算长宽比
                aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
                
                # 只考虑形状规则的文字区域
                if aspect_ratio < 5:  # 过滤过于细长的区域
                    valid_contours.append(contour)
                    angle = rect[-1]
                    if w < h:
                        angle += 90
                    angles.append(angle)
            
            if not valid_contours:
                return None, "未找到有效文本区域"
            
            # 3. 统计角度
            angle_counts = {0: 0, 90: 0, 180: 0, 270: 0}
            for angle in angles:
                # 将角度规范化到0-360度
                angle = angle % 360
                if angle > 180:
                    angle -= 360
                
                # 根据角度范围进行分类
                if -self.angle_threshold <= angle <= self.angle_threshold:
                    angle_counts[0] += 1
                elif 90 - self.angle_threshold <= angle <= 90 + self.angle_threshold:
                    angle_counts[90] += 1
                elif abs(angle) > 180 - self.angle_threshold:
                    angle_counts[180] += 1
                elif -90 - self.angle_threshold <= angle <= -90 + self.angle_threshold:
                    angle_counts[270] += 1
            
            # 4. 分析文本分布
            # 获取所有轮廓的边界框
            bboxes = [cv2.boundingRect(cnt) for cnt in valid_contours]
            
            # 计算文本区域的分布信息
            if len(bboxes) < 3:
                return max(angle_counts.items(), key=lambda x: x[1])[0], f"文本区域角度分布: {angle_counts}"
            
            # 分析文本区域在图像中的位置分布
            y_positions = [y+h/2 for x,y,w,h in bboxes]
            x_positions = [x+w/2 for x,y,w,h in bboxes]
            
            # 计算水平和垂直方向上文本区域的分布
            height, width = binary.shape
            
            # 划分图像为3x3网格，统计每个网格中的文本区域数量
            grid_counts = np.zeros((3, 3))
            for x, y in zip(x_positions, y_positions):
                grid_x = min(2, int(3 * x / width))
                grid_y = min(2, int(3 * y / height))
                grid_counts[grid_y, grid_x] += 1
            
            # 分析网格分布来推断文档方向
            top_row = np.sum(grid_counts[0, :])
            middle_row = np.sum(grid_counts[1, :])
            bottom_row = np.sum(grid_counts[2, :])
            
            left_col = np.sum(grid_counts[:, 0])
            middle_col = np.sum(grid_counts[:, 1])
            right_col = np.sum(grid_counts[:, 2])
            
            # 根据文本区域分布特征判断方向
            evidence = f"文本角度分布: {angle_counts}, 网格分布: 上{top_row}/中{middle_row}/下{bottom_row}, 左{left_col}/中{middle_col}/右{right_col}"
            
            # 正常文档上部通常比下部文本少
            if top_row < bottom_row * 0.8:
                angle_counts[0] += 2
            # 倒置文档下部通常比上部文本少
            elif bottom_row < top_row * 0.8:
                angle_counts[180] += 2
            
            # 中文文档右侧通常比左侧文本多（从右到左阅读）
            if right_col > left_col * 1.2 and height > width:
                angle_counts[270] += 2
            # 如果左侧比右侧文本多，可能是90度
            elif left_col > right_col * 1.2 and height > width:
                angle_counts[90] += 2
            
            return max(angle_counts.items(), key=lambda x: x[1])[0], evidence
            
        except Exception as e:
            self.logger.error(f"分析文本区域时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, f"分析出错: {str(e)}"

    def _analyze_lines(self, binary):
        """
        分析图像中的线条特征
        
        Args:
            binary: 二值化图像
            
        Returns:
            建议的旋转角度
        """
        try:
            # 使用霍夫线变换检测直线
            lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50,
                                  minLineLength=self.line_length_threshold,
                                  maxLineGap=20)
            
            if lines is None:
                return None
            
            # 统计水平和垂直线
            h_lines = 0
            v_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < self.angle_threshold or angle > 180 - self.angle_threshold:
                    h_lines += 1
                elif 90 - self.angle_threshold <= angle <= 90 + self.angle_threshold:
                    v_lines += 1
            
            # 根据线条数量判断方向
            if h_lines > v_lines * 1.5:
                return 0
            elif v_lines > h_lines * 1.5:
                return 90
            
            return None
            
        except Exception as e:
            self.logger.error(f"分析线条特征时出错: {str(e)}")
            return None

    def _analyze_projections(self, binary):
        """
        分析图像投影特征
        
        Args:
            binary: 二值化图像
            
        Returns:
            建议的旋转角度和分析证据
        """
        try:
            height, width = binary.shape
            
            # 计算水平和垂直投影
            h_proj = np.sum(binary, axis=1)
            v_proj = np.sum(binary, axis=0)
            
            # 平滑投影曲线
            h_proj_smooth = np.convolve(h_proj, np.ones(5)/5, mode='same')
            v_proj_smooth = np.convolve(v_proj, np.ones(5)/5, mode='same')
            
            # 计算投影的变异系数
            h_mean = np.mean(h_proj_smooth)
            v_mean = np.mean(v_proj_smooth)
            
            if h_mean == 0 or v_mean == 0:
                return None, "投影均值为0"
                
            h_cv = np.std(h_proj_smooth) / h_mean
            v_cv = np.std(v_proj_smooth) / v_mean
            
            # 计算投影的峰值
            from scipy.signal import find_peaks
            h_peaks, _ = find_peaks(h_proj_smooth, height=h_mean*1.5, distance=10)
            v_peaks, _ = find_peaks(v_proj_smooth, height=v_mean*1.5, distance=10)
            
            h_peak_count = len(h_peaks)
            v_peak_count = len(v_peaks)
            
            evidence = f"水平变异={h_cv:.2f},垂直变异={v_cv:.2f},水平峰={h_peak_count},垂直峰={v_peak_count}"
            
            # 分析投影特征来判断方向
            # 1. 根据变异系数判断
            if h_cv < v_cv * 0.7:  # 降低阈值，更容易识别横向文本
                return 0, evidence + ", 水平变异明显小于垂直变异"
            elif v_cv < h_cv * 0.7:  # 降低阈值，更容易识别竖向文本
                if height > width:  # 竖向图片
                    return 270, evidence + ", 垂直变异明显小于水平变异，竖向图片"
                else:
                    return 90, evidence + ", 垂直变异明显小于水平变异，横向图片"
            
            # 2. 根据峰值数量判断
            if h_peak_count > v_peak_count * 1.5:
                return 0, evidence + ", 水平峰值数量明显多于垂直峰值"
            elif v_peak_count > h_peak_count * 1.5:
                if height > width:  # 竖向图片
                    return 270, evidence + ", 垂直峰值数量明显多于水平峰值，竖向图片"
                else:
                    return 90, evidence + ", 垂直峰值数量明显多于水平峰值，横向图片"
            
            # 3. 分析投影的分布特征
            # 检查顶部和底部的投影值以区分0度和180度
            top_quarter = h_proj_smooth[:height//4]
            bottom_quarter = h_proj_smooth[3*height//4:]
            
            top_density = np.mean(top_quarter)
            bottom_density = np.mean(bottom_quarter)
            
            if bottom_density > top_density * 1.3:
                return 0, evidence + f", 底部密度({bottom_density:.2f})明显高于顶部({top_density:.2f})"
            elif top_density > bottom_density * 1.3:
                return 180, evidence + f", 顶部密度({top_density:.2f})明显高于底部({bottom_density:.2f})"
            
            # 检查左侧和右侧的投影值以区分90度和270度
            left_quarter = v_proj_smooth[:width//4]
            right_quarter = v_proj_smooth[3*width//4:]
            
            left_density = np.mean(left_quarter)
            right_density = np.mean(right_quarter)
            
            if height > width:  # 竖向图片
                if right_density > left_density * 1.3:
                    return 270, evidence + f", 右侧密度({right_density:.2f})明显高于左侧({left_density:.2f})"
                elif left_density > right_density * 1.3:
                    return 90, evidence + f", 左侧密度({left_density:.2f})明显高于右侧({right_density:.2f})"
            
            return None, evidence
            
        except Exception as e:
            self.logger.error(f"分析投影特征时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, f"分析出错: {str(e)}"

    def _analyze_text_layout(self, binary):
        """
        分析文本布局特征，用于确定方向
        
        Args:
            binary: 二值化图像
            
        Returns:
            建议的旋转角度和证据
        """
        try:
            height, width = binary.shape
            
            # 1. 计算水平投影和垂直投影
            h_proj = np.sum(binary, axis=1)  # 水平投影
            v_proj = np.sum(binary, axis=0)  # 垂直投影
            
            # 归一化投影
            h_proj = h_proj / width if width > 0 else h_proj
            v_proj = v_proj / height if height > 0 else v_proj
            
            # 2. 检测文本行
            # 使用自适应阈值检测明显的文本行
            h_mean = np.mean(h_proj)
            v_mean = np.mean(v_proj)
            
            h_threshold = h_mean * 1.2
            v_threshold = v_mean * 1.2
            
            # 找出高于阈值的行和列
            text_rows = np.where(h_proj > h_threshold)[0]
            text_cols = np.where(v_proj > v_threshold)[0]
            
            # 3. 分析文本行的特征
            row_groups = []
            if len(text_rows) > 0:
                # 将连续的行组合成行组
                current_group = [text_rows[0]]
                for i in range(1, len(text_rows)):
                    if text_rows[i] - text_rows[i-1] <= 3:  # 相邻的行
                        current_group.append(text_rows[i])
                    else:
                        row_groups.append(current_group)
                        current_group = [text_rows[i]]
                row_groups.append(current_group)
            
            # 类似地，分析列
            col_groups = []
            if len(text_cols) > 0:
                current_group = [text_cols[0]]
                for i in range(1, len(text_cols)):
                    if text_cols[i] - text_cols[i-1] <= 3:
                        current_group.append(text_cols[i])
                    else:
                        col_groups.append(current_group)
                        current_group = [text_cols[i]]
                col_groups.append(current_group)
            
            # 4. 计算文本行和列的特征
            row_lengths = [len(group) for group in row_groups]
            col_lengths = [len(group) for group in col_groups]
            
            row_distances = []
            for i in range(1, len(row_groups)):
                start_of_current = row_groups[i][0]
                end_of_previous = row_groups[i-1][-1]
                row_distances.append(start_of_current - end_of_previous)
            
            col_distances = []
            for i in range(1, len(col_groups)):
                start_of_current = col_groups[i][0]
                end_of_previous = col_groups[i-1][-1]
                col_distances.append(start_of_current - end_of_previous)
            
            # 5. 分析结果
            evidence = ""
            
            # 文本行特征分析
            if len(row_groups) >= 3:
                # 计算行间距的一致性
                row_dist_std = np.std(row_distances) if row_distances else 0
                row_dist_mean = np.mean(row_distances) if row_distances else 0
                row_cv = row_dist_std / row_dist_mean if row_dist_mean > 0 else float('inf')
                
                evidence += f"行组:{len(row_groups)}, 行间变异系数:{row_cv:.2f}"
                
                # 分析文本行的位置分布
                row_positions = [np.mean(group) / height for group in row_groups]
                top_rows = sum(1 for pos in row_positions if pos < 0.33)
                middle_rows = sum(1 for pos in row_positions if 0.33 <= pos < 0.67)
                bottom_rows = sum(1 for pos in row_positions if pos >= 0.67)
                
                evidence += f", 行分布:上{top_rows}/中{middle_rows}/下{bottom_rows}"
                
                # 普通横向文档通常行间距一致且分布均匀
                if row_cv < 0.4 and len(row_groups) >= 3:
                    # 检查是正向还是倒置
                    if bottom_rows > top_rows * 1.5:
                        return 0, evidence + ", 底部行多"
                    elif top_rows > bottom_rows * 1.5:
                        return 180, evidence + ", 顶部行多"
                else:
                    return 0, evidence + ", 行均匀分布"
            
            # 文本列特征分析
            if len(col_groups) >= 3:
                # 计算列间距的一致性
                col_dist_std = np.std(col_distances) if col_distances else 0
                col_dist_mean = np.mean(col_distances) if col_distances else 0
                col_cv = col_dist_std / col_dist_mean if col_dist_mean > 0 else float('inf')
                
                evidence += f", 列组:{len(col_groups)}, 列间变异系数:{col_cv:.2f}"
                
                # 分析文本列的位置分布
                col_positions = [np.mean(group) / width for group in col_groups]
                left_cols = sum(1 for pos in col_positions if pos < 0.33)
                middle_cols = sum(1 for pos in col_positions if 0.33 <= pos < 0.67)
                right_cols = sum(1 for pos in col_positions if pos >= 0.67)
                
                evidence += f", 列分布:左{left_cols}/中{middle_cols}/右{right_cols}"
                
                # 中文竖向文档通常列间距一致，且右侧列较多
                if col_cv < 0.4 and len(col_groups) >= 3:
                    if right_cols > left_cols * 1.5:
                        return 270, evidence + ", 右侧列多"
                    elif left_cols > right_cols * 1.5:
                        return 90, evidence + ", 左侧列多"
            
            # 如果行更明显，偏向于横向文档
            if len(row_groups) > len(col_groups) * 1.5:
                return 0, evidence + ", 行数明显多于列数"
            
            # 如果列更明显，偏向于竖向文档
            if len(col_groups) > len(row_groups) * 1.5:
                return 270, evidence + ", 列数明显多于行数"
            
            return None, evidence
            
        except Exception as e:
            self.logger.error(f"分析文本布局时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, f"分析出错: {str(e)}"

    def _detect_upside_down(self, binary):
        """
        检测图像是否倒置(180度旋转)
        
        Args:
            binary: 二值化图像
            
        Returns:
            倒置可能性分数 (0-1之间，越接近1表示越可能倒置)
        """
        try:
            height, width = binary.shape
            
            # 1. 分析上下部分的文本密度
            top_third = binary[:height//3, :]
            middle_third = binary[height//3:2*height//3, :]
            bottom_third = binary[2*height//3:, :]
            
            top_density = np.sum(top_third) / (top_third.size)
            middle_density = np.sum(middle_third) / (middle_third.size)
            bottom_density = np.sum(bottom_third) / (bottom_third.size)
            
            # 记录密度信息用于调试
            self.logger.info(f"文本密度分析: 上部={top_density:.4f}, 中部={middle_density:.4f}, 下部={bottom_density:.4f}")
            
            # 计算一个倒置分数
            upside_down_score = 0.5  # 默认值
            
            # 正常文档通常底部文本密度更大，或中间部分密度最大
            if bottom_density > top_density * 1.05:
                upside_down_score -= (bottom_density / top_density - 1) * 0.2
            elif top_density > bottom_density * 1.05:
                upside_down_score += (top_density / bottom_density - 1) * 0.2
            elif middle_density > top_density and middle_density > bottom_density:
                # 中间密度最大时，比较上下部分
                if bottom_density > top_density * 1.05:
                    upside_down_score -= 0.1
                elif top_density > bottom_density * 1.05:
                    upside_down_score += 0.1
                
            # 2. 分析文本行的位置
            # 使用水平投影找出文本行
            h_proj = np.sum(binary, axis=1)
            
            # 平滑投影
            h_proj_smooth = np.convolve(h_proj, np.ones(5)/5, mode='same')
            
            # 找出明显的文本行（投影值较高的行）
            threshold = np.mean(h_proj_smooth) * 1.5
            text_line_positions = np.where(h_proj_smooth > threshold)[0]
            
            if len(text_line_positions) > 0:
                # 计算文本行的位置分布
                line_positions = np.array(text_line_positions) / height
                
                # 计算上下三分之一区域的文本行数量
                top_third_lines = np.sum(line_positions < 0.33)
                bottom_third_lines = np.sum(line_positions > 0.67)
                
                # 记录文本行分布用于调试
                self.logger.info(f"文本行分布: 上部={top_third_lines}, 下部={bottom_third_lines}")
                
                # 调整倒置分数
                if bottom_third_lines > top_third_lines * 1.2:
                    upside_down_score = max(0.1, upside_down_score - 0.2)  # 底部文本行多，可能正向
                elif top_third_lines > bottom_third_lines * 1.2:
                    upside_down_score = min(0.9, upside_down_score + 0.2)  # 顶部文本行多，可能倒置
            
            # 3. 分析文本块的垂直分布
            # 尝试检测文本块
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤并获取有效的文本块边界框
            valid_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_text_area or area > binary.size * 0.3:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w > h * 5 or h > w * 5:  # 过滤极端长宽比的区域
                    continue
                valid_boxes.append((x, y, w, h))
            
            # 如果有足够的文本块，分析其分布
            if len(valid_boxes) >= 3:
                # 计算每个块的中心y坐标
                y_centers = [y + h/2 for _, y, _, h in valid_boxes]
                
                # 计算上下三分位数的文本块数量
                top_third_blocks = sum(1 for y in y_centers if y < height/3)
                bottom_third_blocks = sum(1 for y in y_centers if y > 2*height/3)
                
                # 记录文本块分布用于调试
                self.logger.info(f"文本块分布: 上部={top_third_blocks}, 下部={bottom_third_blocks}")
                
                # 调整倒置分数
                if bottom_third_blocks > top_third_blocks * 1.2:
                    upside_down_score = max(0.1, upside_down_score - 0.2)
                elif top_third_blocks > bottom_third_blocks * 1.2:
                    upside_down_score = min(0.9, upside_down_score + 0.2)
                    
                # 4. 分析文本块的上下特征
                # 如果上部区域的文本块高度普遍大于下部区域，可能是倒置的
                top_heights = [h for _, y, _, h in valid_boxes if y < height/3]
                bottom_heights = [h for _, y, _, h in valid_boxes if y > 2*height/3]
                
                if top_heights and bottom_heights:
                    avg_top_height = sum(top_heights) / len(top_heights)
                    avg_bottom_height = sum(bottom_heights) / len(bottom_heights)
                    
                    self.logger.info(f"文本块高度分析: 上部平均={avg_top_height:.2f}, 下部平均={avg_bottom_height:.2f}")
                    
                    # 标题和页眉通常字体较大，如果在底部发现大字体，可能是倒置的
                    if avg_bottom_height > avg_top_height * 1.3:
                        upside_down_score = min(0.9, upside_down_score + 0.2)
                        self.logger.info("底部文本块高度明显大于顶部，可能是倒置的")
            
            self.logger.info(f"倒置分析最终得分: {upside_down_score:.2f}")
            return upside_down_score
            
        except Exception as e:
            self.logger.error(f"检测图像倒置时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0.5  # 出错时返回默认值
    
    def _rotate_image(self, image, angle):
        """
        旋转图像
        
        Args:
            image: OpenCV图像对象
            angle: 旋转角度 (0, 90, 180, 270)
            
        Returns:
            旋转后的图像
        """
        # 确保角度为0, 90, 180, 270之一
        angle = int(angle)
        if angle not in [0, 90, 180, 270]:
            angle = 0
        
        # 如果不需要旋转，直接返回原图
        if angle == 0:
            return image
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 计算旋转矩阵
        if angle == 90:
            # 顺时针旋转90度
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            # 旋转180度
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            # 顺时针旋转270度 (或逆时针旋转90度)
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = image
        
        return rotated 

    def force_rotate(self, image_path, angle):
        """
        强制旋转图片（供UI调用的手动旋转功能）
        
        Args:
            image_path: 图片路径
            angle: 旋转角度 (90, 180, 270)
            
        Returns:
            旋转后的图片路径
        """
        try:
            self.logger.info(f"手动旋转图片: {os.path.basename(image_path)}，角度: {angle}度")
            
            # 确保角度为90, 180, 270之一
            angle = int(angle)
            if angle not in [90, 180, 270]:
                raise ValueError(f"不支持的旋转角度: {angle}，请使用90, 180或270")
            
            # 读取图片
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 旋转图片
            rotated_img = self._rotate_image(img, angle)
            
            # 生成新的文件名
            output_path = get_temp_path()
            
            # 保存旋转后的图片
            cv2.imencode('.jpg', rotated_img)[1].tofile(output_path)
            self.logger.info(f"图片已手动旋转，保存至: {output_path}")
            
            # 记录元数据
            with open(output_path + ".meta", "w", encoding="utf-8") as f:
                f.write(f"original_path={image_path}\n")
                f.write(f"processed_path={output_path}\n")
                f.write(f"rotation_angle={angle}\n")
                f.write(f"manual_rotation=True\n")
                f.write(f"processing_time={time.time()}\n")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"手动旋转图片时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise 