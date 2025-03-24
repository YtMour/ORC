import os
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
                             QFileDialog, QMessageBox, QProgressBar, QRadioButton,
                             QButtonGroup, QFrame, QScrollArea, QTextEdit)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QMimeData
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
import easyocr
import img2pdf
from PIL import Image
import numpy as np
import tempfile
import time
import cv2
import re
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed



# 尝试导入TkinterDnD，用于支持拖放功能
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
except ImportError:
    TKDND_AVAILABLE = False

def setup_logging():
    """设置日志记录"""
    # 创建logs目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_{current_time}.log")
    
    # 配置日志记录器
    logger = logging.getLogger("ImageProcessor")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ProcessThread(QThread):
    """图片处理线程"""
    # 定义信号
    progress = Signal(int)  # 进度信号
    status = Signal(str)    # 状态信号
    finished = Signal(list) # 完成信号，返回处理后的图片列表
    error = Signal(str)     # 错误信号

    def __init__(self, files, reader, lang_group, rotation_group):
        super().__init__()
        self.files = files
        self.reader = reader
        self.lang_group = lang_group
        self.rotation_group = rotation_group
        self.processed_images = []
        self.debug_mode = True
        self.max_image_size = 2000  # 增加处理图片的最大尺寸
        self.num_workers = os.cpu_count() or 1  # 获取CPU核心数
        
        # 设置日志记录器
        self.logger = logging.getLogger("ImageProcessor.ProcessThread")
    
    def preprocess_image(self, img_np):
        """预处理图片以提高处理质量和速度"""
        try:
            # 转换为灰度图
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
                
            # 调整图片大小以加快处理速度
            h, w = gray.shape[:2]
            if max(h, w) > self.max_image_size:
                scale = self.max_image_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 1. 去噪处理
            # 使用双边滤波保留边缘信息
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            # 再使用非局部均值去噪
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
            
            # 2. 对比度增强
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. 锐化处理
            # 使用拉普拉斯算子
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                sharpened,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # 5. 形态学操作
            # 创建核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            # 先闭运算，填充小孔
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            # 再开运算，去除小噪点
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # 6. 倾斜校正
            # 检测直线
            edges = cv2.Canny(morph, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is not None:
                # 计算主要方向
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    # 使用RANSAC算法筛选异常值
                    angles = np.array(angles)
                    median_angle = np.median(angles)
                    mad = np.median(np.abs(angles - median_angle))
                    inliers = angles[np.abs(angles - median_angle) < 2.5 * mad]
                    
                    if len(inliers) > 0:
                        final_angle = np.mean(inliers)
                        if abs(final_angle) > 0.5:  # 如果倾斜角度大于0.5度
                            # 旋转校正
                            h, w = morph.shape
                            center = (w//2, h//2)
                            M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
                            morph = cv2.warpAffine(morph, M, (w, h), 
                                                 flags=cv2.INTER_CUBIC,
                                                 borderMode=cv2.BORDER_REPLICATE)
            
            # 7. 边缘增强
            edge_enhanced = cv2.addWeighted(morph, 0.8, cv2.Canny(morph, 50, 150), 0.2, 0)
            
            # 8. 最终清理
            # 移除小连通区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_enhanced, connectivity=8)
            min_size = 50  # 最小连通区域大小
            
            # 保留大于最小尺寸的连通区域
            for i in range(1, num_labels):  # 从1开始，跳过背景
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    edge_enhanced[labels == i] = 0
                
            return edge_enhanced
            
        except Exception as e:
            self.logger.error(f"图像预处理错误: {str(e)}", exc_info=True)
            return img_np
    
    def process_single_image(self, file_path):
        """处理单个图片，识别文字并确定正确方向"""
        try:
            # 打开图片
            img = Image.open(file_path)
            img_np = np.array(img)
            
            # 检查旋转模式
            rotation_mode = "auto"
            for button in self.rotation_group.buttons():
                if button.isChecked():
                    rotation_mode = button.text().replace("旋转", "").replace("°", "")
                    if rotation_mode == "自动检测":
                        rotation_mode = "auto"
                    break
            
            # 如果不是自动模式，直接按指定角度旋转
            if rotation_mode != "auto":
                rotation_angle = int(rotation_mode)
                if rotation_angle != 0:
                    img = img.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
                return img
            
            # 自动模式：执行OCR识别和角度判断
            # 预处理图片
            processed_img = self.preprocess_image(img_np)
            
            # 尝试四个方向的OCR识别
            orientations = [0, 90, 180, 270]
            best_confidence = -1
            best_orientation = 0
            best_results = None
            
            for angle in orientations:
                # 旋转图像进行OCR
                if angle != 0:
                    rotated = self.rotate_image(processed_img, angle)
                else:
                    rotated = processed_img
                
                # OCR识别
                results = self.reader.readtext(rotated)
                
                if results:
                    # 计算综合置信度得分
                    confidence = self.calculate_confidence_score(results, angle)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_orientation = angle
                        best_results = results
            
            # 如果OCR结果不理想，使用线条分析
            if best_confidence < 0.4:  # 置信度阈值
                line_orientation = self.determine_orientation_by_lines(processed_img)
                if line_orientation != 0:
                    best_orientation = line_orientation
            
            # 执行最终旋转
            if best_orientation != 0:
                img = img.rotate(best_orientation, expand=True, resample=Image.BICUBIC)
            
            return img
            
        except Exception as e:
            self.logger.error(f"处理图片 {os.path.basename(file_path)} 时出错: {str(e)}", exc_info=True)
            return None
    
    def calculate_confidence_score(self, results, angle):
        """计算OCR结果的综合置信度得分"""
        if not results:
            return -1
        
        # 1. 基础置信度计算
        confidences = []
        weights = []
        total_text = ""
        
        for box, text, prob in results:
            confidences.append(prob)
            weights.append(len(text))
            total_text += text + " "
        
        weights = np.array(weights) / np.sum(weights)
        base_confidence = np.sum(np.array(confidences) * weights)
        
        # 2. 文本特征分析
        text_lines = len(results)
        line_bonus = min(text_lines / 8.0, 1.2)
        
        # 3. 文本框分析
        boxes = np.array([box for box, _, _ in results])
        if len(boxes) > 1:
            # 3.1 计算文本框的对齐程度
            x_coords = boxes[:, :, 0].flatten()
            y_coords = boxes[:, :, 1].flatten()
            
            # 计算对齐度
            def calculate_alignment(coords):
                if len(coords) < 2:
                    return 0.5
                sorted_coords = np.sort(coords)
                diffs = np.diff(sorted_coords)
                alignment = 1.0 / (1.0 + np.std(diffs) / np.mean(diffs))
                return alignment
            
            h_alignment = calculate_alignment(x_coords)
            v_alignment = calculate_alignment(y_coords)
            
            # 3.2 根据角度选择合适的对齐度
            if angle in [0, 180]:
                alignment_score = h_alignment * 1.2
            elif angle in [90, 270]:
                alignment_score = v_alignment * 1.2
            else:
                alignment_score = max(h_alignment, v_alignment)
            
            # 3.3 计算文本框间距的规律性
            if angle in [0, 180]:
                gaps = np.diff(sorted(y_coords.reshape(-1, 2).mean(axis=1)))
            else:
                gaps = np.diff(sorted(x_coords.reshape(-1, 2).mean(axis=1)))
            
            if len(gaps) > 0:
                gap_regularity = 1.0 / (1.0 + np.std(gaps) / np.mean(gaps))
                if np.max(gaps) > 3 * np.mean(gaps):
                    gap_regularity *= 0.8
            else:
                gap_regularity = 0.5
        else:
            alignment_score = 0.5
            gap_regularity = 0.5
        
        # 4. 增强的文本内容分析
        content_score = 0
        total_chars = 0
        valid_text_lines = 0
        
        # 4.1 标点符号位置分析
        punctuation_score = 0
        chinese_puncts = '。，；：！？、'
        english_puncts = '.,;:!?'
        all_puncts = chinese_puncts + english_puncts
        
        for _, text, _ in results:
            text = text.strip()
            if not text:
                continue
            
            total_chars += len(text)
            valid_text_lines += 1
            
            # 检查标点符号位置
            if angle == 180:  # 特别关注倒置情况
                # 如果文本以标点开始，这可能是倒置的迹象
                if text[0] in all_puncts:
                    punctuation_score -= 0.2
                # 如果文本以标点结束，这是正常的
                if text[-1] in all_puncts:
                    punctuation_score += 0.1
            else:
                # 正常方向的文本
                if text[-1] in all_puncts:
                    punctuation_score += 0.1
        
        # 4.2 中文文本特征分析
        chinese_ratio = sum(1 for c in total_text if '\u4e00' <= c <= '\u9fff') / len(total_text) if total_text else 0
        
        # 4.3 行首缩进分析（针对中文）
        if chinese_ratio > 0.5 and angle in [0, 180]:
            indent_score = 0
            for _, text, _ in results:
                if text.strip() and text[0] == '　':  # 检查全角空格
                    indent_score += 0.1
            content_score += min(indent_score, 0.3)
        
        # 5. 方向特征分析
        direction_score = 1.0
        
        # 5.1 处理倒置情况 (180度)
        if angle == 180:
            # 检查是否存在明显的倒置特征
            inverted_features = 0
            
            # 检查标点符号位置
            if punctuation_score < 0:
                inverted_features += 1
            
            # 检查中文文本的特征
            if chinese_ratio > 0.5:
                # 检查是否存在不合理的行首标点
                for _, text, _ in results:
                    if text.strip() and text[0] in chinese_puncts:
                        inverted_features += 1
            
            # 根据倒置特征数量调整方向得分
            if inverted_features >= 2:
                direction_score *= 0.7
        
        # 6. 计算最终得分
        weights = {
            'base_confidence': 0.20,
            'alignment': 0.25,
            'gap_regularity': 0.15,
            'content': 0.20,
            'direction': 0.20
        }
        
        final_score = (
            weights['base_confidence'] * base_confidence +
            weights['alignment'] * alignment_score +
            weights['gap_regularity'] * gap_regularity +
            weights['content'] * (content_score + punctuation_score) +
            weights['direction'] * direction_score
        )
        
        # 7. 应用额外的方向惩罚
        if angle == 180:
            # 如果存在强烈的倒置迹象，显著降低分数
            if inverted_features >= 2:
                final_score *= 0.6
            # 如果文本行很少，增加惩罚
            if text_lines < 3:
                final_score *= 0.8
        
        return final_score
    
    def rotate_image(self, img, angle):
        """旋转OpenCV格式的图像"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        # 计算新的边界尺寸以确保所有内容都在视图中
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        
        # 调整旋转矩阵
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
        return rotated
    
    def confirm_orientation(self, img_np, original_results, best_orientation, best_results):
        """通过多种特征确认最终的旋转方向"""
        # 如果最佳方向是0度，则不需要旋转
        if best_orientation == 0:
            return 0
        
        # 如果没有检测到文本，则尝试使用Hough线变换来确定方向
        if not original_results and not best_results:
            return self.determine_orientation_by_lines(img_np)
        
        # 如果检测到的文本非常少，增加额外检查
        if best_results and len(best_results) < 3:
            lines_orientation = self.determine_orientation_by_lines(img_np)
            # 如果线条方向与OCR方向一致，则增强置信度
            if lines_orientation == best_orientation:
                return best_orientation
            # 如果文本太少且线条分析结果不同，使用线条分析结果
            return lines_orientation
        
        return best_orientation
    
    def determine_orientation_by_lines(self, img_np):
        """使用多种特征分析确定图像方向"""
        try:
            # 转换为灰度图
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # 1. 图像预处理
            # 使用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 自适应阈值处理
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # 边缘检测
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)
            
            # 2. 增强的线段检测
            # 使用概率霍夫变换检测线段
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                  threshold=100,
                                  minLineLength=50,  # 降低最小线段长度以检测更多线段
                                  maxLineGap=10)
            
            if lines is None:
                return 0
            
            # 3. 改进的方向分析
            angles = []
            lengths = []  # 存储线段长度
            confidences = []  # 存储每个线段的置信度
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算线段长度
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # 计算角度
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 归一化角度到 [-90, 90]
                angle = angle % 180
                if angle > 90:
                    angle -= 180
                    
                # 计算线段的置信度
                # 基于线段长度和边缘强度
                edge_strength = np.mean([edges[y1, x1], edges[y2, x2]])
                confidence = length * edge_strength / 255.0
                
                angles.append(angle)
                lengths.append(length)
                confidences.append(confidence)
            
            # 4. 加权统计分析
            angles = np.array(angles)
            lengths = np.array(lengths)
            confidences = np.array(confidences)
            
            # 使用RANSAC算法筛选异常值
            if len(angles) > 10:
                median_angle = np.median(angles)
                mad = np.median(np.abs(angles - median_angle))
                inlier_mask = np.abs(angles - median_angle) < 2.5 * mad
                
                angles = angles[inlier_mask]
                lengths = lengths[inlier_mask]
                confidences = confidences[inlier_mask]
            
            # 计算加权方向
            weights = lengths * confidences
            weighted_angles = angles * weights
            
            # 使用更细的直方图bins
            hist, bins = np.histogram(weighted_angles, bins=36, range=(-90, 90), weights=weights)
            smoothed_hist = np.convolve(hist, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')  # 平滑直方图
            
            # 5. 文本行分析
            # 计算投影
            row_projection = np.sum(binary, axis=1)
            col_projection = np.sum(binary, axis=0)
            
            # 使用小波变换进行投影分析
            def analyze_projection(proj):
                # 使用haar小波变换检测周期性
                import pywt
                coeffs = pywt.wavedec(proj, 'haar', level=3)
                # 计算每个尺度的能量
                energies = [np.sum(c**2) for c in coeffs]
                return np.array(energies)
            
            row_energies = analyze_projection(row_projection)
            col_energies = analyze_projection(col_projection)
            
            # 6. 综合判断
            # 获取主方向
            main_angle_idx = np.argmax(smoothed_hist)
            main_angle = (main_angle_idx * 5) - 90  # 将bin索引转换为角度
            
            # 计算方向的置信度
            direction_confidence = smoothed_hist[main_angle_idx] / np.sum(smoothed_hist)
            
            # 分析文本行特征
            is_horizontal = row_energies[1] > col_energies[1]  # 比较水平和垂直方向的周期性
            
            # 7. 最终决策
            if direction_confidence > 0.3:  # 如果有明显的主方向
                if abs(main_angle) < 30:  # 接近水平
                    return 0 if is_horizontal else 90
                elif main_angle > 30:  # 需要逆时针旋转
                    return 90 if is_horizontal else 180
                else:  # 需要顺时针旋转
                    return 270 if is_horizontal else 0
            else:  # 如果没有明显的主方向，依据文本行特征
                return 0 if is_horizontal else 90
            
        except Exception as e:
            self.logger.error(f"方向检测错误: {str(e)}", exc_info=True)
            return 0

    def run(self):
        """线程运行函数"""
        try:
            total = len(self.files)
            processed_count = 0
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(self.process_single_image, file_path): file_path 
                    for file_path in self.files
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        processed_img = future.result()
                        if processed_img:
                            self.processed_images.append(processed_img)
                        processed_count += 1
                        
                        # 更新进度
                        progress = int((processed_count / total) * 100)
                        self.progress.emit(progress)
                        self.status.emit(f"已完成 {processed_count}/{total} 张图片 ({progress}%)")
                        
                    except Exception as e:
                        self.logger.error(f"处理图片 {os.path.basename(file_path)} 时出错: {str(e)}")
            
            self.progress.emit(100)
            self.status.emit(f"处理完成! 成功处理 {len(self.processed_images)} 张图片")
            self.finished.emit(self.processed_images)
            
        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}", exc_info=True)
            error_msg = f"处理图片时出错: {str(e)}"
            self.error.emit(error_msg)

class ImageProcessorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("缘缘专用资料整理工具")
        self.setMinimumSize(900, 700)
        
        self.selected_files = []
        self.processed_images = []
        self.reader = None
        self.debug_mode = True
        self.process_thread = None
        
        # 语言代码映射
        self.lang_map = {
            "简体中文": "ch_sim",
            "英文": "en"
        }
        
        # 设置日志记录器
        self.logger = setup_logging()
        self.logger.info("程序启动")
        
        # 记录当前环境信息
        self.logger.info(f"操作系统: {sys.platform}")
        self.logger.info(f"Python版本: {sys.version}")
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        self.initialize_ui(main_layout)
        
        # 设置拖放
        self.setAcceptDrops(True)
    
    def closeEvent(self, event):
        """重写关闭事件，确保线程正确清理"""
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.quit()
            self.process_thread.wait()
        event.accept()
    
    def initialize_ui(self, main_layout):
        # 顶部按钮区域
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        self.select_btn = QPushButton("选择图片")
        self.process_btn = QPushButton("处理图片")
        self.save_btn = QPushButton("保存为PDF")
        self.clear_btn = QPushButton("清空列表")
        self.debug_btn = QPushButton("调试日志")
        self.help_btn = QPushButton("使用帮助")
        
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.debug_btn)
        button_layout.addWidget(self.help_btn)
        
        main_layout.addWidget(button_frame)
        
        # 处理选项区域
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        control_layout = QVBoxLayout(control_frame)
        
        # 语言选择
        lang_frame = QFrame()
        lang_layout = QHBoxLayout(lang_frame)
        lang_label = QLabel("主要识别语言:")
        self.lang_group = QButtonGroup()
        self.ch_radio = QRadioButton("简体中文")
        self.en_radio = QRadioButton("英文")
        self.ch_radio.setChecked(True)
        self.lang_group.addButton(self.ch_radio)
        self.lang_group.addButton(self.en_radio)
        
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.ch_radio)
        lang_layout.addWidget(self.en_radio)
        control_layout.addWidget(lang_frame)
        
        # 旋转模式选择
        rotation_frame = QFrame()
        rotation_layout = QHBoxLayout(rotation_frame)
        rotation_label = QLabel("旋转模式:")
        self.rotation_group = QButtonGroup()
        
        rotation_modes = [("自动检测", "auto"), ("旋转90°", "90"),
                         ("旋转180°", "180"), ("旋转270°", "270"),
                         ("不旋转", "0")]
        
        for text, mode in rotation_modes:
            radio = QRadioButton(text)
            if mode == "auto":
                radio.setChecked(True)
            self.rotation_group.addButton(radio)
            rotation_layout.addWidget(radio)
        
        control_layout.addWidget(rotation_frame)
        main_layout.addWidget(control_frame)
        
        # 文件列表区域
        list_frame = QFrame()
        list_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        list_layout = QVBoxLayout(list_frame)
        
        self.file_list = QListWidget()
        list_layout.addWidget(self.file_list)
        
        main_layout.addWidget(list_frame)
        
        # 预览区域
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        preview_layout = QVBoxLayout(preview_frame)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        main_layout.addWidget(preview_frame)
        
        # 状态栏和进度条
        self.statusBar().showMessage("准备就绪")
        self.progress_bar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # 连接信号
        self.select_btn.clicked.connect(self.select_files)
        self.process_btn.clicked.connect(self.process_images)
        self.save_btn.clicked.connect(self.save_as_pdf)
        self.clear_btn.clicked.connect(self.clear_files)
        self.debug_btn.clicked.connect(self.show_debug_log)
        self.help_btn.clicked.connect(self.show_help)
        self.file_list.currentRowChanged.connect(self.on_file_select)
        
    def setup_drag_drop(self):
        """设置拖放功能"""
        if TKDND_AVAILABLE:
            try:
                # 设置文件列表接收拖拽
                self.file_list.setAcceptDrops(True)
                self.file_list.dragEnterEvent = self.dragEnterEvent
                self.file_list.dragMoveEvent = self.dragMoveEvent
                
                # 设置预览区接收拖拽
                self.preview_label.setAcceptDrops(True)
                self.preview_label.dragEnterEvent = self.dragEnterEvent
                self.preview_label.dragMoveEvent = self.dragMoveEvent
                
                # 设置整个窗口接收拖拽
                self.setAcceptDrops(True)
                
                self.statusBar().showMessage("准备就绪 - 支持拖拽文件")
                self.logger.info("已启用TkinterDnD拖放支持")
            except Exception as e:
                self.logger.error(f"TkinterDnD设置错误: {str(e)}", exc_info=True)
                self.setup_drag_drop_fallback()
        else:
            # 使用内置Tkinter拖放支持(有限)
            self.setup_drag_drop_fallback()
    
    def setup_drag_drop_fallback(self):
        """备用拖放设置方法"""
        try:
            # 使用普通Tkinter的拖放支持
            self.setAcceptDrops(True)
            
            # 尝试Windows特有的拖放支持
            if sys.platform == 'win32':
                self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
                
            self.statusBar().showMessage("准备就绪 - 拖拽功能可能受限")
            self.logger.info("已启用备用拖放支持")
        except Exception as e:
            self.logger.error(f"备用拖放设置错误: {str(e)}", exc_info=True)
    
    def dragEnterEvent(self, event):
        """处理拖放进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """处理拖放移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """处理拖放事件"""
        try:
            # 获取拖放的文件数据
            urls = event.mimeData().urls()
            
            self.logger.info(f"收到拖拽数据: {urls}")
                
            # 解析拖拽的文件列表
            files = self.parse_dropped_data(urls)
            
            self.logger.info(f"解析后的文件列表: {files}")
                
            self.process_dropped_files(files)
                
        except Exception as e:
            self.logger.error(f"拖放处理错误: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "警告", f"处理拖放文件时出错: {str(e)}")
    
    def parse_dropped_data(self, urls):
        """解析拖拽数据，提取文件路径列表"""
        files = []
        
        # 如果数据为空，直接返回
        if not urls:
            return files
        
        self.logger.debug(f"正在解析拖拽数据: {urls}")
        
        try:
            # 处理每个URL
            for url in urls:
                path = url.toLocalFile() if not isinstance(url, str) else url
                path = path.strip('"\'').strip()
                
                # Windows路径规则化
                if sys.platform == 'win32':
                    path = path.replace('/', '\\')
                
                if path and os.path.exists(path):
                    files.append(path)
                    self.logger.debug(f"添加有效路径: {path}")
            
            # 如果没有找到有效文件，尝试其他方法
            if not files and urls:
                first_url = urls[0]
                path = first_url.toLocalFile() if not isinstance(first_url, str) else first_url
                path = path.strip('"\'').strip()
                
                if sys.platform == 'win32':
                    path = path.replace('/', '\\')
                
                if path and os.path.exists(path):
                    files.append(path)
                    self.logger.debug(f"使用备用方法添加路径: {path}")
        
        except Exception as e:
            self.logger.error(f"解析路径时出错: {str(e)}", exc_info=True)
        
        # 最终验证
        files = [f for f in files if f.strip() and os.path.exists(f)]
        self.logger.info(f"最终解析结果: {files}")
        return files
    
    def process_dropped_files(self, files):
        """处理拖放的文件列表"""
        # 过滤出有效的图片文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')
        valid_files = []
        
        for file_path in files:
            # 移除可能的引号和空格
            file_path = file_path.strip('"\' ')
            
            self.logger.debug(f"处理文件: {file_path}")
                
            # 检查文件是否存在且是图片
            if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in valid_extensions):
                valid_files.append(file_path)
                self.logger.debug(f"有效图片文件: {file_path}")
            else:
                self.logger.warning(f"无效文件或不是图片: {file_path}")
        
        if valid_files:
            # 添加到已选文件列表
            original_count = len(self.selected_files)
            self.selected_files.extend(valid_files)
            # 去重
            self.selected_files = list(dict.fromkeys(self.selected_files))
            final_count = len(self.selected_files)
            
            self.logger.info(f"添加了 {len(valid_files)} 个文件，有效添加 {final_count - original_count} 个（去重后）")
            
            self.update_file_list()
            self.process_btn.setEnabled(True)
            self.statusBar().showMessage(f"已添加 {len(valid_files)} 个文件，共 {len(self.selected_files)} 个")
            
            # 显示第一张新添加的图片预览
            self.show_preview(valid_files[0])
        else:
            self.logger.warning("没有添加有效的图片文件")
            QMessageBox.information(self, "提示", "没有添加有效的图片文件\n支持的格式: jpg, jpeg, png, bmp, tiff, gif")
    
    def update_file_list(self):
        """更新文件列表显示"""
        self.file_list.clear()
        for file in self.selected_files:
            # 只显示文件名，不显示完整路径
            self.file_list.addItem(os.path.basename(file))
        
        # 绑定点击事件
        self.file_list.currentRowChanged.connect(self.on_file_select)
    
    def on_file_select(self, index):
        """当用户选择列表中的文件时显示预览"""
        if index < 0 or index >= len(self.selected_files):
            return
        
            self.show_preview(self.selected_files[index])
    
    def show_preview(self, filepath_or_image):
        """显示图片预览
        Args:
            filepath_or_image: 可以是图片文件路径(str)或PIL Image对象
        """
        try:
            if isinstance(filepath_or_image, str):
                # 处理文件路径
                img = QImage(filepath_or_image)
                if img.isNull():
                    raise Exception("无法加载图片文件")
            else:
                # 处理PIL Image对象
                img_array = np.array(filepath_or_image)
                height, width = img_array.shape[:2]
                if len(img_array.shape) == 2:  # 灰度图
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                img = QImage(img_array.data, width, height, width * 3, QImage.Format_RGB888)
            
            # 等比例缩放图片到固定大小
            pixmap = QPixmap.fromImage(img)
            scaled_pixmap = pixmap.scaled(
                800, 800,  # 使用更大的固定预览尺寸
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            self.logger.debug(f"成功预览图片")
            
        except Exception as e:
            self.logger.error(f"预览图片出错: {str(e)}", exc_info=True)
            self.preview_label.clear()
            self.preview_label.setText("预览失败")
    
    def initialize_ocr(self):
        """初始化OCR引擎"""
        try:
            if self.reader is None:
                self.statusBar().showMessage("初始化OCR引擎中，请稍候...")
                
                # 检测GPU是否可用
                gpu_available = False
                gpu_type = "none"
                try:
                    import torch
                    if torch.cuda.is_available():  # 检测NVIDIA GPU
                        gpu_info = torch.cuda.get_device_properties(0)
                        if 'NVIDIA' in gpu_info.name:
                            gpu_available = True
                            gpu_type = "nvidia"
                            # 设置CUDA内存分配策略
                            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
                            torch.backends.cudnn.benchmark = True  # 优化性能
                    elif hasattr(torch, 'has_rocm') and torch.has_rocm:  # 检测AMD GPU
                        gpu_available = True
                        gpu_type = "amd"
                        # ROCm特定设置可以在这里添加
                except Exception as e:
                    self.logger.warning(f"GPU检测出错: {str(e)}")

                # 根据用户选择确定语言配置
                selected_lang = self.ch_radio.text()
                main_lang = self.lang_map[selected_lang]
                languages = [main_lang]
                
                if main_lang != "en":
                    languages.append("en")
                
                self.logger.info(f"初始化OCR引擎，使用语言: {languages}")
                self.logger.info(f"GPU状态: {gpu_type.upper() if gpu_available else 'CPU模式'}")
                
                # 初始化OCR引擎
                self.reader = easyocr.Reader(languages, gpu=gpu_available)
                
                # 显示GPU状态
                status_msg = "OCR引擎已准备就绪"
                if not gpu_available:
                    status_msg += " (GPU加速未启用，处理速度可能较慢)"
                    QMessageBox.information(self, "提示", 
                        "未检测到支持的GPU，OCR处理将使用CPU模式运行，可能会较慢。\n"
                        "建议：\n"
                        "1. 如果您有NVIDIA显卡，请安装CUDA工具包\n"
                        "2. 如果您有AMD显卡，请安装ROCm和支持ROCm的PyTorch\n"
                        "3. 确保已安装正确版本的PyTorch\n"
                        "4. 重启应用以启用GPU加速")
                else:
                    status_msg += f" ({gpu_type.upper()} GPU加速已启用)"
                
                self.statusBar().showMessage(status_msg)
                
        except Exception as e:
            self.logger.error(f"初始化OCR引擎时出错: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "OCR初始化错误", f"初始化OCR引擎时出错: {str(e)}")
            raise
    
    def process_images(self):
        """处理所有选择的图片"""
        if not self.selected_files:
            QMessageBox.information(self, "提示", "请先选择图片文件")
            return
        
        try:
            # 初始化OCR引擎（如果还没有初始化）
            if self.reader is None:
                self.initialize_ocr()
            
            # 禁用按钮，防止重复操作
            self.process_btn.setEnabled(False)
            self.select_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            
            # 如果存在旧的处理线程，确保它被清理
            if self.process_thread and self.process_thread.isRunning():
                self.process_thread.quit()
                self.process_thread.wait()
            
            # 创建并启动新的处理线程
            self.process_thread = ProcessThread(self.selected_files, self.reader, self.lang_group, self.rotation_group)
            self.process_thread.finished.connect(self.process_finished)
            self.process_thread.progress.connect(self.progress_bar.setValue)
            self.process_thread.status.connect(self.statusBar().showMessage)
            self.process_thread.error.connect(self.handle_process_error)
            self.process_thread.start()
            
        except Exception as e:
            self.logger.error(f"启动处理线程时出错: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "错误", f"启动处理失败: {str(e)}")
            self.enable_buttons()
    
    def handle_process_error(self, error_msg):
        """处理处理线程中的错误"""
        self.logger.error(f"处理过程出错: {error_msg}")
        QMessageBox.warning(self, "处理错误", error_msg)
        self.enable_buttons()
    
    def enable_buttons(self):
        """重新启用所有按钮"""
        self.process_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
    
    def process_finished(self, processed_images):
        """处理线程完成后的处理逻辑"""
        try:
            self.processed_images = processed_images
            self.update_file_list()
            self.enable_buttons()
            self.statusBar().showMessage(f"处理完成! 成功处理 {len(self.processed_images)} 张图片")
            
            # 显示第一张处理后的图片
            if self.processed_images:
                self.show_preview(self.processed_images[0])
            
        except Exception as e:
            self.logger.error(f"处理完成回调时出错: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "错误", f"更新界面时出错: {str(e)}")
            self.enable_buttons()
        
        # 清理线程
        if self.process_thread:
            self.process_thread.quit()
            self.process_thread.wait()
            self.process_thread = None
    
    def save_as_pdf(self):
        """将处理后的图片保存为PDF文件"""
        if not self.processed_images:
            QMessageBox.information(self, "提示", "没有已处理的图片可保存")
            return
        
        # 让用户选择保存位置
        initialdir = os.path.expanduser("~/Desktop")
        if not os.path.exists(initialdir):
            initialdir = os.path.expanduser("~")
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存PDF文件", initialdir, "PDF文件 (*.pdf);;所有文件 (*.*)")
        
        if not file_path:
            return  # 用户取消了保存
        
        try:
            self.statusBar().showMessage("正在生成PDF...")
            self.save_btn.setEnabled(False)
            
            # 创建临时文件夹存储处理后的图片
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_image_paths = []
                
                # 保存处理后的图片到临时文件
                for i, img in enumerate(self.processed_images):
                    temp_path = os.path.join(temp_dir, f"processed_{i}.jpg")
                    img.save(temp_path, "JPEG", quality=95)  # 提高保存质量
                    temp_image_paths.append(temp_path)
                
                # 生成PDF
                with open(file_path, "wb") as f:
                    f.write(img2pdf.convert(temp_image_paths))
            
            self.statusBar().showMessage(f"PDF已保存到: {file_path}")
            QMessageBox.information(self, "成功", f"PDF已成功保存到:\n{file_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "保存错误", f"保存PDF时出错: {str(e)}")
            self.statusBar().showMessage("保存PDF出错")
        finally:
            self.save_btn.setEnabled(True)

    def show_help(self):
        """显示帮助对话框"""
        help_text = """缘缘专用资料整理工具 - 使用说明

◆ 添加图片方法：
  1. 点击"选择图片"按钮从文件对话框中选择
  2. 直接将图片文件拖放到程序窗口中
  3. 可添加多组图片，会自动去重

◆ 图片处理功能：
  1. 选择合适的主要识别语言(中文/英文)
  2. 选择旋转模式:
     - 自动检测：程序根据文字方向自动判断
     - 指定角度：强制按指定角度旋转
  3. 点击"处理图片"按钮开始处理

◆ 旋转图片处理逻辑:
  1. 自动模式会在四个方向分别进行文字识别
  2. 程序会选择文字识别效果最好的方向
  3. 如自动旋转效果不理想，可选择手动指定角度

◆ 其他功能:
  1. 预览区可查看选中图片
  2. "清空列表"可删除所有已选图片
  3. "保存为PDF"将处理后的图片合并为PDF文件

◆ 注意事项:
  1. 首次启动时OCR引擎初始化可能较慢
  2. 处理大量图片时请耐心等待
  3. 文字识别准确度受图片质量影响
  4. 旋转功能适用于包含文字的图片
"""
        help_window = QMessageBox(self)
        help_window.setIcon(QMessageBox.Information)
        help_window.setWindowTitle("使用帮助")
        help_window.setText(help_text)
        help_window.setStandardButtons(QMessageBox.Ok)
        help_window.exec()

    def show_debug_log(self):
        """显示调试日志对话框"""
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(log_dir):
            QMessageBox.information(self, "提示", "日志目录不存在")
            return
            
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if not log_files:
            QMessageBox.information(self, "提示", "没有找到日志文件")
            return
            
        # 按修改时间排序，显示最新的日志
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        latest_log = os.path.join(log_dir, log_files[0])
        
        # 创建日志查看器窗口
        log_window = QMessageBox(self)
        log_window.setIcon(QMessageBox.Information)
        log_window.setWindowTitle("调试日志")
        log_window.setText(f"查看最新的日志文件: {latest_log}")
        log_window.setStandardButtons(QMessageBox.Ok)
        log_window.exec()

    def select_files(self):
        """选择图片文件"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif);;所有文件 (*.*)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            self.selected_files.extend(files)
            # 去重
            self.selected_files = list(dict.fromkeys(self.selected_files))
            
            self.update_file_list()
            self.process_btn.setEnabled(True)
            self.statusBar().showMessage(f"已添加 {len(files)} 个文件，共 {len(self.selected_files)} 个")
            
            # 显示第一张新添加的图片预览
            if files:
                self.show_preview(files[0])
    
    def clear_files(self):
        """清空文件列表"""
        self.selected_files = []
        self.processed_images = []
        self.file_list.clear()
        self.preview_label.clear()
        self.process_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("已清空文件列表")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        # 判断是否有TkinterDnD支持
        if TKDND_AVAILABLE:
            window = ImageProcessorWindow()
            window.show()
        else:
            window = ImageProcessorWindow()
            window.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        QMessageBox.warning(None, "程序错误", f"程序启动出错: {str(e)}\n{traceback.format_exc()}") 
