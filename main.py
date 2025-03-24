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
        
        # 设置日志记录器
        self.logger = logging.getLogger("ImageProcessor.ProcessThread")
    
    def run(self):
        """线程运行函数"""
        try:
            total = len(self.files)
            for i, file_path in enumerate(self.files):
                self.status.emit(f"处理图片 {i+1}/{total}: {os.path.basename(file_path)}")
                self.progress.emit(int((i / total) * 100))
                
                # 处理单个图片
                processed_img = self.process_single_image(file_path)
                if processed_img:
                    self.processed_images.append(processed_img)
                
                # 短暂暂停，避免界面卡顿
                time.sleep(0.1)
            
            self.progress.emit(100)
            self.status.emit(f"处理完成! 成功处理 {len(self.processed_images)} 张图片")
            self.finished.emit(self.processed_images)
            
        except Exception as e:
            self.logger.error(f"处理图片时出错: {str(e)}", exc_info=True)
            error_msg = f"处理图片时出错: {str(e)}\n{traceback.format_exc()}" if self.debug_mode else f"处理图片时出错: {str(e)}"
            self.error.emit(error_msg)
    
    def process_single_image(self, file_path):
        """处理单个图片，识别文字并确定正确方向"""
        try:
            # 打开图片
            img = Image.open(file_path)
            
            # 检查旋转模式
            rotation_mode = "auto"  # 默认自动模式
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
                    img = img.rotate(rotation_angle, expand=True)
                return img
            
            # 自动模式：执行OCR识别和角度判断
            # 转换为OpenCV格式以便处理
            img_np = np.array(img)
            
            # 检查图片是否为灰度图
            if len(img_np.shape) == 2:
                # 灰度图转为三通道
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                # RGBA转为RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
            # 尝试在四个方向上进行OCR识别，选择结果最好的方向
            orientations = [0, 90, 180, 270]
            best_confidence = -1
            best_orientation = 0
            best_result = None
            
            # 获取原始OCR结果
            results = self.reader.readtext(img_np)
            if results:
                # 获取原始方向的总置信度
                total_confidence = sum(prob for _, _, prob in results)
                best_confidence = total_confidence
                best_result = results
                if self.debug_mode:
                    self.logger.debug(f"原始方向(0°)识别到{len(results)}个文本，总置信度: {total_confidence}")
            else:
                if self.debug_mode:
                    self.logger.debug(f"原始方向未识别到文本")
            
            # 尝试其他三个方向
            for angle in [90, 180, 270]:
                # 旋转图像进行测试
                rotated = self.rotate_image(img_np, angle)
                rotated_results = self.reader.readtext(rotated)
                
                # 计算这个方向的总置信度
                if rotated_results:
                    angle_confidence = sum(prob for _, _, prob in rotated_results)
                    
                    # 对90度和270度的旋转添加额外惩罚，因为垂直文本更难识别
                    if angle in [90, 270] and self.lang_group.checkedButton().text() == "简体中文":
                        # 中文文档对垂直方向有偏好
                        angle_confidence *= 1.2
                    
                    if self.debug_mode:
                        self.logger.debug(f"旋转{angle}°识别到{len(rotated_results)}个文本，总置信度: {angle_confidence}")
                    
                    # 如果识别文本数更多或置信度更高，则选择此方向
                    if angle_confidence > best_confidence or (len(rotated_results) > len(best_result or []) * 1.5):
                        best_confidence = angle_confidence
                        best_orientation = angle
                        best_result = rotated_results
                else:
                    if self.debug_mode:
                        self.logger.debug(f"旋转{angle}°未识别到文本")
            
            # 使用投票机制来确定最终的旋转方向
            final_orientation = self.confirm_orientation(img_np, results, best_orientation, best_result)
            
            # 如果需要旋转，就旋转图片
            if final_orientation != 0:
                img = img.rotate(final_orientation, expand=True)
                if self.debug_mode:
                    self.logger.debug(f"图片 {os.path.basename(file_path)} 旋转了 {final_orientation} 度")
            
            return img
            
        except Exception as e:
            self.logger.error(f"处理图片 {os.path.basename(file_path)} 时出错: {str(e)}", exc_info=True)
            return None
    
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
        """使用Hough线变换来确定图像方向"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 使用概率Hough变换检测线段
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) == 0:
                return 0  # 没有检测到线条，假设不需要旋转
            
            # 计算线段方向
            horizontal_count = 0
            vertical_count = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算线段角度
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # 将角度归一化到 [-90, 90]
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                
                # 判断线段是水平还是垂直
                if abs(angle) < 30:  # 接近水平
                    horizontal_count += 1
                elif abs(angle) > 60:  # 接近垂直
                    vertical_count += 1
            
            # 根据水平和垂直线段的数量判断方向
            if horizontal_count > vertical_count * 1.5:
                # 水平线明显多于垂直线，可能是正确方向或180度旋转
                return 0
            elif vertical_count > horizontal_count * 1.5:
                # 垂直线明显多于水平线，可能是90度或270度旋转
                # 默认选择90度旋转，如果需要更精确，可以添加更多判断
                return 90
            
            return 0  # 无明确方向时，不旋转
            
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"线条分析错误: {str(e)}", exc_info=True)
            return 0  # 出错时不旋转

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
        
        # Windows路径规则化
        def normalize_path(url):
            if isinstance(url, str):
                path = url
            else:
                path = url.toLocalFile()
            
            path = path.strip('"\'')
            path = path.strip()
            # 替换Windows中的正斜杠为反斜杠
            if sys.platform == 'win32':
                path = path.replace('/', '\\')
            return path
        
        try:
            # 处理每个URL
            for url in urls:
                if isinstance(url, str):
                    path = url
                else:
                    path = url.toLocalFile()
                
                # 规范化路径
                normalized_path = normalize_path(path)
                if normalized_path and os.path.exists(normalized_path):
                    files.append(normalized_path)
                    self.logger.debug(f"添加有效路径: {normalized_path}")
            
            # 如果没有找到有效文件，尝试其他方法
            if not files:
                # 尝试直接从第一个URL获取路径
                first_url = urls[0]
                if isinstance(first_url, str):
                    path = first_url
                else:
                    path = first_url.toLocalFile()
                
                normalized_path = normalize_path(path)
                if normalized_path and os.path.exists(normalized_path):
                    files.append(normalized_path)
                    self.logger.debug(f"使用备用方法添加路径: {normalized_path}")
        
        except Exception as e:
            self.logger.error(f"解析路径时出错: {str(e)}", exc_info=True)
        
        # 最终验证
        files = [f for f in files if f.strip() and os.path.exists(f)]  # 过滤空字符串和不存在的文件
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
    
    def show_preview(self, filepath):
        """显示图片预览"""
        try:
            img = Image.open(filepath)
            # 调整图片大小以适应预览区域
            img.thumbnail((400, 400))
            pixmap = QPixmap.fromImage(QImage(filepath))
            
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "预览错误", f"无法预览该图片: {str(e)}")
    
    def initialize_ocr(self):
        """初始化OCR引擎"""
        try:
            if self.reader is None:
                self.statusBar().showMessage("初始化OCR引擎中，请稍候...")
                
                # 根据用户选择确定语言配置
                selected_lang = self.ch_radio.text()
                main_lang = self.lang_map[selected_lang]
                languages = [main_lang]
                
                # 如果主语言不是英文，添加英文作为辅助语言
                if main_lang != "en":
                    languages.append("en")
                
                self.logger.info(f"初始化OCR引擎，使用语言: {languages}")
                self.reader = easyocr.Reader(languages)
                self.statusBar().showMessage("OCR引擎已准备就绪")
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