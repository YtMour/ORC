#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QListWidget, QLabel, 
                            QProgressBar, QMessageBox, QListWidgetItem, QMenu, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QPixmap, QImage, QIcon, QCursor, QDrag

from src.core.image_processor import ImageProcessor
from src.core.pdf_converter import convert_to_pdf
from src.utils.file_utils import get_temp_path, cleanup_temp_files


class ProcessThread(QThread):
    """
    处理图片的工作线程
    """
    progress_signal = pyqtSignal(int, int)  # 当前进度, 总数
    result_signal = pyqtSignal(str, str)  # 图片路径, 处理结果路径
    finished_signal = pyqtSignal(list)  # 所有处理后的图片路径
    error_signal = pyqtSignal(str)  # 错误信息

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.processor = ImageProcessor()
        self.processed_paths = []
        
        # 添加批量处理分析
        self.batch_mode = len(image_paths) > 1  # 只有多张图片时才启用批量模式

    def run(self):
        try:
            # 启用批量处理模式，确保旋转方向一致
            if self.batch_mode:
                self.processor.start_batch_processing()
                logging.info("批量处理模式：启动旋转方向一致性控制")
            
            # 处理每张图片，第一遍先分析确定一致方向
            if self.batch_mode:
                logging.info("批量处理模式：第一轮分析确定最佳旋转方向")
                # 先对每张图片进行单独分析，不保存结果，只收集旋转方向数据
                for i, image_path in enumerate(self.image_paths):
                    try:
                        # 处理单张图片
                        self.processor.process_image(image_path)
                        # 更新进度 (以50%进度计算第一轮)
                        self.progress_signal.emit(i + 1, len(self.image_paths))
                    except Exception as e:
                        logging.error(f"预分析图片 {os.path.basename(image_path)} 失败: {str(e)}")
                
                # 第一轮完成后，确定最佳一致旋转方向
                self.processor.end_batch_processing()
                consistent_angle = self.processor.get_consistent_rotation()
                logging.info(f"批量处理模式：确定最终一致旋转角度为 {consistent_angle}度")
            
            # 第二轮处理，应用一致旋转角度处理所有图片
            logging.info("批量处理模式：开始最终处理并应用一致旋转角度")
            self.processed_paths = []
            
            for i, image_path in enumerate(self.image_paths):
                try:
                    # 处理单张图片
                    result_path = self.processor.process_image(image_path)
                    self.processed_paths.append(result_path)
                    self.result_signal.emit(image_path, result_path)
                except Exception as e:
                    self.error_signal.emit(f"处理图片 {os.path.basename(image_path)} 失败: {str(e)}")
                    continue
                finally:
                    if self.batch_mode:
                        # 更新进度 (第二轮从50%到100%)
                        progress = len(self.image_paths) + i + 1
                        total = len(self.image_paths) * 2
                        self.progress_signal.emit(progress, total)
                    else:
                        # 单图模式正常进度
                        self.progress_signal.emit(i + 1, len(self.image_paths))
            
            # 完成所有处理
            self.finished_signal.emit(self.processed_paths)
        except Exception as e:
            self.error_signal.emit(f"处理过程中发生错误: {str(e)}")


class PdfThread(QThread):
    """
    转换PDF的工作线程
    """
    finished_signal = pyqtSignal(str)  # PDF文件路径
    error_signal = pyqtSignal(str)  # 错误信息

    def __init__(self, image_paths, output_path):
        super().__init__()
        self.image_paths = image_paths
        self.output_path = output_path

    def run(self):
        try:
            result_path = convert_to_pdf(self.image_paths, self.output_path)
            self.finished_signal.emit(result_path)
        except Exception as e:
            self.error_signal.emit(f"转换PDF失败: {str(e)}")


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片方向矫正与PDF转换工具")
        self.setMinimumSize(800, 600)
        
        # 设置应用程序图标
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 图片路径列表
        self.image_paths = []
        self.processed_paths = []
        
        # 初始化UI
        self.setup_ui()
        
    def setup_ui(self):
        """设置界面布局"""
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("选择图片")
        self.select_btn.clicked.connect(self.select_images)
        button_layout.addWidget(self.select_btn)
        
        self.process_btn = QPushButton("处理图片")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.convert_btn = QPushButton("转换为PDF")
        self.convert_btn.clicked.connect(self.convert_to_pdf)
        self.convert_btn.setEnabled(False)
        button_layout.addWidget(self.convert_btn)
        
        # 添加旋转按钮
        rotate_layout = QHBoxLayout()
        
        self.rotate_right_btn = QPushButton("顺时针旋转")
        self.rotate_right_btn.clicked.connect(lambda: self._batch_rotate_images(90))
        self.rotate_right_btn.setEnabled(False)
        rotate_layout.addWidget(self.rotate_right_btn)
        
        self.rotate_left_btn = QPushButton("逆时针旋转")
        self.rotate_left_btn.clicked.connect(lambda: self._batch_rotate_images(270))
        self.rotate_left_btn.setEnabled(False)
        rotate_layout.addWidget(self.rotate_left_btn)
        
        self.make_same_dir_btn = QPushButton("统一旋转方向")
        self.make_same_dir_btn.clicked.connect(self._make_all_same_direction)
        self.make_same_dir_btn.setEnabled(False)
        rotate_layout.addWidget(self.make_same_dir_btn)
        
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_images)
        rotate_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(button_layout)
        main_layout.addLayout(rotate_layout)
        
        # 图片列表区域
        self.list_widget = QListWidget()
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setSpacing(10)
        
        # 启用拖放功能
        self.list_widget.setAcceptDrops(True)
        self.list_widget.setDragEnabled(True)
        self.list_widget.setDragDropMode(QListWidget.DropOnly)
        self.list_widget.dragEnterEvent = self._dragEnterEvent
        self.list_widget.dragMoveEvent = self._dragMoveEvent
        self.list_widget.dropEvent = self._dropEvent
        
        # 添加右键菜单
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        
        main_layout.addWidget(self.list_widget)
        
        # 状态和进度条区域
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("就绪 - 可以拖放图片到窗口进行添加")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)
        
        # 设置主窗口
        self.setCentralWidget(main_widget)
        
        # 也允许整个窗口接收拖放
        self.setAcceptDrops(True)
    
    def select_images(self):
        """选择图片文件"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff)")
        
        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            
            # 清空现有列表
            self.list_widget.clear()
            self.processed_paths = []
            
            # 添加到列表并显示缩略图
            for image_path in self.image_paths:
                self._add_image_to_list(image_path)
            
            # 更新UI状态
            self.process_btn.setEnabled(len(self.image_paths) > 0)
            self.convert_btn.setEnabled(False)
            self.status_label.setText(f"已选择 {len(self.image_paths)} 张图片")
    
    def _add_image_to_list(self, image_path, is_processed=False):
        """添加图片到列表控件"""
        try:
            # 添加更详细的日志
            print(f"尝试加载图片: {image_path}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "错误", f"图片文件不存在: {image_path}")
                return
            
            # 使用PIl尝试打开图片进行验证
            try:
                from PIL import Image
                img = Image.open(image_path)
                img.verify()  # 验证图像文件完整性
                print(f"PIL验证成功: {image_path}, 格式: {img.format}, 大小: {img.size}")
            except Exception as pil_err:
                print(f"PIL验证失败: {str(pil_err)}")
            
            # 创建缩略图
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise ValueError(f"QPixmap无法加载图片: {image_path}")
            
            pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 创建列表项
            item = QListWidgetItem()
            
            # 创建QIcon并设置图标 - 修复类型错误
            icon = QIcon(pixmap)
            item.setIcon(icon)
            
            # 设置标题为文件名
            filename = os.path.basename(image_path)
            if is_processed:
                filename = f"[已处理] {filename}"
            item.setText(filename)
            
            # 设置数据
            item.setData(Qt.UserRole, image_path)
            
            # 添加到列表
            self.list_widget.addItem(item)
            print(f"成功添加图片到列表: {filename}")
        except Exception as e:
            print(f"添加图片失败详细错误: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"无法加载图片 {os.path.basename(image_path)}: {str(e)}")
    
    def process_images(self):
        """处理图片"""
        if not self.image_paths:
            return
        
        # 禁用按钮，避免重复处理
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        
        # 使用本地自定义方案，不再尝试导入OCR
        ocr_available = False
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 根据图片数量设置进度条范围
        total_images = len(self.image_paths)
        if total_images > 1:
            # 批量处理模式：两轮处理，所以最大值是图片数量的2倍
            self.progress_bar.setMaximum(total_images * 2)
            self.progress_bar.setValue(0)
        else:
            # 单图模式：最大值就是图片数量
            self.progress_bar.setMaximum(total_images)
            self.progress_bar.setValue(0)
        
        # 更新状态
        self.status_label.setText(f"正在处理 {total_images} 张图片，检测文字方向...")
        
        # 创建并启动工作线程
        self.process_thread = ProcessThread(self.image_paths)
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.result_signal.connect(self.update_result)
        self.process_thread.finished_signal.connect(self.process_finished)
        self.process_thread.error_signal.connect(self.show_error)
        self.process_thread.start()
    
    def update_progress(self, current, total):
        """更新进度条"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"处理进度: {current}/{total}")
    
    def update_result(self, original_path, processed_path):
        """更新处理结果"""
        # 暂不处理单个结果更新，等待全部完成后更新UI
        pass
    
    def process_finished(self, processed_paths):
        """处理完成"""
        # 保存处理后的图片路径
        self.processed_paths = processed_paths
        
        # 清空现有列表
        self.list_widget.clear()
        
        # 添加处理后的图片到列表
        for path in self.processed_paths:
            self._add_image_to_list(path, True)
        
        # 更新UI状态
        self.select_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.convert_btn.setEnabled(len(self.processed_paths) > 0)
        self.rotate_right_btn.setEnabled(len(self.processed_paths) > 0)
        self.rotate_left_btn.setEnabled(len(self.processed_paths) > 0)
        self.make_same_dir_btn.setEnabled(len(self.processed_paths) > 1)  # 多于一张图片时才启用
        
        self.status_label.setText(f"处理完成，成功处理 {len(self.processed_paths)} 张图片")
    
    def convert_to_pdf(self):
        """将处理后的图片转换为PDF"""
        if not self.processed_paths:
            return
        
        # 选择保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存PDF文件", "", "PDF文件 (*.pdf)")
        
        if not file_path:
            return
        
        # 如果没有.pdf后缀，添加它
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'
        
        # 禁用按钮
        self.convert_btn.setEnabled(False)
        
        # 更新状态
        self.status_label.setText("正在转换为PDF...")
        
        # 创建并启动工作线程
        self.pdf_thread = PdfThread(self.processed_paths, file_path)
        self.pdf_thread.finished_signal.connect(self.pdf_finished)
        self.pdf_thread.error_signal.connect(self.show_error)
        self.pdf_thread.start()
    
    def pdf_finished(self, pdf_path):
        """PDF转换完成"""
        self.convert_btn.setEnabled(True)
        self.status_label.setText(f"PDF转换完成: {pdf_path}")
        
        # 询问是否打开PDF
        reply = QMessageBox.question(
            self, "转换完成", 
            f"PDF已成功保存到: {pdf_path}\n是否打开文件?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # 打开PDF文件
            os.startfile(pdf_path)
    
    def clear_images(self):
        """清空图片列表"""
        self.list_widget.clear()
        self.image_paths = []
        self.processed_paths = []
        
        # 更新UI状态
        self.process_btn.setEnabled(False)
        self.convert_btn.setEnabled(False)
        self.status_label.setText("就绪")
        
        # 清理临时文件
        cleanup_temp_files()
    
    def show_error(self, error_message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", error_message)
    
    def closeEvent(self, event):
        """关闭窗口时清理临时文件"""
        cleanup_temp_files()
        event.accept()

    def dragEnterEvent(self, event):
        """窗口接收拖放事件"""
        self._dragEnterEvent(event)
    
    def dragMoveEvent(self, event):
        """窗口接收拖放移动事件"""
        self._dragMoveEvent(event)
    
    def dropEvent(self, event):
        """窗口接收拖放释放事件"""
        self._dropEvent(event)

    def _dragEnterEvent(self, event):
        """处理拖拽进入事件"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                # 检查是否为支持的图片格式
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _dragMoveEvent(self, event):
        """处理拖拽移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _dropEvent(self, event):
        """处理拖拽释放事件"""
        if event.mimeData().hasUrls():
            file_paths = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                # 检查是否为支持的图片格式
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    file_paths.append(file_path)
            
            if file_paths:
                # 清空现有列表
                self.list_widget.clear()
                self.processed_paths = []
                self.image_paths = file_paths
                
                # 添加拖放的图片到列表
                for image_path in file_paths:
                    self._add_image_to_list(image_path)
                
                # 更新UI状态
                self.process_btn.setEnabled(len(self.image_paths) > 0)
                self.convert_btn.setEnabled(False)
                self.status_label.setText(f"已添加 {len(self.image_paths)} 张图片")
                
                event.acceptProposedAction()
                return
        
        event.ignore()

    def _show_context_menu(self, position):
        """显示右键菜单"""
        item = self.list_widget.itemAt(position)
        
        # 创建菜单
        context_menu = QMenu()
        
        # 旋转选项子菜单
        rotate_menu = QMenu("旋转图片", context_menu)
        
        if item:  # 选中了某个项目
            rotate_90_action = QAction("顺时针旋转90°", self)
            rotate_90_action.triggered.connect(lambda: self._rotate_image(item, 90))
            rotate_menu.addAction(rotate_90_action)
            
            rotate_180_action = QAction("旋转180°", self)
            rotate_180_action.triggered.connect(lambda: self._rotate_image(item, 180))
            rotate_menu.addAction(rotate_180_action)
            
            rotate_270_action = QAction("逆时针旋转90°", self)
            rotate_270_action.triggered.connect(lambda: self._rotate_image(item, 270))
            rotate_menu.addAction(rotate_270_action)
        
        # 批量旋转选项
        rotate_menu.addSeparator()
        
        batch_rotate_90_action = QAction("所有图片顺时针旋转90°", self)
        batch_rotate_90_action.triggered.connect(lambda: self._batch_rotate_images(90))
        rotate_menu.addAction(batch_rotate_90_action)
        
        batch_rotate_270_action = QAction("所有图片逆时针旋转90°", self)
        batch_rotate_270_action.triggered.connect(lambda: self._batch_rotate_images(270))
        rotate_menu.addAction(batch_rotate_270_action)
        
        # 统一旋转方向选项
        rotate_menu.addSeparator()
        
        force_rotate_90_action = QAction("强制统一到顺时针90°方向", self)
        force_rotate_90_action.triggered.connect(lambda: self._set_default_rotation(90))
        rotate_menu.addAction(force_rotate_90_action)
        
        force_rotate_270_action = QAction("强制统一到逆时针90°方向", self)
        force_rotate_270_action.triggered.connect(lambda: self._set_default_rotation(270))
        rotate_menu.addAction(force_rotate_270_action)
        
        context_menu.addMenu(rotate_menu)
        
        # 在当前位置显示菜单
        context_menu.exec_(QCursor.pos())
    
    def _rotate_image(self, item, angle):
        """手动旋转选中的图片"""
        image_path = item.data(Qt.UserRole)
        self.status_label.setText(f"正在旋转图片: {os.path.basename(image_path)}")
        
        try:
            # 创建图像处理器
            processor = ImageProcessor()
            
            # 使用正确的force_rotate方法
            result_path = processor.force_rotate(image_path, angle)
            
            # 更新列表项
            self._update_list_item(item, result_path)
            
            self.status_label.setText(f"图片已旋转 {angle}°")
            
            # 将旋转后的图片添加到已处理图片列表中
            # 更新已处理路径列表
            if image_path in self.processed_paths:
                # 替换原有路径
                index = self.processed_paths.index(image_path)
                self.processed_paths[index] = result_path
            else:
                # 添加新路径
                self.processed_paths.append(result_path)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"旋转图片时出错: {str(e)}")
            self.status_label.setText("旋转图片失败")
    
    def _update_list_item(self, item, new_image_path):
        """更新列表项显示新图片"""
        try:
            # 创建缩略图
            pixmap = QPixmap(new_image_path)
            if pixmap.isNull():
                raise ValueError(f"无法加载图片: {new_image_path}")
            
            pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 创建QIcon并设置图标
            icon = QIcon(pixmap)
            item.setIcon(icon)
            
            # 设置新文件路径
            item.setData(Qt.UserRole, new_image_path)
            
            # 更新文件名
            filename = os.path.basename(new_image_path)
            if "已处理" not in item.text():
                item.setText(f"[已处理] {filename}")
            else:
                item.setText(f"[已处理] {filename}")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"更新图片失败: {str(e)}")
    
    def _batch_rotate_images(self, angle):
        """批量旋转所有图片"""
        if not self.processed_paths:
            QMessageBox.information(self, "提示", "请先处理图片后再使用批量旋转功能")
            return
            
        # 确认提示
        direction = "顺时针" if angle == 90 else "逆时针"
        reply = QMessageBox.question(
            self, "确认批量旋转", 
            f"确定要将所有图片{direction}旋转{abs(angle)}度吗？\n这将更新图片的方向，影响最终PDF。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        self.status_label.setText(f"正在旋转所有图片...")
        
        try:
            # 清空当前列表
            self.list_widget.clear()
            
            # 新的处理结果路径
            new_processed_paths = []
            
            # 创建图像处理器
            processor = ImageProcessor()
            
            # 处理每张图片
            for i, image_path in enumerate(self.processed_paths):
                try:
                    # 使用force_rotate方法进行强制旋转
                    result_path = processor.force_rotate(image_path, angle)
                    new_processed_paths.append(result_path)
                    
                    # 添加到列表
                    self._add_image_to_list(result_path, True)
                    
                    # 更新进度
                    self.progress_bar.setValue(int((i+1) / len(self.processed_paths) * 100))
                    self.status_label.setText(f"旋转进度: {i+1}/{len(self.processed_paths)}")
                    
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"旋转图片 {os.path.basename(image_path)} 失败: {str(e)}")
                    
            # 更新处理路径
            self.processed_paths = new_processed_paths
            
            self.status_label.setText(f"已完成所有图片旋转")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量旋转过程中出错: {str(e)}")
            self.status_label.setText("批量旋转失败")
            
    def _set_default_rotation(self, angle):
        """为所有图片设置默认旋转方向"""
        if not self.processed_paths:
            QMessageBox.information(self, "提示", "请先处理图片后再使用此功能")
            return
            
        direction = "顺时针" if angle == 90 else ("逆时针" if angle == 270 else "")
        reply = QMessageBox.question(
            self, "确认操作", 
            f"确定要将所有图片统一{direction}旋转到{angle}度方向吗？\n这将覆盖自动检测和之前的旋转结果。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        self.status_label.setText(f"正在将所有图片旋转到{angle}度方向...")
        
        try:
            # 禁用所有按钮，防止操作冲突
            self.select_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.convert_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.rotate_right_btn.setEnabled(False)
            self.rotate_left_btn.setEnabled(False)
            self.make_same_dir_btn.setEnabled(False)
            
            # 清空当前列表
            self.list_widget.clear()
            
            # 新的处理结果路径
            new_processed_paths = []
            
            # 创建图像处理器
            processor = ImageProcessor()
            
            # 更新进度条
            self.progress_bar.setRange(0, len(self.processed_paths))
            self.progress_bar.setValue(0)
            
            # 处理每张图片
            for i, image_path in enumerate(self.processed_paths):
                try:
                    # 使用force_rotate方法进行强制旋转
                    result_path = processor.force_rotate(image_path, angle)
                    new_processed_paths.append(result_path)
                    
                    # 添加到列表
                    self._add_image_to_list(result_path, True)
                    
                    # 更新进度
                    self.progress_bar.setValue(i + 1)
                    self.status_label.setText(f"旋转进度: {i+1}/{len(self.processed_paths)}")
                    
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"旋转图片 {os.path.basename(image_path)} 失败: {str(e)}")
                
            # 更新处理路径
            self.processed_paths = new_processed_paths
            
            self.status_label.setText(f"已完成所有图片旋转到{angle}度")
            
            # 重新启用按钮
            self.select_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.convert_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_right_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_left_btn.setEnabled(len(self.processed_paths) > 0)
            self.make_same_dir_btn.setEnabled(len(self.processed_paths) > 1)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"统一旋转方向过程中出错: {str(e)}")
            self.status_label.setText("统一方向失败")
            print(f"统一方向失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 重新启用按钮
            self.select_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.convert_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_right_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_left_btn.setEnabled(len(self.processed_paths) > 0)
            self.make_same_dir_btn.setEnabled(len(self.processed_paths) > 1)
    
    def _make_all_same_direction(self):
        """统一所有图片的旋转方向，优先考虑文字方向"""
        if not self.processed_paths:
            QMessageBox.information(self, "提示", "请先处理图片后再使用统一旋转方向功能")
            return
            
        self.status_label.setText("正在分析图片文字方向...")
        
        try:
            # 禁用所有按钮，防止操作冲突
            self.select_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.convert_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.rotate_right_btn.setEnabled(False)
            self.rotate_left_btn.setEnabled(False)
            self.make_same_dir_btn.setEnabled(False)
            
            # 使用新的批处理API，运行一个新的处理线程
            self.status_label.setText("正在分析并统一图片方向...")
            
            # 收集当前所有已处理图片的原始路径
            original_paths = []
            for processed_path in self.processed_paths:
                # 尝试从元数据读取原始路径
                meta_path = processed_path + ".meta"
                original_path = None
                
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("original_path="):
                                    original_path = line.strip().split("=", 1)[1]
                                    break
                except:
                    pass
                
                # 如果找不到原始路径，使用当前处理后的路径
                if original_path is None or not os.path.exists(original_path):
                    original_path = processed_path
                    
                original_paths.append(original_path)
            
            # 创建进度条
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            
            # 创建并启动处理线程
            self.process_thread = ProcessThread(original_paths)
            
            # 连接信号
            self.process_thread.progress_signal.connect(self.update_progress)
            self.process_thread.result_signal.connect(self.update_result)
            self.process_thread.finished_signal.connect(self.process_finished)
            self.process_thread.error_signal.connect(self.show_error)
            
            # 开始处理
            self.process_thread.start()
            
            # 清空当前列表
            self.list_widget.clear()
            self.processed_paths = []
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"统一旋转方向过程中出错: {str(e)}")
            self.status_label.setText("统一方向失败")
            print(f"统一方向失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 重新启用按钮
            self.select_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.convert_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_right_btn.setEnabled(len(self.processed_paths) > 0)
            self.rotate_left_btn.setEnabled(len(self.processed_paths) > 0)
            self.make_same_dir_btn.setEnabled(len(self.processed_paths) > 1) 