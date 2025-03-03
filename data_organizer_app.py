from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QGroupBox, QComboBox, QLabel, QCheckBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox
from PyQt5.QtCore import QDateTime
from organize_worker import OrganizeWorker
import os
import sys
class DataOrganizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        self.setWindowTitle("智能数据组织工具 v1.0")
        self.setGeometry(300, 300, 800, 600)

        # 主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 路径选择
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择或拖入数据集路径...")
        self.path_edit.setMinimumHeight(40)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)

        # 任务类型选择
        task_group = QGroupBox("任务配置")
        task_layout = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["分类任务", "分割任务"])
        # 连接信号和槽
        self.task_combo.currentTextChanged.connect(self.update_log)

        self.convert_combo = QComboBox()
        self.convert_combo.addItems(["无格式转换", "PNG转BMP", "BMP转PNG"])
        task_layout.addWidget(QLabel("任务类型:"))
        task_layout.addWidget(self.task_combo)
        task_layout.addWidget(QLabel("格式转换:"))
        task_layout.addWidget(self.convert_combo)
        task_group.setLayout(task_layout)

        # 统计选项
        self.stats_check = QCheckBox("生成统计报表")
        self.stats_check.setChecked(True)

        # 创建一个复选框
        self.rename_check = QCheckBox("重命名", self)
        # 设置复选框默认勾选
        self.rename_check.setChecked(False)
        # 新增可视化复选框 ▼▼▼
        self.visual_check = QCheckBox("生成分割可视化结果")
        self.visual_check.setChecked(False)
        self.visual_check.setToolTip("勾选后将生成标注可视化预览图")
        # 新增分割切图功能 ▼▼▼
        self.SegCrop_check = QCheckBox("分割切图")
        self.SegCrop_check.setChecked(False)

        # 重命名规则输入框
        rename_layout = QHBoxLayout()
        self.rename_edit = QLineEdit()
        self.rename_edit.setPlaceholderText("重命名规则，使用 {category} 和 {index} 占位符，如 {category}_{index}")
        rename_layout.addWidget(QLabel("重命名规则:"))
        rename_layout.addWidget(self.rename_edit)

        # 进度显示
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        # 操作按钮
        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self.start_processing)
        self.run_btn.setMinimumHeight(45)
        
        # 在操作按钮区域添加重启按钮
        self.restart_btn = QPushButton("重启应用")
        self.restart_btn.clicked.connect(self.restart_application)
        self.restart_btn.setMinimumHeight(45)
        # 创建一个水平布局容器
        checkboxes_layout = QHBoxLayout()
        checkboxes_layout.addWidget(self.stats_check)
        checkboxes_layout.addWidget(self.rename_check)
        checkboxes_layout.addWidget(self.visual_check)
        checkboxes_layout.addWidget(self.SegCrop_check)
        # 布局组装
        layout.addLayout(path_layout)
        layout.addWidget(task_group)
        # layout.addWidget(self.stats_check)
        # layout.addWidget(self.rename_check)
        layout.addLayout(checkboxes_layout) 
        layout.addLayout(rename_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_area)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.restart_btn)
        layout.addLayout(button_layout)

        # 启用拖放
        self.setAcceptDrops(True)

    def setup_styles(self):
        # 设置Fusion风格并自定义
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D2D;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #404040;
                color: #FFFFFF;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 8px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
            QGroupBox {
                border: 1px solid #606060;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
                color: #AAAAAA;
            }
            QCheckBox {
                color: #FFFFFF;
            }
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 4px;
                background-color: #202020;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
            QPushButton#restart_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #FF5722, stop:1 #E64A19);
            }
            QPushButton#restart_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #FF7043, stop:1 #F4511E);
            }
        """)
        # self.restart_btn.setObjectName("restart_btn")

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.path_edit.setText(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(path):
            self.path_edit.setText(path)

    def start_processing(self):
        # 参数收集
        params = {
            "folder_path": self.path_edit.text(),
            "task_type": "classification" if self.task_combo.currentText() == "分类任务" else "segmentation",
            "do_statistics": self.stats_check.isChecked(),
            "convert_format": None if self.convert_combo.currentIndex() == 0 else 
                            "png_to_bmp" if self.convert_combo.currentIndex() == 1 else "bmp_to_png",
            "do_rename": self.rename_check.isChecked(),
            "rename_rule": self.rename_edit.text() if self.rename_edit.text() else "{category}_{index}",
            'do_visualization': self.visual_check.isChecked(), 
            'do_crop': self.SegCrop_check.isChecked(),
            
        }

        # 验证路径
        if not os.path.exists(params["folder_path"]):
            QMessageBox.warning(self, "路径错误", "请选择有效的数据集路径！")
            return

        # 禁用按钮
        self.run_btn.setEnabled(False)
        self.log_area.clear()

        # 创建工作线程
        self.worker = OrganizeWorker(params)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def setup_restart(self):
        """初始化重启相关状态"""
        self.restart_requested = False

    def restart_application(self):
        """安全重启应用程序"""
        # 检查是否有正在运行的任务
        if hasattr(self, 'worker') and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '任务运行中',
                '当前有任务正在执行，确定要强制重启吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
                
            # 终止工作线程
            self.worker.terminate()
            self.worker.wait(2000)  # 最多等待2秒
        
        # 执行重启
        self.restart_requested = True
        python = sys.executable
        os.execl(python, python, *sys.argv)
    def update_log(self, message):
        self.log_area.append(f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {message}")
        if message == "分割任务":
            description = f"你选择了分割任务，分割任务的文件夹结构通常如下：\n"
            description += " - 文件目录：包含所有与分割任务相关的数据。\n"
            description += "   - other：该文件夹存放其他文件。\n"
            description += "   - train：该文件夹存放用于训练的相关数据。\n"
            description += "     - JPEGImages：该文件夹存放训练用的原始图像数据。\n"
            description += "     - annotations：该文件夹存放与JPEGImages中训练图像对应的分割掩码。\n"
            description += "     - json：该文件夹存放与训练图像相关的 JSON 格式标注信息。\n"
            description += "   - val：该文件夹存放用于验证的相关数据。\n"
            description += "     - JPEGImages：该文件夹存放验证用的原始图像数据。\n"
            description += "     - annotations：该文件夹存放与JPEGImages中验证图像对应的分割掩码。\n"
            description += "     - json：该文件夹存放与验证图像相关的 JSON 格式标注信息。\n"
            self.log_area.append(f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {description}")
        elif message == "分类任务":
            description = f"你选择了分类任务，分类任务的文件夹结构通常如下：\n"
            description += " - 文件目录：包含所有与分类任务相关的数据。\n"
            description += "   - train：该文件夹存放用于训练模型的图像数据。\n"
            description += "   - val：该文件夹存放用于验证模型性能的图像数据。\n"
            self.log_area.append(f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {description}")

    def on_finished(self):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", "数据处理完成！")

    def on_error(self, message):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误：\n{message}")
       
