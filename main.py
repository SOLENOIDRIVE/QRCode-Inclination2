"""
QRコード回転速度測定システム - メインアプリケーション
タイヤに取り付けたQRコードの回転を検出し、速度とタイヤの上下移動を測定します
"""
import cv2
import numpy as np
import json
import sys
import os
import threading
import time
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent / 'src'))

from src.qr_rotation_detector import QRCodeRotationDetector
from src.tire_tracker import TireTracker
from src.speed_calculator import SpeedCalculator

# APIサーバーをインポート
try:
    from api_server import start_api_server, update_data, api_data, data_lock, get_debug_log
    API_AVAILABLE = True
except ImportError:
    print("警告: APIサーバーモジュールが見つかりません。API機能は無効です。")
    API_AVAILABLE = False
    api_data = None
    data_lock = None
    def get_debug_log(): return []


class QRCodeSpeedMeasurementSystem:
    """QRコード速度測定システムのメインクラス"""
    
    def __init__(self, config_path: str = 'config.json'):
        """
        システムの初期化
        
        Args:
            config_path: 設定ファイルのパス（デフォルト: 'config.json'）
        """
        # 設定ファイルを読み込む（タイヤサイズ、カメラ設定など）
        self.config = self.load_config(config_path)
        
        # 各モジュールを初期化
        # QRコード検出・回転速度計算モジュール
        self.qr_detector = QRCodeRotationDetector()
        
        # タイヤ中心点追跡・垂直移動測定モジュール
        self.tire_tracker = TireTracker(
            pixel_to_mm_ratio=self.config['pixel_to_mm_ratio']  # ピクセル→mm変換比率
        )
        
        # RPM→速度変換モジュール（タイヤ直径から速度を計算）
        self.speed_calculator = SpeedCalculator(
            tire_diameter_mm=self.config['tire_diameter_mm']  # タイヤ直径（mm）
        )
        
        # カメラ設定
        self.camera_index = self.config['camera_index']  # 使用するカメラのインデックス
        self.cap = None  # カメラキャプチャオブジェクト（後で初期化）
        
        # 状態管理フラグ
        self.is_running = False  # メインループ実行中フラグ
        self.calibration_mode = False  # キャリブレーションモード中フラグ
        self.calibration_points = []  # キャリブレーション用の選択点リスト
        self.calibration_input_mode = False  # 距離入力モード中フラグ
        self.calibration_input_text = ""  # 入力中のテキスト
        self.calibration_distance_px = 0.0  # 測定した距離（ピクセル）
        
        # タイヤ手動選択モード（現在未使用）
        self.tire_selection_mode = False  # タイヤ選択モード中フラグ
        self.tire_roi = None  # タイヤ領域 (x, y, w, h)
        self.tire_selection_points = []  # タイヤ選択用の点リスト
        
        # APIサーバー設定
        self.api_thread = None  # APIサーバースレッド
        self.api_enabled = API_AVAILABLE  # API機能の有効/無効
        
        # カメラ選択機能
        self.available_cameras = []  # 利用可能なカメラのインデックスリスト
        self.camera_menu_visible = False  # カメラ選択メニュー表示中フラグ
        
        # 速度履歴（直近平均速度の計算用）
        self.speed_history = []  # 速度の履歴リスト（m/s）
        self.max_speed_history = 60  # 履歴の最大サイズ（60フレーム分）
        
        # RPM履歴（直近平均RPMの計算用）
        self.rpm_history = []  # RPMの履歴リスト
        self.max_rpm_history = 60  # 履歴の最大サイズ（60フレーム分）
        
        # 上下移動履歴（グラフ表示用、直近5分間）
        self.vertical_movement_history = []  # (timestamp, displacement_mm) のリスト
        self.max_history_duration = 60  # 1分間（秒）
        
        # 外れ値検出状態（APIバッファリングシステムと連動）
        self.is_outlier = False  # 外れ値フラグ
        
    def load_config(self, config_path: str) -> dict:
        """
        設定ファイルを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定内容の辞書
        """
        try:
            # JSONファイルから設定を読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # ファイルが存在しない場合はデフォルト設定を返す
            print(f"警告: 設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。")
            return {
                'tire_diameter_mm': 200.0,  # タイヤ直径（mm）
                'camera_index': 0,  # カメラインデックス
                'pixel_to_mm_ratio': 1.0,  # ピクセル→mm変換比率
                'display_settings': {
                    'window_width': 1280,  # ウィンドウ幅
                    'window_height': 720  # ウィンドウ高さ
                }
            }
    
    def save_config(self, config_path: str = 'config.json'):
        """
        設定をファイルに保存
        
        Args:
            config_path: 保存先ファイルパス
        """
        # 現在の設定をJSONファイルとして保存
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print("設定を '{config_path}' に保存しました。")
    
    def update_speed_history(self, speed_ms: float):
        """
        速度履歴を更新（平均速度計算用）
        
        Args:
            speed_ms: 現在の速度（m/s）
        """
        # 速度を履歴に追加
        self.speed_history.append(speed_ms)
        
        # 履歴が最大サイズを超えたら古いデータを削除
        if len(self.speed_history) > self.max_speed_history:
            self.speed_history.pop(0)
    
    def get_average_speed(self) -> float:
        """
        直近の平均速度を計算
        
        Returns:
            平均速度（m/s）
        """
        # 履歴が空の場合は0を返す
        if not self.speed_history:
            return 0.0
        
        # 履歴の平均値を計算
        return sum(self.speed_history) / len(self.speed_history)
    
    def update_rpm_history(self, rpm: float):
        """
        RPM履歴を更新（平均RPM計算用）
        
        Args:
            rpm: 現在のRPM
        """
        # RPMを履歴に追加
        self.rpm_history.append(rpm)
        
        # 履歴が最大サイズを超えたら古いデータを削除
        if len(self.rpm_history) > self.max_rpm_history:
            self.rpm_history.pop(0)
    
    def get_average_rpm(self) -> float:
        """
        直近の平均RPMを計算
        
        Returns:
            平均RPM
        """
        # 履歴が空の場合は0を返す
        if not self.rpm_history:
            return 0.0
        
        # 履歴の平均値を計算
        return sum(self.rpm_history) / len(self.rpm_history)
    
    def update_vertical_movement_history(self, displacement_mm: float):
        """
        上下移動履歴を更新（グラフ表示用）
        
        Args:
            displacement_mm: 現在の上下移動量（mm）
        """
        current_time = time.time()
        
        # 履歴に追加
        self.vertical_movement_history.append((current_time, displacement_mm))
        
        # 5分以上古いデータを削除
        cutoff_time = current_time - self.max_history_duration
        self.vertical_movement_history = [
            (t, d) for t, d in self.vertical_movement_history if t >= cutoff_time
        ]
    
    def create_graph_panel(self, width: int, height: int) -> np.ndarray:
        """
        上下移動のグラフパネルを作成
        
        Args:
            width: パネルの幅
            height: パネルの高さ
            
        Returns:
            グラフパネルの画像
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)  # 暗いグレー背景
        
        # グラフ領域のマージン
        margin_left = 50
        margin_right = 10
        margin_top = 25
        margin_bottom = 25
        
        graph_width = width - margin_left - margin_right
        graph_height = height - margin_top - margin_bottom
        
        # グラフ背景
        cv2.rectangle(panel, 
                     (margin_left, margin_top), 
                     (width - margin_right, height - margin_bottom),
                     (35, 35, 35), -1)
        
        # タイトル
        cv2.putText(panel, "Vertical Movement (mm)", (margin_left, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # データがない場合
        if not self.vertical_movement_history:
            cv2.putText(panel, "No Data", (width // 2 - 30, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return panel
        
        # データ範囲を計算
        current_time = time.time()
        displacements = [d for _, d in self.vertical_movement_history]
        
        # Y軸の範囲を決定（中心が0、対称）
        max_displacement = max(abs(min(displacements)), abs(max(displacements)), 1.0)
        # 少し余裕を持たせる
        y_range = max_displacement * 1.2
        
        # 中心線（0）を描画
        center_y = margin_top + graph_height // 2
        cv2.line(panel, (margin_left, center_y), (width - margin_right, center_y),
                (80, 80, 80), 1)
        
        # Y軸ラベル
        cv2.putText(panel, f"+{y_range:.1f}", (5, margin_top + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(panel, "0", (5, center_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(panel, f"-{y_range:.1f}", (5, height - margin_bottom - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # X軸ラベル（時間）
        cv2.putText(panel, "-1min", (margin_left, height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(panel, "now", (width - margin_right - 20, height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # データポイントを描画
        if len(self.vertical_movement_history) >= 2:
            points = []
            for timestamp, displacement in self.vertical_movement_history:
                # X座標: 時間を0-1に正規化（最新が右）
                time_offset = current_time - timestamp
                x_ratio = 1.0 - (time_offset / self.max_history_duration)
                x = int(margin_left + x_ratio * graph_width)
                
                # Y座標: 変位を中心基準で変換（上がプラス）
                y_ratio = displacement / y_range
                y = int(center_y - y_ratio * (graph_height // 2))
                
                # 範囲内にクリップ
                x = max(margin_left, min(width - margin_right, x))
                y = max(margin_top, min(height - margin_bottom, y))
                
                points.append((x, y))
            
            # 線を描画
            for i in range(len(points) - 1):
                # グラデーション色（青→白→赤）
                displacement = self.vertical_movement_history[i][1]
                if displacement >= 0:
                    # プラス（上）: 緑色系
                    intensity = min(255, int(abs(displacement) / y_range * 255))
                    color = (0, 255, intensity)
                else:
                    # マイナス（下）: 赤色系
                    intensity = min(255, int(abs(displacement) / y_range * 255))
                    color = (intensity, 100, 255)
                
                cv2.line(panel, points[i], points[i + 1], color, 1)
        
        # 最新値を表示
        if self.vertical_movement_history:
            latest_displacement = self.vertical_movement_history[-1][1]
            latest_text = f"Current: {latest_displacement:.2f} mm"
            cv2.putText(panel, latest_text, (width - 120, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        return panel
    
    def scan_cameras(self, max_cameras=10):
        """
        利用可能なカメラをスキャン
        
        Args:
            max_cameras: スキャンする最大カメラ数
            
        Returns:
            利用可能なカメラインデックスのリスト
        """
        available_cameras = []
        print("カメラをスキャン中...")
        
        # カメラインデックス0～9を順番に試す
        for i in range(max_cameras):
            # DirectShowバックエンドでカメラを開く（Windows環境）
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            
            # カメラが正常に開けるか確認
            if cap.isOpened():
                # 1フレーム読み込んでカメラが実際に動作するか確認
                ret, frame = cap.read()
                if ret:
                    # 正常に動作するカメラをリストに追加
                    available_cameras.append(i)
                    print(f"カメラ {i} を検出")
                
                # カメラを解放
                cap.release()
        
        # カメラが1つも見つからない場合は警告
        if not available_cameras:
            print("警告: 利用可能なカメラが見つかりませんでした。")
        
        return available_cameras
    
    def switch_camera(self, camera_index: int) -> bool:
        """
        カメラを切り替える
        
        Args:
            camera_index: 切り替え先のカメラインデックス
            
        Returns:
            切り替え成功ならTrue、失敗ならFalse
        """
        # 現在使用中のカメラを解放
        if self.cap is not None:
            self.cap.release()
        
        # 新しいカメラインデックスを設定
        self.camera_index = camera_index
        
        # 新しいカメラを開く
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # カメラが正常に開けたか確認
        if self.cap.isOpened():
            print(f"カメラ {camera_index} に切り替えました。")
            
            # 設定ファイルを更新して保存
            self.config['camera_index'] = camera_index
            self.save_config()
            return True
        else:
            print(f"エラー: カメラ {camera_index} を開けません。")
            return False
    
    def initialize_camera(self) -> bool:
        """
        カメラを初期化
        
        Returns:
            初期化成功ならTrue、失敗ならFalse
        """
        # スキャン済みのカメラリストがある場合
        if hasattr(self, 'available_cameras') and self.available_cameras:
            # 設定ファイルで指定されたカメラが利用可能か確認
            if self.camera_index in self.available_cameras:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    print(f"カメラ {self.camera_index} を使用します。")
                    return True
            
            # 指定カメラが使えない場合、最初に見つかったカメラを使用
            for cam_idx in self.available_cameras:
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    print(f"カメラ {cam_idx} を使用します。")
                    self.camera_index = cam_idx
                    self.config['camera_index'] = cam_idx
                    self.save_config()
                    return True
        else:
            # スキャン前の初期化（フォールバック方式）
            # DirectShow バックエンドを使用（Windows環境での仮想カメラ対応）
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            # 指定カメラが開けない場合
            if not self.cap.isOpened():
                print(f"エラー: カメラ {self.camera_index} を開けません。")
                print("他のカメラインデックスを試します...")
                
                # インデックス0～4を順番に試す
                for i in range(5):
                    self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        print(f"カメラ {i} を使用します。")
                        self.camera_index = i
                        return True
                
                # どのカメラも開けなかった
                return False
            
            # 指定カメラが正常に開けた
            return True
        
        # カメラが1つも見つからなかった
        print("エラー: 利用可能なカメラが見つかりません。")
        return False
    
    def tire_selection_mouse_callback(self, event, x, y, flags, param):
        """タイヤ選択用マウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tire_selection_points.append((x, y))
            print(f"点 {len(self.tire_selection_points)}: ({x}, {y})")
            
            if len(self.tire_selection_points) == 2:
                # 矩形ROIを計算
                p1, p2 = self.tire_selection_points
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                w, h = x2 - x1, y2 - y1
                
                self.tire_roi = (x1, y1, w, h)
                print(f"タイヤ領域を設定しました: x={x1}, y={y1}, w={w}, h={h}")
                
                # タイヤ選択モードを終了
                self.tire_selection_mode = False
                self.tire_selection_points.clear()
                cv2.setMouseCallback('QR Speed Measurement', lambda *args: None)
    
    def calibration_mouse_callback(self, event, x, y, flags, param):
        """キャリブレーション用マウスコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibration_input_mode:
            self.calibration_points.append((x, y))
            print(f"点 {len(self.calibration_points)}: ({x}, {y})")
            
            if len(self.calibration_points) == 2:
                # 2点間の距離を計算
                p1, p2 = self.calibration_points
                self.calibration_distance_px = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
                
                print(f"\n測定された距離: {self.calibration_distance_px:.2f} ピクセル")
                print("画面上で実際の距離（mm）を入力してください。")
                
                # 入力モードに移行
                self.calibration_input_mode = True
                self.calibration_input_text = ""
    
    def draw_vertical_movement(self, frame, center):
        """
        QRコードの中心点の垂直移動を描画
        
        Args:
            frame: 描画対象のフレーム
            center: QRコードの中心座標 (x, y)
        """
        # 初期位置が設定されている場合のみ描画
        if self.tire_tracker.initial_center is not None:
            # 基準線（初期位置）を黄色で描画
            cv2.line(frame, 
                    (center[0] - 50, self.tire_tracker.initial_center[1]),  # 左端
                    (center[0] + 50, self.tire_tracker.initial_center[1]),  # 右端
                    (0, 255, 255), 2)  # 黄色、太さ2
            
            # 現在位置の線をシアン色で描画
            cv2.line(frame,
                    (center[0] - 50, center[1]),  # 左端
                    (center[0] + 50, center[1]),  # 右端
                    (255, 255, 0), 2)  # シアン、太さ2
            
            # 基準線から現在位置への変位矢印を緑色で描画
            cv2.arrowedLine(frame,
                          (center[0] + 60, self.tire_tracker.initial_center[1]),  # 始点（基準線）
                          (center[0] + 60, center[1]),  # 終点（現在位置）
                          (0, 255, 0), 3, tipLength=0.3)  # 緑色、太さ3
            
            # 変位量をテキスト表示（矢印の横）
            displacement_text = f"{self.tire_tracker.vertical_displacement_mm:.2f}mm"
            text_y = (self.tire_tracker.initial_center[1] + center[1]) // 2  # 中間位置
            cv2.putText(frame, displacement_text, (center[0] + 70, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def create_info_panel(self, width: int, height: int, rpm: float, speed_ms: float, qr_detected: bool):
        """情報パネルを作成"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # 濃いグレー背景
        
        y_offset = 30
        line_height = 35
        
        # タイトル
        cv2.putText(panel, "QR Speed Measurement", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # QR検出ステータス
        qr_status = "QR: OK" if qr_detected else "QR: --"
        qr_color = (0, 255, 0) if qr_detected else (100, 100, 100)
        cv2.putText(panel, qr_status, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, qr_color, 2)
        y_offset += line_height
        
        # 外れ値インジケーターランプ
        outlier_x = width - 80
        outlier_y = y_offset - 20
        if self.is_outlier:
            # 外れ値検出時：赤く点滅風のランプ
            cv2.circle(panel, (outlier_x, outlier_y), 12, (0, 0, 255), -1)  # 赤丸
            cv2.circle(panel, (outlier_x, outlier_y), 12, (255, 255, 255), 2)  # 白枠
            cv2.putText(panel, "IMPACT", (outlier_x - 35, outlier_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            # 通常時：緑のランプ
            cv2.circle(panel, (outlier_x, outlier_y), 12, (0, 150, 0), -1)  # 暗緑丸
            cv2.circle(panel, (outlier_x, outlier_y), 12, (100, 100, 100), 2)  # グレー枠
            cv2.putText(panel, "STABLE", (outlier_x - 35, outlier_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
        
        # RPM
        rpm_text = f"RPM: {rpm:.2f}"
        cv2.putText(panel, rpm_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += line_height + 5
        
        # 平均RPM
        avg_rpm = self.get_average_rpm()
        avg_rpm_text = f"Avg RPM: {avg_rpm:.2f}"
        cv2.putText(panel, avg_rpm_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height
        
        # 速度（m/s、大きく表示）
        speed_text = f"{speed_ms:.2f}"
        cv2.putText(panel, speed_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(panel, "m/s", (120, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_offset += line_height + 20
        
        # スケールスピード（8倍）
        scale_speed = speed_ms * 8.0
        scale_text = f"Scale x8: {scale_speed:.2f} m/s"
        cv2.putText(panel, scale_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        y_offset += line_height + 10
        
        # 直近平均速度
        avg_speed = self.get_average_speed()
        avg_text = f"Avg: {avg_speed:.2f} m/s"
        cv2.putText(panel, avg_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        y_offset += line_height + 10
        
        # 区切り線
        cv2.line(panel, (10, y_offset), (width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        # タイヤ情報
        tire_info = self.speed_calculator.get_tire_info()
        tire_text = f"Tire: {tire_info['diameter_mm']:.0f} mm"
        cv2.putText(panel, tire_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height - 5
        
        # 上下移動
        vertical_text = f"Vert: {self.tire_tracker.vertical_displacement_mm:.2f} mm"
        cv2.putText(panel, vertical_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height - 5
        
        # キャリブレーション
        calib_text = f"Scale: {self.tire_tracker.pixel_to_mm_ratio:.4f}"
        cv2.putText(panel, calib_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 40
        
        # 操作方法（下部）
        cv2.line(panel, (10, y_offset), (width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        cv2.putText(panel, "Controls:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        controls = [
            "Q: Quit",
            "R: Reset",
            "C: Calibrate",
            "M: Camera Menu"
        ]
        
        for control in controls:
            cv2.putText(panel, control, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y_offset += 22
        
        return panel
    
    def draw_ui(self, frame, rpm: float, speed_ms: float, qr_detected: bool):
        """UI情報を描画 - カメラ映像に枠を付けて情報パネルを配置"""
        cam_height, cam_width = frame.shape[:2]
        
        # 情報パネルのサイズ
        panel_width = 250
        panel_height = cam_height
        
        # ログパネルの高さ
        log_panel_height = 250
        
        # 最終的な表示サイズ
        panel_spacing = 10  # パネル間のスペース
        total_width = cam_width + panel_spacing + panel_width + 20  # 左右マージン追加
        total_height = cam_height + 100 + log_panel_height  # 上下にマージン + ログパネル
        
        # 全体の背景を作成
        display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        display[:] = (30, 30, 30)  # 濃いグレー背景
        
        # タイトルバー
        cv2.rectangle(display, (0, 0), (total_width, 50), (50, 50, 50), -1)
        cv2.putText(display, "QR Code Speed Measurement System", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # カメラ映像を配置（枠付き）
        cam_x = 10
        cam_y = 60
        display[cam_y:cam_y+cam_height, cam_x:cam_x+cam_width] = frame
        
        # カメラ映像の枠
        cv2.rectangle(display, (cam_x-2, cam_y-2), 
                     (cam_x+cam_width+2, cam_y+cam_height+2), 
                     (100, 100, 100), 2)
        
        # 情報パネルを作成して配置
        info_panel = self.create_info_panel(panel_width, panel_height, rpm, speed_ms, qr_detected)
        panel_x = cam_x + cam_width + panel_spacing
        display[cam_y:cam_y+panel_height, panel_x:panel_x+panel_width] = info_panel
        
        # ステータスバー（下部）
        status_y = total_height - 35
        cv2.rectangle(display, (0, status_y), (total_width, total_height), (50, 50, 50), -1)
        
        status_text = "Ready - Tracking QR Code Center"
        if self.calibration_mode:
            status_text = "Calibration Mode"
        
        cv2.putText(display, status_text, (20, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # キャリブレーションモードの表示オーバーレイ
        if self.calibration_mode and frame is not None:
            if self.calibration_input_mode:
                # 入力モード - カメラ映像エリアにオーバーレイ
                overlay_x = cam_x + cam_width // 2 - 250
                overlay_y = cam_y + cam_height // 2 - 80
                cv2.rectangle(display, (overlay_x, overlay_y), 
                            (overlay_x + 500, overlay_y + 160), (0, 0, 0), -1)
                cv2.rectangle(display, (overlay_x, overlay_y), 
                            (overlay_x + 500, overlay_y + 160), (0, 255, 255), 2)
                
                # 入力プロンプト
                prompt_text = f"Distance: {self.calibration_distance_px:.2f} px"
                cv2.putText(display, prompt_text, (overlay_x + 20, overlay_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                input_prompt = "Enter actual distance (mm):"
                cv2.putText(display, input_prompt, (overlay_x + 20, overlay_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # 入力テキスト
                input_text = self.calibration_input_text + "_"
                cv2.putText(display, input_text, (overlay_x + 20, overlay_y + 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # ヘルプ
                help_msg = "Press ENTER to confirm, ESC to cancel"
                cv2.putText(display, help_msg, (overlay_x + 20, overlay_y + 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                # 点選択モード - カメラ映像エリアに表示
                calib_info = "CALIBRATION: Click 2 points"
                cv2.putText(display, calib_info, (cam_x + 20, cam_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                step_info = f"Points: {len(self.calibration_points)}/2"
                cv2.putText(display, step_info, (cam_x + 20, cam_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 選択した点を描画（カメラ座標からディスプレイ座標に変換）
                for i, point in enumerate(self.calibration_points):
                    disp_point = (point[0] + cam_x, point[1] + cam_y)
                    cv2.circle(display, disp_point, 8, (0, 255, 0), -1)
                    cv2.circle(display, disp_point, 12, (255, 255, 255), 2)
                    cv2.putText(display, f"P{i+1}", (disp_point[0] + 15, disp_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 2点間に線を描画
                if len(self.calibration_points) == 2:
                    p1 = (self.calibration_points[0][0] + cam_x, self.calibration_points[0][1] + cam_y)
                    p2 = (self.calibration_points[1][0] + cam_x, self.calibration_points[1][1] + cam_y)
                    cv2.line(display, p1, p2, (0, 255, 0), 2)
        
        # カメラ選択メニューの表示
        if self.camera_menu_visible and len(self.available_cameras) > 0:
            menu_width = 300
            menu_height = min(50 + len(self.available_cameras) * 40, 400)
            menu_x = cam_x + (cam_width - menu_width) // 2
            menu_y = cam_y + 50
            
            # メニュー背景
            cv2.rectangle(display, (menu_x, menu_y), 
                         (menu_x + menu_width, menu_y + menu_height), 
                         (40, 40, 40), -1)
            cv2.rectangle(display, (menu_x, menu_y), 
                         (menu_x + menu_width, menu_y + menu_height), 
                         (100, 100, 100), 2)
            
            # タイトル
            cv2.putText(display, "Select Camera", (menu_x + 20, menu_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # カメラリスト
            for idx, cam_idx in enumerate(self.available_cameras):
                item_y = menu_y + 60 + idx * 40
                is_current = (cam_idx == self.camera_index)
                
                # 選択中のカメラをハイライト
                if is_current:
                    cv2.rectangle(display, (menu_x + 10, item_y - 25),
                                 (menu_x + menu_width - 10, item_y + 5),
                                 (0, 100, 200), -1)
                
                # カメラ名
                cam_text = f"{idx + 1}. Camera {cam_idx}"
                if is_current:
                    cam_text += " (Current)"
                
                color = (255, 255, 255) if is_current else (200, 200, 200)
                cv2.putText(display, cam_text, (menu_x + 20, item_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # ヘルプ
            help_y = menu_y + menu_height - 20
            cv2.putText(display, "Press 1-9 to select, M to close", 
                       (menu_x + 20, help_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # ログパネルを描画（画面下部左半分）
        log_y = cam_y + cam_height + 10
        log_x = 10
        log_width = (total_width - 30) // 2  # 半分の幅に変更
        
        # ログパネル背景
        cv2.rectangle(display, (log_x, log_y), (log_x + log_width, log_y + log_panel_height - 10),
                     (25, 25, 25), -1)
        cv2.rectangle(display, (log_x, log_y), (log_x + log_width, log_y + log_panel_height - 10),
                     (60, 60, 60), 1)
        
        # ログタイトル
        cv2.putText(display, "Debug Log (IMPACT Events):", (log_x + 10, log_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # ログ内容を表示（最新10行）
        debug_logs = get_debug_log()
        log_lines = debug_logs[-10:] if len(debug_logs) > 10 else debug_logs
        log_text_y = log_y + 35
        for log_line in log_lines:
            # IMPACT行は赤、その他は白
            if "IMPACT" in log_line:
                color = (100, 100, 255)  # 赤（BGR）
            elif "ERROR" in log_line:
                color = (0, 100, 255)  # オレンジ
            else:
                color = (150, 150, 150)  # グレー
            
            # 長いログは切り詰め（幅が半分になったので文字数も調整）
            display_line = log_line[:40] + "..." if len(log_line) > 40 else log_line
            cv2.putText(display, display_line, (log_x + 10, log_text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            log_text_y += 12
        
        # グラフパネルを描画（画面下部右半分）
        graph_x = log_x + log_width + 10
        graph_width = (total_width - 30) // 2
        graph_panel = self.create_graph_panel(graph_width, log_panel_height - 10)
        
        # グラフパネルを配置
        display[log_y:log_y + log_panel_height - 10, graph_x:graph_x + graph_width] = graph_panel
        
        # グラフパネル枠
        cv2.rectangle(display, (graph_x, log_y), (graph_x + graph_width, log_y + log_panel_height - 10),
                     (60, 60, 60), 1)
        
        return display
    
    def run(self):
        """メインループを実行"""
        # APIサーバーを起動
        if self.api_enabled:
            self.api_thread = threading.Thread(target=start_api_server, args=('localhost', 3001), daemon=True)
            self.api_thread.start()
            print("APIサーバーをバックグラウンドで起動しました。")
        
        # 利用可能なカメラをスキャン（カメラ初期化前に実行）
        self.available_cameras = self.scan_cameras()
        
        # カメラを初期化
        if not self.initialize_camera():
            print("カメラの初期化に失敗しました。")
            return
        
        print("QRコード速度測定システムを起動しました。")
        print("操作方法:")
        print("  Q: 終了")
        print("  R: 測定値をリセット")
        print("  C: キャリブレーションモード（ピクセル-mm変換比率を設定）")
        print("  M: カメラ選択メニュー")
        print()
        print("※QRコードの赤い中心点の上下移動を測定します")
        
        self.is_running = True
        cv2.namedWindow('QR Speed Measurement')
        
        try:
            while self.is_running:
                # フレームを取得
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました。")
                    break
                
                # フレームのコピーを作成
                display_frame = frame.copy()
                
                # QRコードを検出
                qr_result = self.qr_detector.detect_qr_code(frame)
                qr_detected = False
                qr_center = None
                if qr_result is not None:
                    angle, rpm, qr_points = qr_result
                    qr_detected = True
                    
                    # QRコードの中心点を計算
                    qr_center = qr_points.mean(axis=0).astype(int)
                    qr_center = tuple(qr_center)
                    
                    # QRコードの中心点で垂直移動を追跡
                    self.tire_tracker.track_vertical_movement(qr_center)
                    
                    # 上下移動履歴を更新（グラフ表示用）
                    self.update_vertical_movement_history(self.tire_tracker.vertical_displacement_mm)
                    
                    # QR情報を描画
                    display_frame = self.qr_detector.draw_qr_info(
                        display_frame, angle, rpm, qr_points
                    )
                    
                    # 垂直移動情報を描画（QRコードの中心点ベース）
                    self.draw_vertical_movement(display_frame, qr_center)
                    
                    # 速度を計算（m/s）
                    speed_ms = self.speed_calculator.rpm_to_ms(rpm)
                else:
                    rpm = 0.0
                    speed_ms = 0.0
                
                # RPMと速度履歴を更新（QR検出時のみ）
                if qr_detected and rpm > 0:
                    self.update_rpm_history(rpm)
                    self.update_speed_history(speed_ms)
                
                # 平均値を取得（履歴が空の場合は現在のrpmを使用）
                avg_rpm = self.get_average_rpm() if self.rpm_history else rpm
                avg_speed_ms = self.get_average_speed() if self.speed_history else speed_ms
                
                # APIサーバーにデータを更新
                if self.api_enabled:
                    # QR検出時は現在のRPMを、未検出時は平均RPMを使用
                    api_rpm = rpm if qr_detected else avg_rpm
                    api_speed = speed_ms if qr_detected else avg_speed_ms
                    update_data(
                        api_speed,
                        self.tire_tracker.vertical_displacement_mm,
                        api_rpm,
                        qr_detected
                    )
                    
                    # 外れ値フラグをAPIサーバーから取得してGUIに反映
                    if api_data is not None and data_lock is not None:
                        with data_lock:
                            self.is_outlier = api_data.get('is_outlier', False)
                
                # UI情報を描画（カメラ映像に枠を付けて情報パネルを配置）
                display_frame = self.draw_ui(display_frame, rpm, speed_ms, qr_detected)
                
                # フレームを表示
                cv2.imshow('QR Speed Measurement', display_frame)
                
                # キー入力を処理
                key = cv2.waitKey(1) & 0xFF
                
                # キャリブレーション入力モード
                if self.calibration_input_mode:
                    if key == 13:  # Enter
                        try:
                            known_distance = float(self.calibration_input_text)
                            
                            # キャリブレーション実行
                            self.tire_tracker.calibrate_pixel_to_mm(
                                known_distance, self.calibration_distance_px
                            )
                            self.config['pixel_to_mm_ratio'] = self.tire_tracker.pixel_to_mm_ratio
                            self.save_config()
                            
                            print(f"キャリブレーション完了: 1 pixel = {self.tire_tracker.pixel_to_mm_ratio:.4f} mm")
                            
                            # リセット
                            self.calibration_points.clear()
                            self.calibration_input_text = ""
                            self.calibration_input_mode = False
                            self.calibration_mode = False
                            cv2.setMouseCallback('QR Speed Measurement', lambda *args: None)
                        except ValueError:
                            print("エラー: 数値を入力してください。")
                            self.calibration_input_text = ""
                    
                    elif key == 27:  # ESC
                        # キャンセル
                        self.calibration_points.clear()
                        self.calibration_input_text = ""
                        self.calibration_input_mode = False
                        self.calibration_mode = False
                        cv2.setMouseCallback('QR Speed Measurement', lambda *args: None)
                        print("キャリブレーションをキャンセルしました。")
                    
                    elif key == 8:  # Backspace
                        if len(self.calibration_input_text) > 0:
                            self.calibration_input_text = self.calibration_input_text[:-1]
                    
                    elif key >= 48 and key <= 57:  # 数字 0-9
                        self.calibration_input_text += chr(key)
                    
                    elif key == 46:  # ピリオド '.'
                        if '.' not in self.calibration_input_text:
                            self.calibration_input_text += '.'
                
                elif key == ord('q') or key == ord('Q'):
                    # 終了
                    self.is_running = False
                
                elif key == ord('r') or key == ord('R'):
                    # リセット
                    self.qr_detector.reset()
                    self.tire_tracker.reset()
                    print("測定値をリセットしました。")
                
                elif key == ord('c') or key == ord('C'):
                    # キャリブレーションモード
                    if not self.calibration_mode:
                        self.calibration_mode = True
                        self.calibration_points.clear()
                        self.calibration_input_text = ""
                        self.calibration_input_mode = False
                        cv2.setMouseCallback('QR Speed Measurement', 
                                           self.calibration_mouse_callback)
                        print("\nキャリブレーションモードを開始します。")
                        print("画面上の既知の距離の2点をクリックしてください。")
                
                elif key == ord('m') or key == ord('M'):
                    # カメラメニューの表示/非表示を切り替え
                    self.camera_menu_visible = not self.camera_menu_visible
                    if self.camera_menu_visible:
                        print("\nカメラ選択メニューを表示しました。")
                        print("数字キー(1-9)でカメラを選択してください。")
                
                elif key >= ord('1') and key <= ord('9'):
                    # カメラ選択（メニュー表示中のみ）
                    if self.camera_menu_visible:
                        cam_num = key - ord('1')  # 0-8
                        if cam_num < len(self.available_cameras):
                            selected_cam = self.available_cameras[cam_num]
                            if self.switch_camera(selected_cam):
                                self.camera_menu_visible = False
        
        finally:
            # クリーンアップ
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("システムを終了しました。")


def main():
    """メイン関数"""
    # 設定ファイルのパスを取得
    config_path = 'config.json'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # システムを起動
    system = QRCodeSpeedMeasurementSystem(config_path)
    system.run()


if __name__ == '__main__':
    main()
