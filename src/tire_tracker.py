"""
タイヤ検出・追跡モジュール
円形検出でタイヤを認識し、中心点を追跡します
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class TireTracker:
    """タイヤを検出し、中心点の移動を追跡するクラス"""
    
    def __init__(self, pixel_to_mm_ratio: float = 1.0):
        """
        Args:
            pixel_to_mm_ratio: ピクセルからmmへの変換比率
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.previous_center: Optional[Tuple[int, int]] = None
        self.initial_center: Optional[Tuple[int, int]] = None
        self.vertical_displacement_px: float = 0.0
        self.vertical_displacement_mm: float = 0.0
        self.displacement_history: List[float] = []
        self.max_history_size: int = 30
        
        # Hough円検出のパラメータ
        self.dp = 1.0
        self.min_dist = 50
        self.param1 = 100
        self.param2 = 25
        self.min_radius = 20
        self.max_radius = 500
    
    def set_pixel_to_mm_ratio(self, ratio: float):
        """ピクセル-mm変換比率を設定"""
        self.pixel_to_mm_ratio = ratio
    
    def detect_tire(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Hough円変換でタイヤを検出
        
        Args:
            frame: 入力画像フレーム
            
        Returns:
            (中心x座標, 中心y座標, 半径) または None
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ強調のため、コントラストを改善
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # ノイズ除去
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        # Hough円変換
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # 最も大きな円を選択（タイヤと仮定）
            largest_circle = None
            max_radius = 0
            
            for circle in circles[0, :]:
                x, y, r = circle
                if r > max_radius:
                    max_radius = r
                    largest_circle = (int(x), int(y), int(r))
            
            return largest_circle
        
        return None
    
    def track_vertical_movement(self, current_center: Tuple[int, int]) -> float:
        """
        タイヤ中心点の垂直方向の移動を追跡
        
        Args:
            current_center: 現在の中心点座標 (x, y)
            
        Returns:
            垂直方向の変位（mm）
        """
        if self.initial_center is None:
            self.initial_center = current_center
            self.previous_center = current_center
            return 0.0
        
        # 垂直方向の変位を計算（ピクセル単位）
        # 上をプラス、下をマイナスにするため、符号を反転
        vertical_displacement_px = self.initial_center[1] - current_center[1]
        
        # mmに変換
        vertical_displacement_mm = vertical_displacement_px * self.pixel_to_mm_ratio
        
        # 履歴に追加
        self.displacement_history.append(vertical_displacement_mm)
        if len(self.displacement_history) > self.max_history_size:
            self.displacement_history.pop(0)
        
        # 移動平均で平滑化
        if len(self.displacement_history) >= 3:
            vertical_displacement_mm = np.mean(self.displacement_history[-5:])
        
        self.vertical_displacement_px = vertical_displacement_px
        self.vertical_displacement_mm = vertical_displacement_mm
        self.previous_center = current_center
        
        return vertical_displacement_mm
    
    def draw_tire_info(self, frame: np.ndarray, center: Tuple[int, int], 
                       radius: int) -> np.ndarray:
        """
        タイヤの情報をフレームに描画
        
        Args:
            frame: 入力画像フレーム
            center: タイヤの中心座標
            radius: タイヤの半径
            
        Returns:
            描画後のフレーム
        """
        # タイヤの円を描画
        cv2.circle(frame, center, radius, (255, 255, 0), 2)
        
        # 中心点を描画
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # 初期位置があれば基準線を描画
        if self.initial_center is not None:
            # 初期位置の線
            cv2.line(frame, 
                    (center[0] - 50, self.initial_center[1]),
                    (center[0] + 50, self.initial_center[1]),
                    (0, 255, 255), 2)
            
            # 現在位置の線
            cv2.line(frame,
                    (center[0] - 50, center[1]),
                    (center[0] + 50, center[1]),
                    (255, 0, 255), 2)
            
            # 変位を示す矢印
            cv2.arrowedLine(frame,
                          (center[0] + 60, self.initial_center[1]),
                          (center[0] + 60, center[1]),
                          (0, 255, 0), 2)
        
        # テキスト情報を描画
        displacement_text = f"Vertical: {self.vertical_displacement_mm:.2f} mm"
        pixel_text = f"Pixels: {self.vertical_displacement_px:.1f} px"
        
        cv2.putText(frame, displacement_text, (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, pixel_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def reset(self):
        """追跡データをリセット"""
        self.previous_center = None
        self.initial_center = None
        self.vertical_displacement_px = 0.0
        self.vertical_displacement_mm = 0.0
        self.displacement_history.clear()
    
    def calibrate_pixel_to_mm(self, known_distance_mm: float, 
                              measured_distance_px: float):
        """
        既知の距離を使ってピクセル-mm変換比率をキャリブレーション
        
        Args:
            known_distance_mm: 実際の距離（mm）
            measured_distance_px: 画像上で測定した距離（ピクセル）
        """
        if measured_distance_px > 0:
            self.pixel_to_mm_ratio = known_distance_mm / measured_distance_px
            print(f"Calibrated: 1 pixel = {self.pixel_to_mm_ratio:.4f} mm")
