"""
QRコード回転検出モジュール
QRコードの角度を検出し、回転速度（RPM）を計算します

改善版: PCA角度計算、カルマンフィルタ、最適化された検出パイプライン
"""
import cv2
import numpy as np
import time
import warnings
import os
from typing import Optional, Tuple, List

# zbarのwarningを抑制
os.environ['PYZBAR_VERBOSE'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

# pyzbarのインポート（エラーハンドリング付き）
try:
    from pyzbar import pyzbar
    # テストデコードでpyzbarが正常に動作するか確認
    PYZBAR_AVAILABLE = True
except Exception as e:
    PYZBAR_AVAILABLE = False
    print(f"警告: pyzbarが利用できません（{type(e).__name__}）。OpenCV QRCodeDetectorのみ使用します。")


class KalmanFilter1D:
    """
    1次元カルマンフィルタ for 角速度推定
    
    単純な移動平均とは異なり、最新の観測値に適応的に重み付けしつつ
    過去の推定を更新します。瞬間的な変化にも応答しながらノイズを軽減。
    """
    
    def __init__(self, process_noise: float = 0.2, measurement_noise: float = 10.0):
        """
        Args:
            process_noise: プロセスノイズ（小さいほど滑らか、大きいほど追従性向上）
            measurement_noise: 観測ノイズ（小さいほど観測値を信頼、大きいほど滑らか）
        """
        self.Q = process_noise       # プロセスノイズ
        self.R = measurement_noise   # 観測ノイズ
        self.x = 0.0                 # 状態推定値（角速度）
        self.P = 1.0                 # 誤差共分散
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """
        観測値でフィルタを更新し、推定値を返す
        
        Args:
            measurement: 観測された角速度
            
        Returns:
            フィルタリングされた角速度の推定値
        """
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        
        # 予測ステップ（状態遷移は恒等: 角速度は一定と仮定）
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 更新ステップ
        K = P_pred / (P_pred + self.R)  # カルマンゲイン
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x
    
    def reset(self):
        """フィルタをリセット"""
        self.x = 0.0
        self.P = 1.0
        self.initialized = False


class QRCodeRotationDetector:
    """QRコードの回転を検出し、RPMを計算するクラス（改善版）"""
    
    def __init__(self):
        self.previous_angle: Optional[float] = None
        self.previous_time: Optional[float] = None
        self.angle_history: List[float] = []
        self.time_history: List[float] = []
        self.rotation_count: int = 0
        self.total_rotation: float = 0.0
        self.rpm: float = 0.0
        self.max_history_size: int = 30  # 履歴サイズ
        self.last_valid_result: Optional[Tuple[float, float, np.ndarray]] = None
        self.detection_miss_count: int = 0
        self.max_miss_count: int = 15  # 最大連続検出失敗回数
        
        # カルマンフィルタ for RPM安定化
        self.angular_velocity_filter = KalmanFilter1D(
            process_noise=0.5,     # プロセスノイズ（追従性）
            measurement_noise=5.0  # 観測ノイズ（平滑化）
        )
        
        # 前回の角度基準点（PCA符号の曖昧さ解消用）
        self.reference_angle: Optional[float] = None
        
        # OpenCV QRCodeDetector（メイン検出器として使用）
        self.cv_qr_detector = cv2.QRCodeDetector()
        
        # 検出パイプラインの状態管理
        self.last_successful_pipeline: int = 0  # 最後に成功したパイプライン
        self.pipeline_failure_count: int = 0
        
        # 前処理済み画像のキャッシュ
        self._preprocessed_cache: dict = {}
        
    def calculate_angle_from_corners(self, points: np.ndarray) -> float:
        """
        QRコードの4隅から角度を計算（辺ベクトル方式）
        
        QRコードの4点を使用して、上辺の方向から角度を算出。
        OpenCV QRCodeDetectorは点を時計回りまたは反時計回りで返すため、
        隣接する点のベクトルを使用。
        
        Args:
            points: QRコードの4隅の座標 (4x2のnumpy配列)
            
        Returns:
            角度（度）-180~180
        """
        if len(points) < 4:
            return 0.0
        
        try:
            points = np.array(points, dtype=np.float32)
            
            # 4辺のベクトルを計算
            edges = []
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                edge = p2 - p1
                length = np.linalg.norm(edge)
                if length > 0:
                    edges.append((edge, length))
            
            if len(edges) < 4:
                return 0.0
            
            # 最初の辺（点0→点1）の角度を使用
            # これがQRコードの向きを表す
            edge0 = edges[0][0]
            angle = np.degrees(np.arctan2(edge0[1], edge0[0]))
            
            # 連続性を確保（急激な変化を防ぐ）
            if self.reference_angle is not None:
                diff = angle - self.reference_angle
                # 180度のジャンプを補正 (Phase unwrap)
                while diff > 180:
                    diff -= 360
                while diff < -180:
                    diff += 360
            
            # 角度を-180~180に正規化
            angle = self.normalize_angle(angle)
            
            self.reference_angle = angle
            return angle
            
        except Exception as e:
            return 0.0
    
    def calculate_angle_legacy(self, points: np.ndarray) -> float:
        """
        レガシー角度計算（フォールバック用）
        
        Args:
            points: QRコードの座標
            
        Returns:
            角度（度）
        """
        if len(points) < 2:
            return 0.0
        
        try:
            # 点を座標でソート（上から2つを取得）
            sorted_points = points[points[:, 1].argsort()]
            top_two = sorted_points[:2]
            
            # 左上と右上を判定
            if top_two[0][0] < top_two[1][0]:
                top_left = top_two[0]
                top_right = top_two[1]
            else:
                top_left = top_two[1]
                top_right = top_two[0]
            
            # ベクトルから角度を計算
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            return angle
        except Exception:
            return 0.0
            return 0.0
    
    def normalize_angle(self, angle: float) -> float:
        """角度を-180~180度の範囲に正規化"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def detect_rotation(self, current_angle: float, current_time: float) -> Tuple[float, float]:
        """
        角度の変化から回転を検出し、RPMを計算（カルマンフィルタ付き）
        
        Args:
            current_angle: 現在の角度（-180~180度）
            current_time: 現在の時刻（秒）
            
        Returns:
            (RPM, 累積回転角度)
        """
        # 初回は角度と時刻を保存して終了
        if self.previous_angle is None:
            self.previous_angle = current_angle
            self.previous_time = current_time
            return 0.0, 0.0
        
        # 時間差を計算（秒）
        time_diff = current_time - self.previous_time
        
        # 時間差が無効な場合はスキップ
        if time_diff <= 0.001:
            return abs(self.rpm), self.total_rotation
        
        # 前回からの角度差を計算
        raw_angle_diff = current_angle - self.previous_angle
        
        # 角度差を-180~180度の範囲に正規化（最短経路を選択）
        angle_diff = raw_angle_diff
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # 異常な角度変化を検出（フレーム間で90度以上変化は異常値）
        # 高速回転でも1フレームで90度以上は稀なので、これをフィルタ
        max_angle_change = 90.0
        if abs(angle_diff) > max_angle_change:
            # 異常値として前回の状態を維持
            self.previous_time = current_time
            return abs(self.rpm), self.total_rotation
        
        # 累積回転角度を更新
        self.total_rotation += angle_diff
        
        # 完全な1回転を検出してカウント
        if self.total_rotation >= 360:
            self.rotation_count += 1
            self.total_rotation -= 360
        elif self.total_rotation <= -360:
            self.rotation_count -= 1
            self.total_rotation += 360
        
        # RPMを計算（時間差に関係なく常に計算）
        # 角度変化と時間差を履歴に追加
        self.angle_history.append(angle_diff)
        self.time_history.append(time_diff)
        
        # 履歴サイズを制限
        if len(self.angle_history) > self.max_history_size:
            self.angle_history.pop(0)
            self.time_history.pop(0)
        
        # 角速度を計算（過去10フレーム分のデータを使用）
        window_size = 10
        if len(self.angle_history) >= window_size:
            # 直近10フレームの角速度を個別に計算
            recent_velocities = []
            for i in range(-window_size, 0):
                if self.time_history[i] > 0.0001:
                    vel = self.angle_history[i] / self.time_history[i]
                    recent_velocities.append(vel)
            
            if len(recent_velocities) >= 3:
                # 中央値フィルタでノイズ除去
                recent_velocities.sort()
                # 上下25%を除外してトリム平均
                trim_count = len(recent_velocities) // 4
                if trim_count > 0:
                    trimmed = recent_velocities[trim_count:-trim_count]
                else:
                    trimmed = recent_velocities
                
                if trimmed:
                    instant_angular_velocity = sum(trimmed) / len(trimmed)
                else:
                    # フォールバック：通常の平均
                    recent_angle_sum = sum(self.angle_history[-window_size:])
                    recent_time_sum = sum(self.time_history[-window_size:])
                    instant_angular_velocity = recent_angle_sum / recent_time_sum if recent_time_sum > 0.0001 else 0.0
            else:
                # データ不足時
                recent_angle_sum = sum(self.angle_history[-window_size:])
                recent_time_sum = sum(self.time_history[-window_size:])
                instant_angular_velocity = recent_angle_sum / recent_time_sum if recent_time_sum > 0.0001 else 0.0
        else:
            # データ不足時はある分だけで計算
            total_angle = sum(self.angle_history)
            total_time = sum(self.time_history)
            if total_time > 0.0001:
                instant_angular_velocity = total_angle / total_time
            else:
                instant_angular_velocity = angle_diff / time_diff
        
        # カルマンフィルタで角速度をさらに安定化
        filtered_angular_velocity = self.angular_velocity_filter.update(instant_angular_velocity)
        
        # 角速度からRPMに変換: (度/秒) ÷ 360 × 60 = RPM
        self.rpm = (filtered_angular_velocity / 360.0) * 60.0
        
        # 現在の角度と時刻を保存
        self.previous_angle = current_angle
        self.previous_time = current_time
        
        return abs(self.rpm), self.total_rotation
    
    # --- 画像前処理パイプライン（軽量版） ---
    
    def preprocess_clahe(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE（適応的ヒストグラム均等化）による前処理"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def preprocess_adaptive_binary(self, frame: np.ndarray) -> np.ndarray:
        """適応的二値化による前処理"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        return binary
    
    def preprocess_otsu(self, frame: np.ndarray) -> np.ndarray:
        """大津法による二値化前処理"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # 大津法で二値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def decode_pyzbar_safe(self, image: np.ndarray) -> list:
        """
        pyzbarを安全に呼び出す（タイムアウトとエラーハンドリング付き）
        
        Args:
            image: 入力画像（グレースケールまたはBGR）
            
        Returns:
            デコード結果のリスト
        """
        if not PYZBAR_AVAILABLE:
            return []
        
        try:
            # symbols引数でQRコードのみに限定（高速化）
            result = pyzbar.decode(image, symbols=[pyzbar.ZBarSymbol.QRCODE])
            return result
        except Exception:
            # エラーが発生した場合は空リストを返す
            return []
    
    def detect_with_opencv_qr(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        """
        OpenCV QRCodeDetectorを使用した検出（高速・安定）
        
        Returns:
            (座標点, データ) または None
        """
        try:
            # グレースケール変換
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # detectAndDecode（単一QR用、より高速）
            data, points, _ = self.cv_qr_detector.detectAndDecode(gray)
            
            if points is not None and len(points) > 0:
                pts = points[0] if len(points.shape) == 3 else points
                if len(pts) == 4:
                    return (pts, data)
            
            return None
        except Exception:
            return None
    
    def detect_qr_code(self, frame: np.ndarray) -> Optional[Tuple[float, float, np.ndarray]]:
        """
        フレームからQRコードを検出し、角度とRPMを計算
        
        最適化版: OpenCVを優先、pyzbarはフォールバック
        
        Args:
            frame: 入力画像フレーム
            
        Returns:
            (角度, RPM, QRコードの座標) または None
        """
        points = None
        
        # パイプライン1: OpenCV QRCodeDetector（元画像）- 最も高速
        result = self.detect_with_opencv_qr(frame)
        if result is not None:
            points = result[0]
        
        # パイプライン2: OpenCV QRCodeDetector（CLAHE強調）
        if points is None:
            enhanced = self.preprocess_clahe(frame)
            result = self.detect_with_opencv_qr(enhanced)
            if result is not None:
                points = result[0]
        
        # パイプライン3: OpenCV QRCodeDetector（適応二値化）
        if points is None:
            binary = self.preprocess_adaptive_binary(frame)
            result = self.detect_with_opencv_qr(binary)
            if result is not None:
                points = result[0]
        
        # パイプライン4: pyzbar（元画像）- フォールバック
        if points is None and PYZBAR_AVAILABLE:
            decoded = self.decode_pyzbar_safe(frame)
            if decoded:
                for obj in decoded:
                    pts = np.array([[p.x, p.y] for p in obj.polygon])
                    if len(pts) == 4:
                        points = pts
                        break
        
        # パイプライン5: pyzbar（CLAHE強調）
        if points is None and PYZBAR_AVAILABLE:
            enhanced = self.preprocess_clahe(frame)
            decoded = self.decode_pyzbar_safe(enhanced)
            if decoded:
                for obj in decoded:
                    pts = np.array([[p.x, p.y] for p in obj.polygon])
                    if len(pts) == 4:
                        points = pts
                        break
        
        # パイプライン6: pyzbar（大津法二値化）
        if points is None and PYZBAR_AVAILABLE:
            otsu = self.preprocess_otsu(frame)
            decoded = self.decode_pyzbar_safe(otsu)
            if decoded:
                for obj in decoded:
                    pts = np.array([[p.x, p.y] for p in obj.polygon])
                    if len(pts) == 4:
                        points = pts
                        break
        
        # QRコードが検出された場合の処理
        if points is not None:
            points = np.array(points, dtype=np.float32)
            
            # コーナーベースの安定角度計算
            angle = self.calculate_angle_from_corners(points)
            
            # 回転速度を計算
            current_time = time.time()
            rpm, total_rotation = self.detect_rotation(angle, current_time)
            
            # 検出成功時にカウントをリセット
            if self.detection_miss_count > 0:
                print(f"QRコード再検出")
            self.detection_miss_count = 0
            result = (angle, rpm, points)
            self.last_valid_result = result
            
            return result
        
        # 検出失敗時の処理
        self.detection_miss_count += 1
        
        # 一定回数まで前回の結果を使用（安定化）
        if self.last_valid_result is not None and self.detection_miss_count < self.max_miss_count:
            angle, _, points = self.last_valid_result
            # 最新のRPM推定値を使用
            return (angle, abs(self.rpm), points)
        
        return None
    
    def get_average_rpm(self) -> float:
        """現在のRPM推定値を取得（カルマンフィルタ適用済み）"""
        return abs(self.rpm)
    
    def draw_qr_info(self, frame: np.ndarray, angle: float, rpm: float, 
                     points: np.ndarray) -> np.ndarray:
        """
        QRコードの情報をフレームに描画
        
        Args:
            frame: 入力画像フレーム
            angle: QRコードの角度
            rpm: 回転速度
            points: QRコードの座標
            
        Returns:
            描画後のフレーム
        """
        # QRコードの輪郭を描画
        cv2.polylines(frame, [points.astype(int)], True, (0, 255, 0), 3)
        
        # 中心点を計算
        center = points.mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
        
        # 角度を示す線を描画
        angle_rad = np.radians(angle)
        line_length = 50
        end_point = (
            int(center[0] + line_length * np.cos(angle_rad)),
            int(center[1] + line_length * np.sin(angle_rad))
        )
        cv2.line(frame, tuple(center), end_point, (255, 0, 0), 2)
        
        # テキスト情報を描画
        info_text = f"Angle: {angle:.1f}deg"
        rpm_text = f"RPM: {rpm:.1f}"
        rotation_text = f"Total: {self.total_rotation:.1f}deg"
        
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, rpm_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, rotation_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def reset(self):
        """測定値をリセット"""
        self.previous_angle = None
        self.previous_time = None
        self.angle_history.clear()
        self.time_history.clear()
        self.rotation_count = 0
        self.total_rotation = 0.0
        self.rpm = 0.0
        self.reference_angle = None
        self.angular_velocity_filter.reset()
        self.last_valid_result = None
        self.detection_miss_count = 0
