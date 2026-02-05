"""
速度計算モジュール
RPMとタイヤ直径から実速度を計算します
"""
import math
from typing import Tuple


class SpeedCalculator:
    """回転速度とタイヤサイズから速度を計算するクラス"""
    
    def __init__(self, tire_diameter_mm: float):
        """
        Args:
            tire_diameter_mm: タイヤの直径（mm）
        """
        self.tire_diameter_mm = tire_diameter_mm
        self.tire_circumference_mm = math.pi * tire_diameter_mm
        self.tire_circumference_m = self.tire_circumference_mm / 1000.0
    
    def set_tire_diameter(self, diameter_mm: float):
        """
        タイヤ直径を設定
        
        Args:
            diameter_mm: タイヤの直径（mm）
        """
        self.tire_diameter_mm = diameter_mm
        self.tire_circumference_mm = math.pi * diameter_mm
        self.tire_circumference_m = self.tire_circumference_mm / 1000.0
    
    def calculate_speed(self, rpm: float) -> Tuple[float, float, float]:
        """
        RPMから速度を計算
        
        Args:
            rpm: 回転速度（RPM: Revolutions Per Minute）
            
        Returns:
            (速度 km/h, 速度 m/s, 速度 m/min)
        """
        # 1分間の移動距離（メートル）
        distance_per_minute_m = self.tire_circumference_m * rpm
        
        # 1時間の移動距離（キロメートル）
        distance_per_hour_km = (distance_per_minute_m * 60) / 1000.0
        
        # 1秒間の移動距離（メートル）
        distance_per_second_m = distance_per_minute_m / 60.0
        
        return distance_per_hour_km, distance_per_second_m, distance_per_minute_m
    
    def rpm_to_kmh(self, rpm: float) -> float:
        """
        RPMをkm/hに変換
        
        Args:
            rpm: 回転速度（RPM）
            
        Returns:
            速度（km/h）
        """
        speed_kmh, _, _ = self.calculate_speed(rpm)
        return speed_kmh
    
    def rpm_to_ms(self, rpm: float) -> float:
        """
        RPMをm/sに変換
        
        Args:
            rpm: 回転速度（RPM）
            
        Returns:
            速度（m/s）
        """
        _, speed_ms, _ = self.calculate_speed(rpm)
        return speed_ms
    
    def get_tire_info(self) -> dict:
        """
        タイヤの情報を取得
        
        Returns:
            タイヤ情報の辞書
        """
        return {
            'diameter_mm': self.tire_diameter_mm,
            'diameter_cm': self.tire_diameter_mm / 10.0,
            'circumference_mm': self.tire_circumference_mm,
            'circumference_m': self.tire_circumference_m
        }
