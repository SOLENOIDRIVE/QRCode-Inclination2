"""
Web APIサーバー - 速度と垂直移動量を提供
"""
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
import math
from collections import deque
import statistics

app = Flask(__name__)
CORS(app)  # CORS を有効化

# バッファ設定
BUFFER_SIZE = 30  # 過去30フレーム分を保持（より安定）
OUTLIER_THRESHOLD = 4.0  # 標準偏差の4倍を超えたら外れ値（より厳しく）

# グローバル変数でデータを共有
api_data = {
    'speed_ms': 0.0,
    'scale_speed_ms': 0.0,
    'vertical_displacement_mm': 0.0,
    'rpm': 0.0,
    'raw_rpm': 0.0,  # 生のRPM値
    'smoothed_rpm': 0.0,  # 平滑化されたRPM
    'qr_detected': False,
    'is_outlier': False,  # 外れ値フラグ
    'timestamp': time.time()
}

# データロック
data_lock = threading.Lock()

# RPM履歴バッファ
rpm_buffer = deque(maxlen=BUFFER_SIZE)
speed_buffer = deque(maxlen=BUFFER_SIZE)

# デバッグログバッファ（100行）
debug_log = deque(maxlen=100)
debug_log_lock = threading.Lock()

# ログファイルパス
LOG_FILE_IMPACT = "impact.log"  # IMPACT専用
LOG_FILE_NORM = "norm.log"      # 通常値専用
LOG_FILE_ALL = "all.log"        # すべてのログ


def write_to_log_file(filepath: str, log_entry: str, max_lines: int = 100):
    """ログファイルに書き込み（max_lines行まで）"""
    try:
        existing_lines = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        except FileNotFoundError:
            pass
        
        existing_lines.append(log_entry + '\n')
        
        if len(existing_lines) > max_lines:
            existing_lines = existing_lines[-max_lines:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(existing_lines)
    except Exception as e:
        print(f"Log file write error ({filepath}): {e}")


def add_debug_log(message: str, is_impact: bool = True):
    """デバッグログに追加（メモリとファイル両方）"""
    with debug_log_lock:
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        debug_log.append(log_entry)
        
        # all.logに常に書き込み
        write_to_log_file(LOG_FILE_ALL, log_entry)
        
        # IMPACT/NORMで分けて書き込み
        if is_impact:
            write_to_log_file(LOG_FILE_IMPACT, log_entry)
        else:
            write_to_log_file(LOG_FILE_NORM, log_entry, max_lines=1000)


def get_debug_log() -> list:
    """デバッグログを取得"""
    with debug_log_lock:
        return list(debug_log)


def safe_float(value, default=0.0):
    """NaN/Noneを安全な値に変換"""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return default
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default


def detect_outlier_and_smooth(value: float, buffer: deque, name: str = "value") -> tuple:
    """
    外れ値検出と平滑化を行う
    
    Args:
        value: 新しい値
        buffer: 履歴バッファ
        name: 値の名前（ログ用）
        
    Returns:
        (smoothed_value, is_outlier, raw_value, debug_info)
    """
    if len(buffer) < 5:
        # データ不足時はそのまま返す（5フレーム以上必要）
        return value, False, value, f"{name}: データ不足({len(buffer)}/5)"
    
    # 統計値を計算
    buffer_list = list(buffer)
    mean = statistics.mean(buffer_list)
    
    try:
        stdev = statistics.stdev(buffer_list)
    except statistics.StatisticsError:
        stdev = 0.0
    
    deviation = abs(value - mean)
    
    # 外れ値判定（平均から標準偏差の3倍以上離れているか）
    # さらに、絶対的な変化量も考慮（meanの10%以上の変化）
    if stdev > 0.01:
        z_score = deviation / stdev
        is_outlier = z_score > OUTLIER_THRESHOLD and deviation > abs(mean * 0.15)
        debug_info = f"{name}: val={value:.2f} mean={mean:.2f} std={stdev:.2f} z={z_score:.2f} out={is_outlier}"
    else:
        # 標準偏差が非常に小さい場合（安定している）
        # meanの15%以上変化した場合のみ外れ値
        is_outlier = deviation > abs(mean * 0.15) if mean != 0 else abs(value) > 5.0
        debug_info = f"{name}: val={value:.2f} mean={mean:.2f} std≈0 dev={deviation:.2f} out={is_outlier}"
    
    if is_outlier:
        # 外れ値の場合：実際の値をそのまま返す（衝撃などの実データ）
        return value, True, value, debug_info
    else:
        # 通常値の場合：移動平均で平滑化
        if len(buffer) >= BUFFER_SIZE:
            alpha = 0.2  # 小さい値でより滑らかに
            smoothed = alpha * value + (1 - alpha) * mean
        else:
            smoothed = (sum(buffer_list) + value) / (len(buffer_list) + 1)
        
        return smoothed, False, value, debug_info


def update_data(speed_ms: float, vertical_displacement_mm: float, rpm: float, qr_detected: bool):
    """メインプログラムからデータを更新（スマートバッファリング付き）"""
    global api_data, rpm_buffer, speed_buffer
    
    try:
        current_time = time.time()
        
        # 安全な値に変換
        safe_speed = safe_float(speed_ms)
        safe_disp = safe_float(vertical_displacement_mm)
        safe_rpm = safe_float(rpm)
        safe_detected = bool(qr_detected)
        
        # RPMの外れ値検出と平滑化
        smoothed_rpm, rpm_is_outlier, raw_rpm, rpm_debug = detect_outlier_and_smooth(safe_rpm, rpm_buffer, "RPM")
        
        # 速度の外れ値検出と平滑化
        smoothed_speed, speed_is_outlier, raw_speed, speed_debug = detect_outlier_and_smooth(safe_speed, speed_buffer, "Speed")
        
        # バッファに追加
        rpm_buffer.append(safe_rpm)
        speed_buffer.append(safe_speed)
        
        # 外れ値判定（RPMが外れ値の場合のみ - 速度は除外）
        is_outlier = rpm_is_outlier
        
        # ログに追加（IMPACT/NORMを区別）
        if is_outlier:
            add_debug_log(f"IMPACT! {rpm_debug}", is_impact=True)
        else:
            add_debug_log(f"NORM: {rpm_debug}", is_impact=False)
        
        # スケール速度
        safe_scale_speed = smoothed_speed * 8.0
        
        with data_lock:
            api_data['speed_ms'] = smoothed_speed
            api_data['scale_speed_ms'] = safe_scale_speed
            api_data['vertical_displacement_mm'] = safe_disp
            api_data['rpm'] = smoothed_rpm
            api_data['raw_rpm'] = raw_rpm
            api_data['smoothed_rpm'] = smoothed_rpm
            api_data['qr_detected'] = safe_detected
            api_data['is_outlier'] = is_outlier
            api_data['timestamp'] = current_time
            
    except Exception as e:
        add_debug_log(f"ERROR: {e}")
        print(f"API Update Error: {e}")


@app.route('/api/status', methods=['GET'])
def get_status():
    """速度と垂直移動量を返すAPI"""
    try:
        with data_lock:
            return jsonify({
                'speed_ms': round(api_data['speed_ms'], 3),
                'scale_speed_ms': round(api_data['scale_speed_ms'], 3),
                'vertical_displacement_mm': round(api_data['vertical_displacement_mm'], 2),
                'rpm': round(api_data['rpm'], 2),
                'raw_rpm': round(api_data['raw_rpm'], 2),
                'is_outlier': api_data['is_outlier'],
                'qr_detected': api_data['qr_detected'],
                'timestamp': api_data['timestamp']
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'speed_ms': 0.0,
            'scale_speed_ms': 0.0,
            'vertical_displacement_mm': 0.0,
            'rpm': 0.0,
            'raw_rpm': 0.0,
            'is_outlier': False,
            'qr_detected': False,
            'timestamp': time.time()
        }), 500


@app.route('/api/speed', methods=['GET'])
def get_speed():
    """速度のみを返すAPI"""
    try:
        with data_lock:
            return jsonify({
                'speed_ms': round(api_data['speed_ms'], 3),
                'scale_speed_ms': round(api_data['scale_speed_ms'], 3),
                'timestamp': api_data['timestamp']
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'speed_ms': 0.0,
            'scale_speed_ms': 0.0,
            'timestamp': time.time()
        }), 500


@app.route('/api/displacement', methods=['GET'])
def get_displacement():
    """垂直移動量のみを返すAPI（+/-付き）"""
    try:
        with data_lock:
            displacement = api_data['vertical_displacement_mm']
            return jsonify({
                'displacement_mm': round(displacement, 2),
                'displacement_string': f"{'+' if displacement >= 0 else ''}{displacement:.2f}",
                'timestamp': api_data['timestamp']
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'displacement_mm': 0.0,
            'displacement_string': '0.00',
            'timestamp': time.time()
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェックAPI"""
    return jsonify({
        'status': 'ok',
        'server': 'QR Speed Measurement API',
        'version': '1.0.0'
    })


def start_api_server(host='localhost', port=3001):
    """APIサーバーを起動"""
    print(f"\n=== API Server ===")
    print(f"Starting API server on http://{host}:{port}")
    print(f"Endpoints:")
    print(f"  - http://{host}:{port}/api/status")
    print(f"  - http://{host}:{port}/api/speed")
    print(f"  - http://{host}:{port}/api/displacement")
    print(f"  - http://{host}:{port}/api/health")
    print()
    
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    start_api_server()
