# QRコード回転速度測定システム

QRコードの回転を検出してタイヤの速度を計算し、タイヤの上下移動を追跡するシステムです。

## 機能

- **QRコード回転検出**: QRコードの角度変化から回転速度（RPM）を計算
- **速度計算**: タイヤの直径と回転速度から実速度（km/h）を算出
- **タイヤ追跡**: タイヤの中心点を検出し、上下移動量（mm）を測定
- **仮想カメラ対応**: OBS Virtual Cameraなどの仮想カメラに対応
- **AVGシステム**: 直近200フレームの値を保存し平均値を返す

## 必要な環境

- Python 3.8以上
- Webカメラまたは仮想カメラ（OBS Virtual Camera等）

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

アプリケーションを起動すると、自動的にWeb APIサーバーが `http://localhost:3001` で起動します。

### 設定

`config.json`ファイルで以下の設定が可能です：

- `tire_diameter_mm`: タイヤの直径（mm）
- `camera_index`: 使用するカメラのインデックス
- `pixel_to_mm_ratio`: ピクセルからmmへの変換比率（キャリブレーション）

### 操作方法

- `q`: 終了
- `c`: キャリブレーションモード（ピクセル-mm変換比率を設定）
- `r`: 測定値リセット

## 出力情報

- 回転速度（RPM）
- 速度（km/h）
- タイヤ中心点の上下移動量（mm）
- リアルタイム映像表示

## 技術詳細

- QRコード検出: pyzbar
- 画像処理: OpenCV
- タイヤ検出: Hough円変換
- 回転角度計算: QRコードの4隅の座標から角度を算出
- Web API: Flask

## Web API

アプリケーション起動時に自動的に `http://localhost:3001` でAPIサーバーが起動します。

### エンドポイント

#### 全ステータス取得
```
GET http://localhost:3001/api/status
```
レスポンス:
```json
{
  "speed_kmh": 0.1,
  "vertical_displacement_mm": -58.19,
  "rpm": 6.28,
  "qr_detected": true,
  "timestamp": 1700000000.0
}
```

#### 速度のみ取得
```
GET http://localhost:3001/api/speed
```
レスポンス:
```json
{
  "speed_kmh": 0.1,
  "timestamp": 1700000000.0
}
```

#### 垂直移動量のみ取得（+/-付き）
```
GET http://localhost:3001/api/displacement
```
レスポンス:
```json
{
  "displacement_mm": -58.19,
  "displacement_string": "-58.19",
  "timestamp": 1700000000.0
}
```

#### ヘルスチェック
```
GET http://localhost:3001/api/health
```
レスポンス:
```json
{
  "status": "ok",
  "server": "QR Speed Measurement API",
  "version": "1.0.0"
}
```
