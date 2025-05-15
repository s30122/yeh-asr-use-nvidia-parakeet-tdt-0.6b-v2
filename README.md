# NVIDIA Parakeet TDT 語音轉文字系統

這是一個使用 `nvidia/parakeet-tdt-0.6b-v2` 模型的 Python 專案，專門用於將語音轉換成文字 (Speech-to-Text)。

## 功能特點

- 支援影片和音訊檔案的語音轉寫
- 自動檢測並使用 GPU 加速（如果可用）
- 智慧記憶體管理，支援長音訊分段處理
- 支援多種輸入格式（透過 FFmpeg 轉換）

> 注意：此專案僅輸出純文字格式。如需時間軸字幕（SRT）格式，建議使用 Whisper 模型。

## 系統需求

### 硬體需求

| 元件 | 最低需求 | 建議配置 |
|------|----------|----------|
| CPU | 任何現代多核心處理器 | - |
| 記憶體 | 8GB RAM | 16GB+ RAM |
| GPU | 4GB VRAM NVIDIA GPU | 6GB+ VRAM NVIDIA GPU |

> **注意**：
> - 使用 4GB VRAM GPU 時會自動啟用分段處理功能
> - 無 GPU 時可使用 CPU 運行，但處理速度較慢

### 軟體需求

| 軟體 | 版本要求 |
|------|----------|
| 作業系統 | Windows 10/11, macOS, 或 Linux |
| Python | 3.10（建議）或 3.9-3.11 |
| CUDA | 11.7+ （僅在使用 GPU 時必要）|

## 安裝與設定

### 1. 安裝 Python
1. 前往 [Python 官網](https://www.python.org/downloads/) 下載 Python 3.10
2. 安裝時請勾選「Add Python to PATH」選項
3. 完成安裝後，開啟終端機輸入 `python --version` 確認版本

### 2. 安裝 FFmpeg
#### Windows 使用者
- 使用 Chocolatey（推薦）：
  ```powershell
  choco install ffmpeg
  ```
- 或手動安裝：
  1. 從 [FFmpeg 官方建置](https://github.com/BtbN/FFmpeg-Builds/releases) 下載
  2. 解壓縮到 `C:\Program Files\ffmpeg`
  3. 將 `C:\Program Files\ffmpeg\bin` 加入系統環境變數 PATH

#### macOS 使用者
```bash
brew install ffmpeg
```

#### Linux 使用者
```bash
sudo apt update && sudo apt install ffmpeg
```

### 3. 安裝 CUDA（僅 GPU 使用者需要）
1. 前往 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads)
2. 選擇適合您作業系統的 CUDA 11.7 或更新版本
3. 按照安裝嚮導完成安裝
4. 重啟電腦

### 4. 安裝 Python 套件
```powershell
pip install moviepy torch torchaudio nemo_toolkit[asr]
```

## 環境驗證
執行以下命令來驗證環境設置：

```powershell
# 檢查 Python 版本
python --version  # 應顯示 3.10.x（或 3.9.x-3.11.x）

# 檢查 FFmpeg
ffmpeg -version  # 應顯示 FFmpeg 版本資訊

# 檢查 CUDA（若使用 GPU）
nvidia-smi  # 應顯示 GPU 資訊和 CUDA 版本

# 檢查 Python 套件
pip list | findstr "moviepy torch torchaudio nemo_toolkit"

# 檢查 GPU 支援
python -c "import torch; print(f'GPU 可用: {torch.cuda.is_available()}')"

# 檢查 NeMo
python -c "from nemo.collections.asr.models import EncDecCTCModelBPE; print('NeMo ASR 已安裝')"
```

## 使用方法

1. Clone 專案：
```bash
git clone https://github.com/s30122/yeh-asr-use-nvidia-parakeet-tdt-0.6b-v2.git
cd yeh-asr-use-nvidia-parakeet-tdt-0.6b-v2
```

2. 執行程式：
```bash
python main.py
```

3. 選擇檔案類型：
   - 1：音訊檔案（.wav 或 .flac）
   - 2：影片檔案（自動提取音訊）

4. 輸入檔案路徑並等待處理

轉錄結果會自動儲存為「原檔名_轉寫結果.txt」

## 注意事項
- 處理長音訊時，程式會自動分段處理以避免顯存不足
- 若要處理大型檔案，建議使用具有更多 VRAM 的 GPU


