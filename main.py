import os
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModelBPE

class ModelLoadError(Exception):
    """模型載入錯誤"""
    pass

class AudioProcessError(Exception):
    """音訊處理錯誤"""
    pass

class AudioTranscriber:
    def __init__(self):
        print("正在初始化轉錄系統...")
    
        # 設定 MoviePy 使用系統安裝的 FFmpeg
        try:
            import moviepy
            ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
            if os.path.exists(ffmpeg_path):
                moviepy.config.FFMPEG_BINARY = ffmpeg_path
                print(f"已設定使用系統 FFmpeg: {ffmpeg_path}")
            else:
                print(f"警告: 找不到系統 FFmpeg 在 {ffmpeg_path}")
        except Exception as e:
            print(f"設定 FFmpeg 時發生錯誤: {str(e)}")
    
        # 設定模型保存路徑
        self.model_name = 'nvidia/parakeet-tdt-0.6b-v2'
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        
        # 檢查 GPU 記憶體和優化設定
        if torch.cuda.is_available():
            # 清空快取，釋放GPU記憶體
            torch.cuda.empty_cache()
            
            # 設定較低精度以節省記憶體
            torch.set_float32_matmul_precision('medium')
            
            # 偵測 GPU 記憶體
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"偵測到 GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU 記憶體: {gpu_mem:.2f} GB")
            
            # 根據記憶體大小決定是否使用 GPU 和批次大小
            if gpu_mem < 4.5:  # 小於 4.5GB 時分段處理
                print("GPU 記憶體有限，將採用分段處理音訊")
                self.device = "cuda"
                self.segment_audio = True
                self.max_audio_length = 20  # 秒
            else:
                self.device = "cuda"
                self.segment_audio = False
        else:
            print("未偵測到 GPU，使用 CPU 進行推論")
            self.device = "cpu"
            self.segment_audio = True
            self.max_audio_length = 60  # CPU 可處理較長片段
    
        try:
            print("正在檢查本地模型...")
            # 嘗試載入本地模型
            self.model = EncDecCTCModelBPE.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            print(f"成功載入本地模型到 {self.device}!")
        except Exception as e:
            print("未找到本地模型，正在從網路下載...")
            try:
                self.model = EncDecCTCModelBPE.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                print("模型下載完成!")
            except Exception as download_error:
                raise ModelLoadError("模型下載失敗，請確認網路連接正常後重試。") from download_error
    
        self.model.eval()
        print("模型初始化完成！")
        
    def process_video(self, video_path: str) -> str:
        """處理影片檔案，轉換成音訊後進行轉寫"""
        print("正在處理影片...")
        
        # 臨時音訊檔案路徑
        temp_audio = 'temp_audio.wav'
        
        try:
            print("正在從影片提取音訊...")
            
            # 判斷檔案類型
            import subprocess
            import json
            
            # 使用 FFprobe 檢查檔案
            ffprobe_path = r"C:\Program Files\ffmpeg\bin\ffprobe.exe"
            if not os.path.exists(ffprobe_path):
                ffprobe_path = "ffprobe"  # 使用 PATH 中的 ffprobe
                
            print("檢查檔案格式...")
            probe_cmd = [
                ffprobe_path, 
                "-v", "quiet", 
                "-print_format", "json", 
                "-show_format", 
                "-show_streams", 
                video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                file_info = json.loads(result.stdout)
                has_video = any(stream["codec_type"] == "video" for stream in file_info.get("streams", []))
                has_audio = any(stream["codec_type"] == "audio" for stream in file_info.get("streams", []))
                
                print(f"檔案包含視訊: {has_video}, 包含音訊: {has_audio}")
                
                # 如果只有音訊沒有視訊，可以直接處理
                if has_audio and not has_video:
                    print("檔案是純音訊，直接進行轉換...")
                    # 直接轉換為 WAV
                    subprocess.run([
                        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                        "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1",
                        temp_audio
                    ], check=True)
                    
                    # 使用音訊處理功能
                    result = self.transcribe_audio(temp_audio)
                    return result
            
            # 嘗試常規的 MoviePy 處理方式
            try:
                video = VideoFileClip(video_path)
                if video.audio is not None:
                    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                    video.close()
                    
                    # 進行音訊轉寫
                    print("正在進行語音轉寫...")
                    result = self.transcribe_audio(temp_audio)
                    return result
                else:
                    raise AudioProcessError("影片不包含音訊軌道")
            except Exception as e:
                print(f"使用 MoviePy 處理失敗: {str(e)}")
                print("嘗試使用直接 FFmpeg 命令...")
                
                # 使用 FFmpeg 直接提取音訊
                subprocess.run([
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    temp_audio
                ], check=True)
                
                # 使用音訊處理功能
                result = self.transcribe_audio(temp_audio)
                return result
            
        except Exception as e:
            raise AudioProcessError(f"影片處理失敗：{str(e)}") from e
        finally:
            # 清理臨時檔案
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """處理音訊檔案（支援 .wav 和 .flac）"""
        print("正在處理音訊...")
        
        try:
            # 載入音訊
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 確保音訊是單聲道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 計算音訊長度
            audio_length_seconds = waveform.shape[1] / sample_rate
            print(f"音訊長度: {audio_length_seconds:.2f} 秒")
            
            # 分段處理長音訊以避免記憶體不足
            if self.segment_audio and audio_length_seconds > self.max_audio_length:
                print(f"音訊較長，將分段處理（每段 {self.max_audio_length} 秒）")
                return self._process_audio_in_segments(audio_path)
            
            # 一次性處理較短音訊
            print("正在進行語音轉寫...")
            
            # 清空 GPU 快取釋放記憶體
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            with torch.no_grad():
                transcription = self.model.transcribe([audio_path])[0]
            
            # 確保回傳的是字串
            if hasattr(transcription, 'text'):
                # 如果是 Hypothesis 物件，取其 text 屬性
                return transcription.text
            elif hasattr(transcription, '__str__'):
                # 如果物件可以轉換為字串
                return str(transcription)
            else:
                # 預設情況
                return f"轉錄結果: {transcription}"
            
        except Exception as e:
            raise AudioProcessError(f"音訊處理失敗：{str(e)}") from e

    def _process_audio_in_segments(self, audio_path: str) -> str:
        """將長音訊分段處理，避免 GPU 記憶體不足"""
        import tempfile
        
        # 載入音訊
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 計算分段
        audio_length = waveform.shape[1]
        segment_length = int(self.max_audio_length * sample_rate)
        num_segments = (audio_length + segment_length - 1) // segment_length
        
        all_transcriptions = []
        
        print(f"將音訊分成 {num_segments} 段進行處理")
        
        for i in range(num_segments):
            print(f"處理片段 {i+1}/{num_segments}...")
            
            # 計算當前片段的開始和結束
            start = i * segment_length
            end = min((i + 1) * segment_length, audio_length)
            
            # 提取當前片段
            segment = waveform[:, start:end]
            
            # 儲存為臨時檔案
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            torchaudio.save(temp_path, segment, sample_rate)
            
            try:
                # 清空 GPU 記憶體
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
                # 轉寫當前片段
                with torch.no_grad():
                    transcription = self.model.transcribe([temp_path])[0]
                    
                # 提取文字
                if hasattr(transcription, 'text'):
                    text = transcription.text
                else:
                    text = str(transcription)
                    
                all_transcriptions.append(text)
                
            except Exception as e:
                print(f"處理片段 {i+1} 時發生錯誤: {str(e)}")
            finally:
                # 刪除臨時檔案
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # 合併所有轉寫結果
        return " ".join(all_transcriptions)

def main():
    try:
        transcriber = AudioTranscriber()
        
        print("\n=== 語音轉文字系統 ===")
        print("請選擇要處理的檔案類型：")
        print("1. 音訊檔案 (.wav 或 .flac)")
        print("2. 影片檔案")
        
        choice = input("\n請輸入選項 (1 或 2)：")
        file_path = input("請輸入檔案路徑：")
        
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            print("錯誤：找不到檔案")
            return
        
        # 檢查檔案大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 轉換為 MB
        print(f"\n檔案大小：{file_size:.2f} MB")
        
        # 處理檔案
        if choice == "1":
            if not (file_path.endswith('.wav') or file_path.endswith('.flac')):
                print("錯誤：不支援的音訊格式，請使用 .wav 或 .flac 格式")
                return
            result = transcriber.transcribe_audio(file_path)
        elif choice == "2":
            result = transcriber.process_video(file_path)
        else:
            print("錯誤：無效的選項")
            return
        
        # 不顯示轉寫結果，只顯示儲存的訊息
        print("\n轉寫已完成")
        
        # 儲存結果到檔案
        output_path = os.path.splitext(file_path)[0] + "_轉寫結果.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"結果已儲存至：{output_path}")
        
    except (ModelLoadError, AudioProcessError) as e:
        print(f"\n處理過程中發生錯誤：{str(e)}")
    except Exception as e:
        print(f"\n發生未預期的錯誤：{str(e)}")
    finally:
        print("\n程式執行完成")

if __name__ == "__main__":
    main()
