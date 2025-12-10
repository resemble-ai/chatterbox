# Chatterbox TTS 服务

本文档介绍如何设置和运行 Chatterbox TTS 服务，用于生产环境的 FastAPI 服务器部署。

## 概述

Chatterbox TTS 服务提供基于 ChatterboxMultilingualTTS 模型的文本转语音合成 RESTful API。服务支持：

- **23 种语言**的多语言零样本语音克隆
- 基于参考音频的**零样本语音克隆**
- **PCM 格式 WAV 输出**（16 位有符号整数），具有最大兼容性
- 基于 **FastAPI 的 REST API**，自动生成 API 文档
- **CORS 支持**，支持跨域请求
- **线程安全的模型推理**，带有参考音频缓存

### 已知问题和限制

> **⚠️ 重要提示：** ChatterboxMultilingualTTS 在非英语语言（特别是中文）上存在已知的质量问题。
>
> **中文语言问题：**
> - 音频输出在结尾处经常出现意外的杂音，例如：
>   - 超出预期长度的喘气声
>   - 小声讲话或低音量语音
>   - 其他音频杂音
>
> **其他语言：**
> - 在 [GitHub issues](https://github.com/resemble-ai/chatterbox/issues/) 中，其他用户也反馈了各种非英语语言的类似问题
>
> **建议：**
> - **目前，仅建议在英文上使用 Chatterbox.**

### API 端点

- `GET /api/v1/list_voice_names` - 列出所有可用的语音配置
- `POST /api/v1/generate_audio` - 使用指定语音从文本生成音频
- `GET /health` - 健康检查端点
- `GET /` - 重定向到 API 文档 `/docs`

## 参考音频准备

参考音频是用于零样本语音克隆的音色样本文件。每个参考音频文件定义一种可用于合成的音色，系统会模仿该音频中的声音特征来生成语音。

### 文件命名规范

参考音频文件必须遵循以下命名模式：
```
{voice_key}_{language_id}.wav
```

其中：
- `voice_key`：音色的唯一标识符（例如："刻晴"、"keqing"）
- `language_id`：两字母语言代码（例如："zh"、"en"、"fr"）

**示例：**
```
刻晴_zh.wav
keqing_en.wav
```

### 文件要求

- **格式**：WAV 文件
- **采样率**：24000 Hz（推荐，如需要将被重采样）
- **声道**：单声道（单通道）
- **时长**：至少 6-10 秒以获得最佳效果

### 目录结构

将所有参考音频文件放在一个目录中（默认：`data/`）：

```
data/
├── 刻晴_zh.wav
├── keqing_en.wav
├── voice1_fr.wav
└── voice2_es.wav
```

服务将在启动时自动扫描此目录并注册所有有效的参考音频。

### 预设参考音频文件

本仓库为 DLP3D 项目提供了预设的参考音频文件，包括部分角色的中英文音色样本。您可以从以下地址下载：

1. **百度网盘**：[https://pan.baidu.com/s/18Syh-_uwEoN-jVSDc--zBQ?pwd=r8ev](https://pan.baidu.com/s/18Syh-_uwEoN-jVSDc--zBQ?pwd=r8ev)
   - 输入提取密码：`r8ev`
   - 下载 `voices.zip` 文件并解压到 `data/` 目录

2. **GitHub Releases**：[https://github.com/LazyBusyYang/chatterbox/releases/download/voices/voices.zip](https://github.com/LazyBusyYang/chatterbox/releases/download/voices/voices.zip)
   - 直接下载 `voices.zip` 文件
   - 将内容解压到 `data/` 目录

下载并解压后，参考音频文件即可使用。重启服务以注册它们。

### 添加新音色

要添加新音色，需要将音频文件转换为所需的 WAV 格式。如果您有 MP3 或其他格式的音频文件，请使用 `ffmpeg` 进行转换：

```bash
ffmpeg \
    -i input_audio.mp3 \
    -ar 24000 \
    -ac 1 \
    -acodec pcm_s16le \
    data/{voice_key}_{language_id}.wav
```

**参数说明：**
- `-i input_audio.mp3`：输入音频文件（可以是 MP3、WAV 或其他格式）
- `-ar 24000`：设置采样率为 24000 Hz
- `-ac 1`：转换为单声道（单通道）
- `-acodec pcm_s16le`：使用 PCM 16 位小端编码
- `data/{voice_key}_{language_id}.wav`：遵循命名规范的输出文件路径

**示例：**
```bash
# 将 MP3 转换为 WAV 用于新音色
ffmpeg \
    -i my_voice.mp3 \
    -ar 24000 \
    -ac 1 \
    -acodec pcm_s16le \
    data/myvoice_en.wav
```

将新音色文件添加到 `data/` 目录后，重启服务以注册它。

## 检查点准备

服务可以从以下两种方式加载 TTS 模型：
1. 本地检查点目录（推荐用于生产环境）
2. HuggingFace 预训练模型（自动下载）

### 选项 1：本地检查点目录

**下载选项：**

您可以从以下来源之一手动下载模型检查点文件：

1. **HuggingFace**：[https://huggingface.co/ResembleAI/chatterbox/tree/main](https://huggingface.co/ResembleAI/chatterbox/tree/main)
   - 导航到仓库并下载下面列出的所需文件

2. **百度网盘**：[https://pan.baidu.com/s/1ivYmAmZS4t1ec-edwtJ9dA?pwd=3nuq](https://pan.baidu.com/s/1ivYmAmZS4t1ec-edwtJ9dA?pwd=3nuq)
   - 输入提取密码：`3nuq`
   - 从共享文件夹下载检查点文件

下载后，按以下方式组织文件：

```
weights/
├── ve.pt
├── t3_mtl23ls_v2.safetensors
├── s3gen.pt
├── grapheme_mtl_merged_expanded_v1.json
├── Cangjie5_TC.json
└── conds.pt
```

**必需文件：**
- `ve.pt` - 语音编码器模型
- `t3_mtl23ls_v2.safetensors` - T3 多语言模型
- `s3gen.pt` - S3Gen 音频生成模型
- `grapheme_mtl_merged_expanded_v1.json` - 多语言分词器词汇表
- `Cangjie5_TC.json` - 中文仓颉转换映射（中文支持所需）
- `conds.pt` - 内置语音条件

**重要提示：**
- 确保 HuggingFace 网络连通性可用。即使本地文件存在，服务在初始化时仍可能从 HuggingFace 下载此处未列出的其他小文件。

### 选项 2：HuggingFace 预训练模型

如果未提供本地检查点，服务将在首次启动时自动从 HuggingFace 下载预训练模型。这需要互联网连接，可能需要一些时间。

## Docker 部署

### 构建 Docker 镜像

**注意：** 对于 amd64 平台，您可以直接使用 Docker Hub 上预构建的镜像，无需本地构建：

```bash
# 选项 1：使用预构建镜像（amd64 平台推荐）
docker pull dockersenseyang/service_chatterbox:latest
```

如果您需要自己构建镜像（例如，用于其他平台或自定义修改），服务包含支持 CUDA 的容器化部署 Dockerfile：

```bash
# 选项 2：从源码构建
docker build -f service/Dockerfile -t dockersenseyang/service_chatterbox:latest .
```

### 使用 Docker 运行

假设您已按照前面章节所述准备好 `data/` 和 `weights/` 目录，运行容器：

```bash
docker run -d \
  --gpus all \
  -p 18085:18085 \
  -v $(pwd)/data:/workspace/chatterbox/data \
  -v $(pwd)/weights:/workspace/chatterbox/weights \
  -v $(pwd)/logs:/workspace/chatterbox/logs \
  dockersenseyang/service_chatterbox:latest
```

## 本地开发环境设置

### 前置要求

- Python 3.10 或更高版本
- 支持 CUDA 的 GPU（推荐）或 CPU
- CUDA 12.4+（用于 GPU 支持）

### 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/LazyBusyYang/chatterbox
   cd chatterbox
   ```

2. **创建 conda 环境：**
   ```bash
   conda create -n chatterbox python=3.10 -y
   conda activate chatterbox
   ```

3. **安装 PyTorch（支持 CUDA）：**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
     --index-url https://download.pytorch.org/whl/cu124
   ```

4. **安装依赖：**
   ```bash
   pip install -e .
   ```

### 配置

编辑 `service/config.py` 以自定义服务器设置：

```python
type = 'FastAPIServer'
checkpoint_dir = "weights"          # 模型检查点路径
audio_prompts_dir = "data"          # 参考音频文件路径
host = '0.0.0.0'                    # 服务器主机
port = 18085                        # 服务器端口
logger_cfg = __logger_cfg__         # 日志配置
```

### 运行服务

1. **准备目录：**
   ```bash
   mkdir -p data weights logs
   ```

2. **放置参考音频和检查点：**
   - 将参考音频文件复制到 `data/`
   - 将模型检查点文件复制到 `weights/`（或留空以使用 HuggingFace）

3. **启动服务器：**
   ```bash
   python service/main.py --config_path service/config.py
   ```

   或使用自定义配置：
   ```bash
   python service/main.py --config_path path/to/your/config.py
   ```

4. **访问 API：**
   - API 文档：http://localhost:18085/docs
   - 健康检查：http://localhost:18085/health
   - 列出音色：http://localhost:18085/api/v1/list_voice_names

## API 使用示例

### 列出可用音色

```bash
curl http://localhost:18085/api/v1/list_voice_names
```

响应：
```json
{
  "voice_names": {
    "刻晴_zh": "刻晴",
    "keqing_en": "keqing"
  }
}
```

### 生成音频

```bash
curl -X POST http://localhost:18085/api/v1/generate_audio \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice_key": "keqing_en"
  }' \
  --output output.wav
```

### Python 客户端示例

```python
import requests

# 列出音色
response = requests.get("http://localhost:18085/api/v1/list_voice_names")
voices = response.json()["voice_names"]
print(f"可用音色: {list(voices.keys())}")

# 生成音频
response = requests.post(
    "http://localhost:18085/api/v1/generate_audio",
    json={
        "text": "Hello, this is a test.",
        "voice_key": "keqing_en"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 许可证

本仓库是 [https://github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) 的 fork，遵循原仓库的 MIT License。详见 LICENSE 文件。

