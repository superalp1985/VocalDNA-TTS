# VocalDNA-TTS

作者：**王大粘**

VocalDNA-TTS 是面向中文语音合成的查表优先架构。项目包含完整的 **578,895 条中文发音词典**，并提供最长匹配 G2P、变调、FAME 八维韵律控制、可替换的 Bert-VITS2-Extra/Kokoro 引擎接口、VEDM 音频能量处理、SCAC 发音校验重试和按句流式生成。

仓库包含词典和源代码，不包含任何模型权重、音色权重或音频样本。

## 能力

- G2P：词典最长匹配、领域筛选、用户覆盖词条、词性/领域元数据、标点边界、未知字回退和普通话变调。
- FAME：`joy`、`calm`、`anger`、`sadness`、`confidence`、`formality`、`surprise`、`tenderness` 八个 0-1 参数；映射到速度、音高、增益、停顿和重音。
- 引擎：统一 `SynthesisEngine` 协议；包含 Kokoro 中文本地引擎，以及可连接外部 Bert-VITS2-Extra 推理函数的适配器。Kokoro 的实际韵律控制受其引擎能力限制。
- VEDM：可复现的能量衰减、低能量微颤和句间恢复。它不伪造或插入呼吸音。
- SCAC：接入可选音素/ASR 对齐器；发现不一致时重新合成，最多尝试指定次数。未提供校验器时会明确返回 `not_configured`。
- 流式生成：按句产出；`prefetch=True` 时在消费当前句期间预生成下一句。
- 本地 API：只绑定本机回环地址时，可提供 G2P JSON 和 WAV 合成端点。

## 安装

```bash
python -m pip install .
```

安装 Kokoro 引擎依赖：

```bash
python -m pip install ".[kokoro]"
```

词典随包安装，`G2PTable.from_resource()` 会自动加载。JSON 或 gzip JSON 格式的自定义词典可通过 `G2PTable.from_json(path)` 加载。大型词典加载使用 `ijson` 流式读取。

## 使用

只查发音：

```bash
vocaldna g2p "北京国家会计学院"
vocaldna g2p "项目词条" --domain 财务
```

合成 WAV。模型文件需自行准备，目录内应有 Kokoro 的 `config.json`、`kokoro-v1_1-zh.pth` 和 `voices/`：

```powershell
$env:VOCALDNA_KOKORO_MODEL_DIR = "D:\models\Kokoro-82M-v1.1-zh"
vocaldna synthesize "你好，欢迎使用 VocalDNA。" --voice zf_001 --output out.wav
```

Python 调用：

```python
from vocaldna import FAME, G2PTable, KokoroEngine, VocalDNA

pipeline = VocalDNA(
    g2p=G2PTable.from_resource(),
    engine=KokoroEngine(),
)

for sentence in pipeline.synthesize_stream(
    "第一句。第二句。",
    voice="zf_001",
    fame=FAME(calm=0.5, confidence=0.4),
    prefetch=True,
):
    consume(sentence.audio, sentence.sample_rate)
```

启动本地 HTTP API：

```powershell
vocaldna serve --host 127.0.0.1 --port 8765
```

- `GET /health`
- `POST /v1/g2p`，JSON：`{"text":"会计"}`
- `POST /v1/tts`，JSON：`{"text":"你好","voice":"zf_001","fame":{"calm":0.5}}`，响应为 WAV

API 默认只监听 `127.0.0.1`。没有身份验证，不要将监听地址改成公网或不受信任的网络接口。

## Bert-VITS2-Extra 与 CLAP

方案目标中的 Bert-VITS2-Extra 权重、CLAP 模型和对应推理源码不在 `D:\Tianyu\V5` 中，因此未伪造模型文件或推理结果。`BertVITS2Engine` 接收外部 `infer` 回调；调用时传入表内读音、FAME 文本条件和八维向量。可通过 `condition` 回调接入已获授权的 CLAP 实例。`make_cleaner()` 提供与 Bert-VITS2 中文 `phone/tone/word2ph` 输入契约兼容的查表清理器。

Kokoro 适配器使用本词典覆盖其中文拼音查询，并使用 FAME 的速度控制。由于 Kokoro 接口本身不提供独立音高/CLAP 控制，相关字段不会被描述为 Kokoro 的模型内情感控制。

## 许可

本项目及随附词典按根目录 [`LICENSE`](LICENSE) 以**非商业许可**发布，作者仅为**王大粘**。商业使用须事先取得作者书面授权。第三方模型和依赖分别遵守其原始许可；本项目不包含模型权重。

## 测试

```bash
python -m unittest discover -s tests -v
python -m build
```
