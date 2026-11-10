# VocalDNA-TTS

作者：**王大粘**

VocalDNA-TTS 是**不绑定模型的通用中文 TTS 增强组件**。它负责文本规范化、发音查表、韵律参数和音频后处理，实际语音由你选择的 TTS 引擎生成。项目随附 **578,895 条中文发音词典**，查词和变调可独立使用。

仓库包含词典和源代码，不包含任何模型权重、音色权重或音频样本。

## 能力

- G2P：词典最长匹配、领域筛选、用户覆盖词条、词性/领域元数据、标点边界、未知字回退和普通话变调。
- FAME：`joy`、`calm`、`anger`、`sadness`、`confidence`、`formality`、`surprise`、`tenderness` 八个 0-1 参数；映射到速度、音高、增益、停顿和重音。
- 引擎：实现 `SynthesisEngine` 即可接入现有的本地引擎或服务。Kokoro 和 Bert-VITS2-Extra 是可选适配器；是否能控制读音、情感和语速取决于所接引擎的接口。
- VEDM：可复现的能量衰减、低能量微颤和句间恢复。它不伪造或插入呼吸音。
- SCAC：接入可选音素/ASR 对齐器；发现不一致时重新合成，最多尝试指定次数。未提供校验器时会明确返回 `not_configured`。
- 流式生成：按句产出；`prefetch=True` 时在消费当前句期间预生成下一句。
- 本地 API：只绑定本机回环地址时，可提供 G2P JSON 和 WAV 合成端点。

## 安装

```bash
python -m pip install .
```

词典随包安装，`G2PTable.from_resource()` 会自动加载。JSON 或 gzip JSON 格式的自定义词典可通过 `G2PTable.from_json(path)` 加载。大型词典加载使用 `ijson` 流式读取。

## 独立查发音

```bash
vocaldna g2p "北京国家会计学院"
vocaldna g2p "项目词条" --domain 财务
```

## 接入你的 TTS

例如，在 Python 可导入的 `my_adapter` 模块中，将你的引擎包装成以下接口。示例中的 `my_tts.synthesize` 由你的 TTS 提供，返回一维单声道浮点波形（通常在 -1 到 1 之间）和采样率：

```python
from vocaldna import AudioResult
from my_tts import synthesize as backend_synthesize


class MyEngine:
    default_voice = "narrator"

    def synthesize(self, text, *, voice, speed, fame, pronunciation, pronunciation_tokens):
        audio, sample_rate = backend_synthesize(text, voice=voice, speed=speed)
        return AudioResult(audio, sample_rate, voice)


def make_engine():
    return MyEngine()
```

`text` 已经规范化，`pronunciation` 是数字声调拼音序列，`pronunciation_tokens` 保留分词及词条信息。支持发音控制的引擎可以使用这些输入，支持韵律控制的引擎可以读取 `fame`。只接收文本的引擎仍可合成，但不保证按词典读音发声。未指定音色时使用适配器的 `default_voice`，未声明时传入 `default`。

在 Python 中传入引擎实例：

```python
from my_adapter import make_engine
from vocaldna import FAME, G2PTable, VocalDNA

pipeline = VocalDNA(
    g2p=G2PTable.from_resource(),
    engine=make_engine(),
)

for chunk in pipeline.synthesize_stream(
    "第一句。第二句。",
    fame=FAME(calm=0.5, confidence=0.4),
    prefetch=True,
):
    print(chunk.text, chunk.sample_rate, len(chunk.audio))
```

命令行和本机 HTTP 服务都可用 `--engine 模块:工厂函数` 载入同一个适配器。工厂函数不接收参数，由启动服务的人指定：

```bash
vocaldna synthesize "会计" --engine my_adapter:make_engine --output out.wav
vocaldna serve --engine my_adapter:make_engine
```

- `GET /health`
- `POST /v1/g2p`，JSON：`{"text":"会计"}`
- `POST /v1/tts`，JSON：`{"text":"你好","fame":{"calm":0.5}}`，响应为 WAV。可选 `voice` 字段覆盖默认音色；响应头 `X-VocalDNA-Voice` 使用 URL 百分号编码。

直接运行 `vocaldna serve` 会启动纯查词服务；未配置引擎时，合成端点返回 `503`。API 默认监听 `127.0.0.1`，不提供身份验证，请勿直接暴露在公网。

## 可选适配器

**Kokoro**：自备模型目录，其中应有 `config.json`、`kokoro-v1_1-zh.pth` 和 `voices/`。适配器使用词典覆盖中文拼音查询，并将 FAME 映射到合成速度。

```powershell
python -m pip install ".[kokoro]"
$env:VOCALDNA_KOKORO_MODEL_DIR = "D:\models\Kokoro-82M-v1.1-zh"
vocaldna synthesize "会计" --engine kokoro --output out.wav
```

**Bert-VITS2-Extra**：`BertVITS2Engine` 接收外部 `infer` 回调，提供表内读音、FAME 文本条件和八维向量。可通过 `condition` 回调接入 CLAP。`make_cleaner()` 提供与 Bert-VITS2 中文 `phone/tone/word2ph` 输入契约兼容的查表清理器。推理程序和模型需自行提供。

## 许可

本项目及随附词典按根目录 [`LICENSE`](LICENSE) 以**非商业许可**发布，作者仅为**王大粘**。商业使用须事先取得作者书面授权。第三方模型和依赖分别遵守其原始许可；本项目不包含模型权重。

## 测试

```bash
python -m unittest discover -s tests -v
python -m build
```
