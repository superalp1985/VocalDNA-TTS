from vocaldna import FAME, G2PTable, KokoroEngine, VocalDNA


table = G2PTable.from_resource()
pipeline = VocalDNA(
    g2p=table,
    engine=KokoroEngine(),
)

for chunk in pipeline.synthesize_stream(
    "欢迎使用 VocalDNA。现在开始生成长文本语音。",
    voice="zf_001",
    fame=FAME(calm=0.6, confidence=0.4),
    prefetch=True,
):
    print(chunk.index, chunk.text, len(chunk.audio))
