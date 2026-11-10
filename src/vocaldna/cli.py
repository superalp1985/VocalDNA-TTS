from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

from .engine import KokoroEngine
from .g2p import G2PTable
from .pipeline import VocalDNA


def _table(args):
    return G2PTable.from_json(args.lexicon) if args.lexicon else G2PTable.from_resource()


def _pipeline(args):
    return VocalDNA(g2p=_table(args), engine=args.engine)


def _load_engine(spec, model_dir):
    if spec == "kokoro" or (spec is None and model_dir is not None):
        return KokoroEngine(model_dir)
    if model_dir is not None:
        raise ValueError("--model-dir is only supported by the Kokoro adapter")
    if spec is None:
        return None
    module_name, _, factory_name = spec.partition(":")
    if not all(part.isidentifier() for part in module_name.split(".")) or not factory_name.isidentifier():
        raise ValueError("--engine must be 'kokoro' or a module:factory")
    factory = getattr(import_module(module_name), factory_name)
    if not callable(factory):
        raise ValueError(f"engine factory is not callable: {spec}")
    engine = factory()
    if not callable(getattr(engine, "synthesize", None)):
        raise ValueError(f"engine factory must return an object with synthesize(): {spec}")
    return engine


def _parser():
    parser = argparse.ArgumentParser(prog="vocaldna")
    commands = parser.add_subparsers(dest="command", required=True)

    g2p = commands.add_parser("g2p", help="查词典并输出拼音")
    g2p.add_argument("text")
    g2p.add_argument("--domain", action="append")
    g2p.add_argument("--lexicon")
    g2p.set_defaults(handler=_run_g2p)

    synth = commands.add_parser("synthesize", help="合成 WAV 音频")
    synth.add_argument("text")
    synth.add_argument("--voice", help="音色 ID，省略时由引擎决定")
    synth.add_argument("--speed", type=float, default=1.0)
    synth.add_argument("--model-dir")
    synth.add_argument("--lexicon")
    synth.add_argument("--output", default="vocaldna.wav")
    synth.set_defaults(handler=_run_synthesize)

    serve = commands.add_parser("serve", help="启动本机 HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model-dir")
    serve.add_argument("--lexicon")
    serve.set_defaults(handler=_run_server)

    for command in (synth, serve):
        command.add_argument("--engine", help="kokoro 或 Python 模块:工厂函数")
    return parser


def _run_g2p(args):
    table = _table(args)
    tokens = table.convert(args.text, domain=args.domain)
    print(json.dumps({
        "text": args.text,
        "pronunciation": table.pronounce(args.text, domain=args.domain),
        "tokens": [
            {"surface": token.surface, "pinyin": token.pinyin, "kind": token.kind, "metadata": token.metadata}
            for token in tokens
        ],
    }, ensure_ascii=False, indent=2))


def _run_synthesize(args):
    import soundfile as sf

    result = _pipeline(args).synthesize(args.text, voice=args.voice, speed=args.speed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.audio, result.sample_rate)
    print(json.dumps({
        "output": str(output.resolve()),
        "sampleRate": result.sample_rate,
        "voice": result.voice,
        "normalizedText": result.normalized_text,
        "metadata": result.metadata,
    }, ensure_ascii=False, indent=2))


def _run_server(args):
    from .server import serve

    pipeline = _pipeline(args)
    print(f"VocalDNA API listening on http://{args.host}:{args.port}")
    serve(pipeline, host=args.host, port=args.port)


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.command != "g2p":
        if args.command == "synthesize" and args.engine is None and args.model_dir is None:
            parser.error("synthesize requires --engine (or --model-dir for Kokoro)")
        try:
            args.engine = _load_engine(args.engine, args.model_dir)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            parser.error(f"could not load engine: {error}")
    args.handler(args)


if __name__ == "__main__":
    main()
