#!/usr/bin/env python3
"""Write a per-layer MoE top-k table into an existing GGUF model.

The table is read by llama.cpp-omni as {arch}.moe_expert_prun and
applied to decode only (small batches). Prefill keeps the model default.

The GGUF header is rebuilt and the tensor data is copied as-is, so this works on
files that already exist without re-running the conversion.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# tools/moe/set_topk.py -> parents[2] = repo root
LLAMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

KEY_SUFFIX = "moe_expert_prun"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def scalar(reader: GGUFReader, key: str, default: object = None) -> object:
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def parse_topk(spec: str, n_layer: int, expert_used: int) -> list[int]:
    if spec.startswith("@"):
        raw = json.loads(Path(spec[1:]).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            values = [int(raw.get(str(il), raw.get(f"layer_{il}", 0))) for il in range(n_layer)]
        elif isinstance(raw, list):
            values = [int(v) for v in raw]
        else:
            fail(f"{spec}: expected a JSON list or an object keyed by layer index")
    else:
        values = [int(v) for v in spec.replace(" ", "").split(",") if v != ""]

    if len(values) == 1:
        values = values * n_layer
    if len(values) > n_layer:
        fail(f"table has {len(values)} values, model has {n_layer} layers")
    values += [0] * (n_layer - len(values))

    for il, value in enumerate(values):
        if value < 0 or value > expert_used:
            fail(f"layer {il}: {value} outside [0, {expert_used}]")

    return values


def check_layout(reader: GGUFReader, src: Path) -> None:
    offset = 0
    for tensor in reader.tensors:
        expected = reader.data_offset + offset
        if expected != tensor.data_offset:
            fail(
                f"tensor '{tensor.name}' starts at {tensor.data_offset}, expected {expected}; "
                "tensor data is not laid out in tensor-info order, refusing to rebuild"
            )
        offset += (tensor.n_bytes + reader.alignment - 1) // reader.alignment * reader.alignment

    if reader.data_offset + offset != src.stat().st_size:
        fail("tensor data does not end at EOF, refusing to rebuild")


def rewrite(reader: GGUFReader, src: Path, dst: Path, key: str, topk: list[int]) -> None:
    arch = str(scalar(reader, "general.architecture", ""))
    if not arch:
        fail("model has no general.architecture")

    writer = GGUFWriter(dst, arch, use_temp_file=False)
    writer.data_alignment = reader.alignment

    skip = {"general.architecture", key}
    for name, field in reader.fields.items():
        if name in skip or name.startswith("GGUF.") or not field.types:
            continue
        sub_type = field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), field.types[0], sub_type)

    writer.add_array(key, topk)

    for tensor in reader.tensors:
        # tensor.shape is stored in GGUF order; writer wants ne0 first.
        # dtype only matters when raw_dtype is None, so a placeholder is fine here.
        writer.add_tensor_info(
            tensor.name,
            list(reversed(tensor.shape.tolist())),
            np.float32,
            int(tensor.n_bytes),
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    fout = writer.fout[0]
    padding = (-fout.tell()) % writer.data_alignment
    if padding:
        fout.write(b"\0" * padding)

    with src.open("rb") as fsrc:
        fsrc.seek(reader.data_offset)
        shutil.copyfileobj(fsrc, fout, 16 << 20)

    fout.flush()
    fout.close()


def verify(dst: Path, key: str, topk: list[int]) -> None:
    reader = GGUFReader(dst, "r")
    field = reader.get_field(key)
    if field is None:
        fail(f"rebuilt file has no '{key}'")
    if field.contents() != topk:
        fail(f"rebuilt table does not match: {field.contents()}")
    print(f"wrote {len(topk)} values to {dst}")
    print(f"  {key} = {topk}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set per-layer MoE top-k in a GGUF file")
    parser.add_argument("--model", required=True, type=Path, help="source GGUF (single file)")
    parser.add_argument("--topk", required=True,
                        help="comma separated values, a single value, or @table.json (list or layer -> topk)")
    parser.add_argument("--out", type=Path, help="output GGUF (default: <model>-topk.gguf)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.model.expanduser().resolve()
    if not src.is_file():
        fail(f"{src} is not a file")

    reader = GGUFReader(src, "r")
    if int(scalar(reader, "split.count", 1)) != 1:
        fail("split models are not supported, merge the shards first")

    arch = str(scalar(reader, "general.architecture", ""))
    n_layer = int(scalar(reader, f"{arch}.block_count", 0))
    n_expert = int(scalar(reader, f"{arch}.expert_count", 0))
    n_expert_used = int(scalar(reader, f"{arch}.expert_used_count", 0))
    if n_layer <= 0 or n_expert <= 0 or n_expert_used <= 0:
        fail(f"{arch} is not a routed MoE model (block_count={n_layer}, expert_count={n_expert}, expert_used_count={n_expert_used})")

    topk = parse_topk(args.topk, n_layer, n_expert_used)
    key = f"{arch}.{KEY_SUFFIX}"

    check_layout(reader, src)

    dst = args.out.expanduser().resolve() if args.out else src.with_name(f"{src.stem}-topk{src.suffix}")
    if dst == src:
        fail("output must differ from the input")
    if dst.exists() and not args.overwrite:
        fail(f"{dst} exists, pass --overwrite")

    print(f"source: {src}")
    print(f"  arch={arch} layers={n_layer} experts={n_expert} default top-k={n_expert_used}")
    rewrite(reader, src, dst, key, topk)
    verify(dst, key, topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
