#!/usr/bin/env python3
from __future__ import annotations
import argparse
import io
import json
import mimetypes
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field

HAB_MAGIC = b"HAB\x00"
HAB_HEADER = struct.Struct("<4sHHIII")

class BundleError(Exception):
    pass


@dataclass
class Entry:
    name: str
    data: bytes
    mime: str = ""
    stored_length: int = 0
    compression: str = "deflate"


@dataclass
class Bundle:
    kind: str
    name: str
    entries: list[Entry] = field(default_factory=list)
    header: dict = field(default_factory=dict)
    manifest: dict | None = None

    def get(self, suffix: str) -> Entry | None:
        for e in self.entries:
            if e.name.lower().endswith(suffix):
                return e
        return None


def _decomp(buf: bytes, expected: int | None, w: str) -> bytes:
    try:
        out = zlib.decompress(buf)
    except zlib.error as exc:
        raise BundleError(f"{w}: zlib error: {exc}") from None
    if expected is not None and len(out) != expected:
        raise BundleError(f"{w}: header mismatch ({len(out)} bytes vs {expected})")
    return out


def read_hab(data: bytes) -> Bundle:
    if len(data) < HAB_HEADER.size or data[:4] != HAB_MAGIC:
        raise BundleError("bad .hab magic")
    magic, version, flags, m_stored, m_orig, p_len = HAB_HEADER.unpack_from(data)
    m_start = HAB_HEADER.size
    p_start = m_start + m_stored
    if p_start + p_len != len(data):
        raise BundleError(f"size mismatch ({p_start + p_len} bytes declared vs {len(data)})")
    r_manifest = data[m_start:p_start]
    if r_manifest[:1] == b"{":
        manifest_bytes = r_manifest
    else:
        manifest_bytes = _decomp(r_manifest, m_orig, "manifest")
    manifest = json.loads(manifest_bytes)
    payload = data[p_start:]

    b = Bundle("hab", manifest.get("name", ""), manifest=manifest, header={
            "version": version,
            "flags": flags,
            "manifestStored": m_stored,
            "manifestOriginal": m_orig,
            "payloadLength": p_len,
        },
    )
    for i, e in enumerate(manifest.get("entries", [])):
        off, size = e["offset"], e["storedLength"]
        if off < 0 or off + size > len(payload):
            raise BundleError(f"entry {e['name']!r} out of bounds")
        blob = payload[off : off + size]
        comp = e.get("compression", "none")
        if comp == "deflate":
            content = _decomp(blob, e.get("originalLength"), e["name"])
        elif comp == "none":
            content = blob
        else:
            raise BundleError(f"entry {e['name']!r}: unknown compression {comp!r}")
        b.entries.append(Entry(e["name"], content, e.get("mimeType", ""), size, comp))
    return b


def read_nitro(data: bytes) -> Bundle:
    o = 0

    def take(n: int) -> bytes:
        nonlocal o
        if o + n > len(data):
            raise BundleError("truncated .nitro file")
        chunk = data[o : o + n]
        o += n
        return chunk

    (count,) = struct.unpack(">H", take(2))
    b = Bundle("nitro", "")
    for _ in range(count):
        (nl,) = struct.unpack(">H", take(2))
        name = take(nl).decode("utf-8")
        (dl,) = struct.unpack(">I", take(4))
        content = _decomp(take(dl), None, name)
        b.entries.append(Entry(name, content, _get_mime(name), dl, "deflate"))
    if o != len(data):
        raise BundleError(f"{len(data) - o} bytes left")
    j = b.get(".json")
    b.name = (
        j.name.rsplit(".", 1)[0]
        if j
        else (b.entries[0].name.rsplit(".", 1)[0] if b.entries else "")
    )
    b.header = {"fileCount": count}
    return b


def read_any(path: str) -> Bundle:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] == HAB_MAGIC:
        return read_hab(data)
    return read_nitro(data)


def _get_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def write_hab(name: str, entries: list[tuple[str, bytes]], level: int = 6) -> bytes:
    payload = bytearray()
    table = []
    for fname, content in entries:
        packed = zlib.compress(content, level)

        if len(packed) < len(content):
            blob, comp = packed, "deflate"
        else:
            blob, comp = content, "none"
        table.append(
            {
                "name": fname,
                "mimeType": _get_mime(fname),
                "offset": len(payload),
                "storedLength": len(blob),
                "originalLength": len(content),
                "compression": comp
            }
        )
        payload += blob
    manifest = json.dumps(
        {"format": "hab", "version": 1, "name": name, "entries": table},
        separators=(",", ":"),
    ).encode()
    m_packed = zlib.compress(manifest, level)
    header = HAB_HEADER.pack(HAB_MAGIC, 1, 1, len(m_packed), len(manifest), len(payload))
    return header + m_packed + bytes(payload)


def write_nitro(entries: list[tuple[str, bytes]], level: int = 6) -> bytes:
    out = bytearray(struct.pack(">H", len(entries)))
    for fname, content in entries:
        n = fname.encode("utf-8")
        blob = zlib.compress(content, level)
        out += struct.pack(">H", len(n)) + n + struct.pack(">I", len(blob)) + blob
    return bytes(out)


def _get_json(target: str, content: bytes, name: str) -> bytes:
    """no documentClass needed for Nitro"""
    try:
        j = json.loads(content)
    except ValueError:
        return content
    if target == "nitro":
        j.pop("documentClass", None)
    else:
        j.setdefault("documentClass", j.get("name", name))
    return json.dumps(j, separators=(",", ":")).encode()


def bundle_to(target: str, b: Bundle) -> bytes:
    items = [
        (
            e.name,
            _get_json(target, e.data, b.name) if e.name.endswith(".json") else e.data
        )
        for e in b.entries
    ]
    return write_hab(b.name, items) if target == "hab" else write_nitro(items)


def cut_sprites(b: Bundle, outdir: str) -> int:
    try:
        from PIL import Image
    except ImportError:
        raise BundleError("run pip install pillow") from None
    j, png = b.get(".json"), b.get(".png")
    if not j or not png:
        raise BundleError("bundle has no json+png")
    meta = json.loads(j.data)
    frames = meta.get("spritesheet", {}).get("frames", {})
    sheet = Image.open(io.BytesIO(png.data)).convert("RGBA")
    os.makedirs(outdir, exist_ok=True)
    prefix = f"{b.name}_"
    for key, fr in frames.items():
        f = fr["frame"]
        img = sheet.crop((f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"]))
        if fr.get("rotated"):
            img = img.rotate(90, expand=True)
        fname = key[len(prefix) :] if key.startswith(prefix * 2) else key
        img.save(os.path.join(outdir, fname + ".png"))
    for asset, a in meta.get("assets", {}).items():
        src = a.get("source")
        if not src:
            continue
        p = os.path.join(outdir, src + ".png")
        if os.path.exists(p) and not os.path.exists(os.path.join(outdir, asset + ".png")):
            img = Image.open(p)
            if a.get("flipH"):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(os.path.join(outdir, asset + ".png"))
    return len(frames)


def cmd_extract(args) -> None:
    for path in args.files:
        b = read_any(path)
        outdir = os.path.join(args.output, b.name or os.path.splitext(os.path.basename(path))[0])
        os.makedirs(outdir, exist_ok=True)
        for e in b.entries:
            nm = os.path.basename(e.name)
            data = e.data
            if nm.endswith(".json") and args.pretty:
                data = json.dumps(json.loads(data), indent=2).encode()
            with open(os.path.join(outdir, nm), "wb") as fh:
                fh.write(data)
        if args.manifest and b.manifest is not None:
            with open(os.path.join(outdir, "_manifest.json"), "w") as fh:
                json.dump({"header": b.header, **b.manifest}, fh, indent=2)
        msg = f"{path} -> {outdir}/ ({len(b.entries)} files"
        if args.sprites:
            n = cut_sprites(b, os.path.join(outdir, "sprites"))
            msg += f", {n} sprites"
        print(msg + ")")


def cmd_convert(args) -> None:
    b = read_any(args.file)
    target = "nitro" if b.kind == "hab" else "hab"
    out = args.output or os.path.splitext(args.file)[0] + "." + target
    with open(out, "wb") as fh:
        fh.write(bundle_to(target, b))
    print(f"{args.file} (.{b.kind}) -> {out} (.{target})")


def cmd_pack(args) -> None:
    files: list[str] = []
    for p in args.inputs:
        if os.path.isdir(p):
            files += sorted(
                os.path.join(p, f)
                for f in os.listdir(p)
                if f.lower().endswith((".json", ".png")) and not f.startswith("_")
            )
        else:
            files.append(p)
    if not files:
        raise BundleError("files not found")
    files.sort(key=lambda f: (not f.endswith(".json"), f))
    items = [(os.path.basename(f), open(f, "rb").read()) for f in files]
    name = args.name or os.path.splitext(items[0][0])[0]
    target = "hab" if args.output.lower().endswith(".hab") else "nitro"
    items = [(n, _get_json(target, d, name) if n.endswith(".json") else d) for n, d in items]
    data = write_hab(name, items) if target == "hab" else write_nitro(items)
    with open(args.output, "wb") as fh:
        fh.write(data)
    print(f"packed {len(items)} files -> {args.output} (.{target})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="habtool",
        description=".hab/.nitro converter and exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract", help="unpack bundle contents")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--output", default="extracted")
    p.add_argument("--sprites", action="store_true", help="cut each frame to pngs")
    p.add_argument("--manifest", action="store_true", help="write the .hab manifest")
    p.add_argument("--pretty", action="store_true", help="prettify the json")
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("convert", help=".hab <-> .nitro")
    p.add_argument("file")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_convert)
    p = sub.add_parser("pack", help="bundle from .json/.png files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", required=True, help="out.hab or out.nitro")
    p.add_argument("--name", help="bundle/class name")
    p.set_defaults(fn=cmd_pack)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except (BundleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
