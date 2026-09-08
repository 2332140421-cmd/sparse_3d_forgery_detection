"""Minimal read-only HTTP-Range ZIP64 access for targeted Charades members.

This is an experiment-local adapter.  It accepts only exact ``206`` range
responses and never downloads or extracts the full archive.
"""

from __future__ import annotations

import binascii
import re
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class RangeProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CentralEntry:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RangeClient:
    def __init__(self, url: str, *, timeout: float = 120.0, opener: Callable[..., object] | None = None):
        self.url = url
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen
        self.size: int | None = None

    def head(self) -> int:
        request = urllib.request.Request(self.url, method="HEAD")
        with self._opener(request, timeout=self.timeout) as response:  # type: ignore[call-arg]
            value = response.headers.get("Content-Length")
        if value is None:
            raise RangeProtocolError("archive HEAD has no Content-Length")
        self.size = int(value)
        return self.size

    def get_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError("invalid byte range")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                with self._opener(request, timeout=self.timeout) as response:  # type: ignore[call-arg]
                    status = getattr(response, "status", None)
                    headers = response.headers
                    body = response.read()
                break
            except OSError as exc:
                last_error = exc
                if attempt == 2:
                    raise
        else:
            raise last_error or OSError("range request failed")
        if status != 206:
            raise RangeProtocolError(f"range response must be HTTP 206, got {status}")
        content_range = headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
        expected = end - start + 1
        if match is None or tuple(map(int, match.groups())) != (start, end, int(match.group(3))):
            raise RangeProtocolError(f"invalid Content-Range for {start}-{end}: {content_range!r}")
        if len(body) != expected:
            raise RangeProtocolError(f"range length mismatch: expected {expected}, got {len(body)}")
        if self.size is not None and int(match.group(3)) != self.size:
            raise RangeProtocolError("archive size changed during range access")
        return body


def _zip64_extra(extra: bytes, *, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
    cursor = 0
    values = {"compressed": compressed, "uncompressed": uncompressed, "offset": offset}
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + field_size]
        cursor += 4 + field_size
        if field_id != 0x0001:
            continue
        pos = 0
        if values["uncompressed"] == 0xFFFFFFFF:
            values["uncompressed"] = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
        if values["compressed"] == 0xFFFFFFFF:
            values["compressed"] = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
        if values["offset"] == 0xFFFFFFFF:
            values["offset"] = struct.unpack_from("<Q", payload, pos)[0]
        return values["compressed"], values["uncompressed"], values["offset"]
    if 0xFFFFFFFF in values.values():
        raise ValueError("ZIP64 sentinel without ZIP64 extra field")
    return values["compressed"], values["uncompressed"], values["offset"]


def parse_zip64_end(tail: bytes, archive_size: int) -> tuple[int, int, int]:
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError("EOCD not found")
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment = struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk != 0 or cd_disk != 0 or disk_entries != entries:
        raise ValueError("multi-disk ZIP is unsupported")
    if entries != 0xFFFF and cd_size != 0xFFFFFFFF and cd_offset != 0xFFFFFFFF:
        return entries, cd_offset, cd_size
    locator = tail.rfind(b"PK\x06\x07", 0, eocd)
    if locator < 0 or locator + 20 > len(tail):
        raise ValueError("ZIP64 locator not found")
    _, locator_disk, zip64_offset, disk_count = struct.unpack_from("<4sLQL", tail, locator)
    if locator_disk != 0 or disk_count != 1:
        raise ValueError("multi-disk ZIP64 archive is unsupported")
    if zip64_offset + 56 > archive_size:
        raise ValueError("ZIP64 EOCD offset out of bounds")
    raise ValueError("ZIP64 EOCD is outside the supplied tail; fetch it with read_central_directory")


def read_central_directory(client: RangeClient, *, tail_size: int = 131072) -> tuple[list[CentralEntry], dict[str, int]]:
    size = client.size if client.size is not None else client.head()
    tail_start = max(0, size - tail_size)
    tail = client.get_range(tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("EOCD not found")
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk != 0 or cd_disk != 0 or disk_entries != entries:
        raise ValueError("multi-disk ZIP is unsupported")
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        if locator < 0:
            raise ValueError("ZIP64 locator not found")
        _, locator_disk, zip64_offset, disk_count = struct.unpack_from("<4sLQL", tail, locator)
        if locator_disk != 0 or disk_count != 1:
            raise ValueError("multi-disk ZIP64 archive is unsupported")
        record = client.get_range(zip64_offset, zip64_offset + 55)
        if record[:4] != b"PK\x06\x06":
            raise ValueError("ZIP64 EOCD signature missing")
        _, record_size, _, _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset = struct.unpack_from("<4sQ2H2L4Q", record)
        if record_size < 44 or disk != 0 or cd_disk != 0 or disk_entries != entries:
            raise ValueError("invalid ZIP64 EOCD")
    central = client.get_range(cd_offset, cd_offset + cd_size - 1)
    result: list[CentralEntry] = []
    cursor = 0
    while cursor < len(central):
        if central[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError(f"central-directory signature missing at {cursor}")
        if cursor + 46 > len(central):
            raise ValueError("truncated central-directory entry")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        flags, method = fields[3], fields[4]
        crc, compressed, uncompressed = fields[7], fields[8], fields[9]
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        end = cursor + 46 + name_len + extra_len + comment_len
        if end > len(central):
            raise ValueError("central-directory entry exceeds range")
        raw_name = central[cursor + 46 : cursor + 46 + name_len]
        extra = central[cursor + 46 + name_len : cursor + 46 + name_len + extra_len]
        compressed, uncompressed, offset = _zip64_extra(extra, compressed=compressed, uncompressed=uncompressed, offset=fields[-1])
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        result.append(CentralEntry(name, flags, method, crc, compressed, uncompressed, offset))
        cursor = end
    if len(result) != entries:
        raise ValueError(f"central entry count mismatch: {len(result)} != {entries}")
    return result, {"archive_bytes": size, "entries": entries, "central_offset": cd_offset, "central_size": cd_size}


def exact_basename(entries: list[CentralEntry], basename: str) -> CentralEntry:
    matches = [entry for entry in entries if Path(entry.name).name == basename]
    if len(matches) != 1:
        raise LookupError(f"expected one exact basename {basename!r}, found {len(matches)}")
    return matches[0]


def extract_member(client: RangeClient, entry: CentralEntry, output_part: Path) -> None:
    if entry.flags & 0x1:
        raise ValueError("encrypted ZIP member is unsupported")
    local = client.get_range(entry.local_header_offset, entry.local_header_offset + 29)
    if local[:4] != b"PK\x03\x04":
        raise ValueError("local-header signature missing")
    _, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from("<4s5H3L2H", local)
    if flags & 0x1 or method != entry.method:
        raise ValueError("local header disagrees with central entry or is encrypted")
    payload_start = entry.local_header_offset + 30 + name_len + extra_len
    payload_end = payload_start + entry.compressed_size - 1
    payload = client.get_range(payload_start, payload_end) if entry.compressed_size else b""
    if method == 0:
        decoded = payload
    elif method == 8:
        decoded = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported ZIP compression method {method}")
    if len(decoded) != entry.uncompressed_size:
        raise ValueError("uncompressed size mismatch")
    if (binascii.crc32(decoded) & 0xFFFFFFFF) != entry.crc32:
        raise ValueError("CRC32 mismatch")
    output_part.parent.mkdir(parents=True, exist_ok=True)
    with output_part.open("wb") as handle:
        handle.write(decoded)
        handle.flush()
    output_part.replace(output_part.with_suffix(""))
