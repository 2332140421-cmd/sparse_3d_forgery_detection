# ADR 0007: PyAV local video decoding

## Status

Accepted

## Context

The accepted video input contract requires real MP4 frames and source PTS timestamps before depth and tracking can run on the research dataset.

## Decision

- PyAV 18.1.0 is the sole media dependency for the initial local MP4 decoder.
- Decoding is sequential and retains only caller-requested, strictly increasing source-frame indices.
- Timestamps come from each requested frame's PTS and time base; FPS-based fallback is not permitted.
- Decoded output is unmodified-size RGB `uint8 [H, W, 3]`; no resize, crop, normalization, cache, remote input, registry, or backend factory is introduced.

## Consequences

DeeptraceReward MP4 files can enter the existing provider-neutral video boundary. Pixel resizing and all depth, tracking, pose, particle, and training choices remain deferred.

## Supersedes

This ADR resolves only the real-decoder and media-dependency item deferred by ADR 0006. It does not supersede ADR 0001 through ADR 0006.
