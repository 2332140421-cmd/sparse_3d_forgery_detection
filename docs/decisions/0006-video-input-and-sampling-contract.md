# ADR 0006: Video input and sampling contract

## Status

Accepted

## Context

The depth, tracking, and camera frontends require a shared, reproducible source-video timeline before they can construct a `ParticleSequence`. This boundary must preserve decoder observations without growing into a general media framework.

## Decision

- The caller supplies `sample_id`, `source_video_id`, and the I/O locator; identity is not inferred from a path.
- Frames retain strictly increasing source-frame indices and finite, strictly increasing timestamps supplied by the decoder. Variable frame-rate timestamps are valid and are not reconstructed as `frame_index / fps`.
- RGB frames are decoder raster output with dtype and shape `uint8 [H, W, 3]`; the boundary does not silently cast, resize, crop, normalize, sort, deduplicate, or repair input.
- Sampling supports only validated explicit indices and deterministic fixed-stride indices.
- Invalid input is exposed with field-specific errors.

## Deferred decisions

- The real decoder and media dependency;
- FPS fallback, seeking, and uniform-count sampling;
- resize, crop, pixel-center convention, UV mapping, and intrinsics mapping;
- depth, tracking, pose, and camera compensation;
- artifact and storage formats.

## Consequences

- Future decoder adapters must supply real timestamps and return the frozen frame representation.
- Downstream frontends can share frame identity, time, raster size, and RGB observations without depending on a media backend.

## Supersedes

This ADR does not supersede ADR 0001 through ADR 0005. It adds the minimal video timeline boundary that precedes the existing `ParticleSequence` contract.
