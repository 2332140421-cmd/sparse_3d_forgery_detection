"""Sequential local MP4 decoding with source PTS timestamps."""

from collections.abc import Sequence
from pathlib import Path

import av

from .sampling import explicit_frame_indices
from .schema import DecodedFrame, DecodedVideoSample, VideoSource


class VideoDecodingError(RuntimeError):
    """Raised when requested source frames cannot be decoded faithfully."""


def decode_video(
    source: VideoSource,
    frame_indices: Sequence[int],
) -> DecodedVideoSample:
    """Decode requested zero-based source frames in one sequential pass."""

    requested = explicit_frame_indices(frame_indices)
    path = Path(source.source_locator)
    if not path.is_file():
        raise VideoDecodingError(f"local video file does not exist: {path}")

    targets = requested.tolist()
    frames: list[DecodedFrame] = []
    target_offset = 0

    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise VideoDecodingError("media contains no video stream")
            stream = container.streams.video[0]

            for source_frame_index, frame in enumerate(container.decode(stream)):
                if source_frame_index != targets[target_offset]:
                    continue
                if frame.pts is None or frame.time_base is None:
                    raise VideoDecodingError(
                        f"requested frame {source_frame_index} has no PTS time base"
                    )

                timestamp_s = float(frame.pts * frame.time_base)
                if frames and timestamp_s <= frames[-1].timestamp_s:
                    raise VideoDecodingError(
                        "requested frame timestamps must be strictly increasing"
                    )

                frames.append(
                    DecodedFrame(
                        source_frame_index=source_frame_index,
                        timestamp_s=timestamp_s,
                        rgb=frame.to_ndarray(format="rgb24"),
                    )
                )
                target_offset += 1
                if target_offset == len(targets):
                    break
    except VideoDecodingError:
        raise
    except (av.FFmpegError, OSError) as exc:
        raise VideoDecodingError(f"failed to decode local video: {path}") from exc

    if target_offset != len(targets):
        missing = targets[target_offset:]
        raise VideoDecodingError(f"requested source frames do not exist: {missing}")

    return DecodedVideoSample(
        sample_id=source.sample_id,
        source_video_id=source.source_video_id,
        frames=tuple(frames),
    )
