#!/usr/bin/env python3
"""Extract one explicitly selected local clip into a ParticleSequence artifact."""

import argparse
from pathlib import Path

from sparse3d_forgery.frontend import VggtFrontend, VggtFrontendConfig
from sparse3d_forgery.particle_sequence import save_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video, explicit_frame_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--tracks", type=int, default=128)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--history-count", type=int)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indices = explicit_frame_indices(args.frames)
    decoded = decode_video(
        VideoSource(
            sample_id=args.sample_id,
            source_video_id=args.source_video_id,
            source_locator=args.video,
        ),
        indices.tolist(),
    )
    frontend = VggtFrontend(
        args.weights,
        VggtFrontendConfig(num_tracks=args.tracks, target_size=args.target_size),
    )
    sequence = (
        frontend.extract_causal_window(decoded, args.history_count)
        if args.history_count is not None
        else frontend.extract(decoded)
    )
    arrays_path, metadata_path = save_particle_sequence(sequence, args.output)
    print(arrays_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
