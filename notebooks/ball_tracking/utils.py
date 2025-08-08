from pathlib import Path
import numpy as np
from tqdm import tqdm

def concat2d(im_list, n_rows, n_cols, fill_value=0, dtype=np.uint8):
    """Concatenate a list of images into a 2D grid."""
    im_list = np.asarray(im_list)
    im = np.full((n_rows * n_cols, *im_list[0].shape), fill_value=fill_value, dtype=dtype)
    n_ims = min(len(im_list), len(im))
    im[:n_ims] = im_list[:n_ims]
    im = im.reshape((n_rows, n_cols, *im.shape[1:]))
    return np.concatenate(np.concatenate(im, axis=1), axis=1)

def uniform_sample_frames(
    video_path: str,
    n_frames: int,
    y0: int = 64,
    n_corridors: int = 6,
    corridor_height: int = 320,
):
    y1 = y0 + n_corridors * corridor_height
    video_path = Path(video_path).as_posix()

    try:
        from decord import VideoReader

        vr = VideoReader(video_path)
        n_frames_total = len(vr)
        assert n_frames <= n_frames_total, f"Requested {n_frames} frames, but video has only {n_frames_total} frames."
        frame_indices = np.linspace(0, n_frames_total, n_frames, endpoint=False)
        frame_indices = np.round(frame_indices).astype(int)
        frames =  vr.get_batch(frame_indices).asnumpy()[:, y0:y1, :, 0]
        t, _, w = frames.shape
        frames = np.ascontiguousarray(frames.reshape((t, n_corridors, corridor_height, w)))
        return frames
    except ImportError:
        import warnings
        warnings.warn(
            "Decord is not installed. Using opencv instead."
            "Note that frames may not be exact: https://github.com/opencv/opencv/issues/9053",
        )

        import cv2

        cap = cv2.VideoCapture(video_path)
        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert n_frames <= n_frames_total, f"Requested {n_frames} frames, but video has only {n_frames_total} frames."
        frame_indices = np.linspace(0, n_frames_total - 1, n_frames, endpoint=False).astype(int)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = np.zeros((n_frames, n_corridors, corridor_height, width), dtype=np.uint8)
        for j, i in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames[j] = frame[y0:y1, :, 0].reshape((n_corridors, corridor_height, width))
        cap.release()
        return frames

def iter_frames(video_path: str, y0: int = 64, n_corridors: int = 6, corridor_height: int = 320, verbose: bool = False):
    video_path = Path(video_path).as_posix()
    y1 = y0 + n_corridors * corridor_height
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        if verbose:
            vr = tqdm(vr, total=len(vr))
        for frame in vr:
            yield frame.asnumpy()[y0:y1, :, 0].reshape((n_corridors, corridor_height, -1))
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if verbose:
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            pbar = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame[y0:y1, :, 0].reshape((n_corridors, corridor_height, -1))
            if pbar is not None:
                pbar.update(1)
        cap.release()
