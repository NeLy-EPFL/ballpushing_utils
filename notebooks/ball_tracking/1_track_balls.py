from pathlib import Path
from tqdm import tqdm

serial_number_to_camera = {
    "17475185": "bottom",
    "17475187": "top",
}
corridor_height = 320
n_corridors = 6
y0 = 64
y1 = y0 + n_corridors * corridor_height
video_paths = sorted(Path('/mnt/upramdya/data/TL/ball_pushing/').glob("*/2025*-*-1747518*.mp4"))

def track_balls(video_path):
    import cv2
    import numpy as np
    import pandas as pd

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"{video_path.stem}.parquet"
    if save_path.exists():
        print(f"Skipping {video_path} as {save_path} already exists.")
        return

    assert video_path.exists(), f"Video file {video_path} does not exist."

    camera = serial_number_to_camera[video_path.stem.split('-')[-1]]
    template_path = f"data/ball_template_{camera}.png"
    assert Path(template_path).exists(), f"Template image {template_path} does not exist."
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(video_path.as_posix())
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_data = []

    i_frame = 0
    x_offset = template.shape[1] // 2
    y_offset = template.shape[0] // 2

    pbar = tqdm(total=n_frames, desc="Processing frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames = frame[y0:y1, :, 0].reshape((n_corridors, corridor_height, -1))

        for i_corridor in range(n_corridors):
            score_map = cv2.matchTemplate(frames[i_corridor], template, cv2.TM_CCOEFF_NORMED)
            ymax, xmax = np.unravel_index(np.argmax(score_map), score_map.shape)
            x = xmax + x_offset
            y = ymax + y_offset + y0 + i_corridor * corridor_height
            ball_data.append((i_corridor, i_frame, x, y, score_map[ymax, xmax]))

        pbar.update(1)
        i_frame += 1

        if i_frame % 2896 == 0:
            print(video_path, i_frame / n_frames)

    cap.release()

    df = pd.DataFrame(ball_data, columns=['corridor', 'frame', 'x', 'y', 'score'])
    df.sort_values(by=['corridor', 'frame'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(save_path, index=False)


from joblib import Parallel, delayed
from tqdm import tqdm

Parallel(n_jobs=4, verbose=10)(
    delayed(track_balls)(video_path)
    for video_path in tqdm(video_paths)
)

