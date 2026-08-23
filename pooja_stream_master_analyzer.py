import csv
import os
import subprocess
import threading
import queue
import cv2
import numpy as np
import audioop

# ============================================================
# CONFIG
# ============================================================

VIDEO_URL = (
    "https://cdn.dailydarshan.live/live-recordings/"
    "9fb51944-6566-43d9-9ff3-9dd556051a2d/"
    "d9831442c449ac1e4ec44e8c122db082_"
    "9fb51944-6566-43d9-9ff3-9dd556051a2d.m3u8"
)

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "master_stream_report.csv"
)

SAMPLE_INTERVAL = 2
BRIGHTNESS_THRESHOLD = 100

STUCK_DIFF_THRESHOLD = 1.5
STUCK_MIN_DURATION = 10

AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLE_WIDTH = 2
AUDIO_CHUNK_SECONDS = 1
AUDIO_SILENCE_RMS = 500
AUDIO_MIN_SILENCE = 5

MATCH_THRESHOLD = 0.363

WIDTH = 360
HEIGHT = 640

MODEL_DIR = "models"
REFERENCE_DIR = "references"

YUNET_MODEL = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = os.path.join(
    MODEL_DIR,
    "face_recognition_sface_2021dec.onnx"
)

REFERENCE_FILES = [
    os.path.join(
        REFERENCE_DIR,
        "pandit_1.png"
    ),
    os.path.join(
        REFERENCE_DIR,
        "pandit_2.png"
    )
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def fmt(seconds):
    seconds = int(max(0, seconds))
    return (
        f"{seconds//3600:02d}:"
        f"{(seconds%3600)//60:02d}:"
        f"{seconds%60:02d}"
    )

def pct(value, total):
    return (
        value / total * 100
        if total else 0
    )

# ============================================================
# CHECK FILES
# ============================================================

for path in [
    YUNET_MODEL,
    SFACE_MODEL,
    *REFERENCE_FILES
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

# ============================================================
# LOAD FACE MODELS
# ============================================================

print("==============================================")
print("        POOJA MASTER STREAM ANALYZER")
print("==============================================")
print()
print("Features:")
print("  [1] Pandit presence")
print("  [2] Low brightness")
print("  [3] Stuck/frozen frames")
print("  [4] Low/no audio")
print()
print("Sampling video every", SAMPLE_INTERVAL, "seconds")
print("Brightness threshold:", BRIGHTNESS_THRESHOLD)
print("Stuck threshold:", STUCK_DIFF_THRESHOLD)
print("Audio RMS threshold:", AUDIO_SILENCE_RMS)
print()
print("No video/audio/images will be saved.")
print()

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (WIDTH, HEIGHT),
    0.7,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

def make_reference(path):
    image = cv2.imread(path)

    if image is None:
        raise RuntimeError(
            f"Could not read reference: {path}"
        )

    h, w = image.shape[:2]

    detector.setInputSize(
        (w, h)
    )

    _, faces = detector.detect(
        image
    )

    if faces is None or len(faces) == 0:
        raise RuntimeError(
            f"No face detected in {path}"
        )

    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    aligned = recognizer.alignCrop(
        image,
        face
    )

    return recognizer.feature(
        aligned
    )

references = [
    make_reference(path)
    for path in REFERENCE_FILES
]

print(
    "Loaded",
    len(references),
    "Pandit reference faces."
)

# ============================================================
# AUDIO WORKER
# ============================================================

audio_results = []
audio_done = threading.Event()
audio_error = []

def run_audio():

    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                VIDEO_URL,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-f",
                "s16le",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        chunk_size = (
            AUDIO_SAMPLE_RATE
            * AUDIO_SAMPLE_WIDTH
            * AUDIO_CHUNK_SECONDS
        )

        t = 0.0

        while True:

            data = process.stdout.read(
                chunk_size
            )

            if not data:
                break

            seconds = (
                len(data)
                / (
                    AUDIO_SAMPLE_RATE
                    * AUDIO_SAMPLE_WIDTH
                )
            )

            rms = audioop.rms(
                data,
                AUDIO_SAMPLE_WIDTH
            )

            audio_results.append(
                {
                    "start": t,
                    "end": t + seconds,
                    "rms": rms,
                    "silent":
                        rms < AUDIO_SILENCE_RMS
                }
            )

            t += seconds

        process.stdout.close()
        process.wait()

    except Exception as exc:
        audio_error.append(
            str(exc)
        )

    finally:
        audio_done.set()

# Start audio analysis in parallel with video analysis.
audio_thread = threading.Thread(
    target=run_audio,
    daemon=True
)

audio_thread.start()

# ============================================================
# VIDEO ANALYSIS
# ============================================================

video_process = subprocess.Popen(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        VIDEO_URL,
        "-vf",
        (
            f"fps=1/{SAMPLE_INTERVAL},"
            f"scale={WIDTH}:{HEIGHT}"
        ),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

buffer = b""
frame_number = 0

previous_gray = None

video_samples = []

pandit_present_seconds = 0
low_brightness_seconds = 0

pandit_absent_start = None
brightness_start = None

pandit_events = []
brightness_events = []

stuck_candidate_start = None
stuck_start = None
stuck_events = []

while True:

    chunk = video_process.stdout.read(
        65536
    )

    if not chunk:
        break

    buffer += chunk

    while True:

        start = buffer.find(
            b"\xff\xd8"
        )

        end = buffer.find(
            b"\xff\xd9"
        )

        if start == -1 or end == -1:
            break

        jpeg = buffer[
            start:end + 2
        ]

        buffer = buffer[
            end + 2:
        ]

        image = cv2.imdecode(
            np.frombuffer(
                jpeg,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:
            continue

        t = (
            frame_number
            * SAMPLE_INTERVAL
        )

        # ----------------------------------------------------
        # BRIGHTNESS
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            gray.mean()
        )

        is_low = (
            brightness
            < BRIGHTNESS_THRESHOLD
        )

        if is_low:

            low_brightness_seconds += (
                SAMPLE_INTERVAL
            )

            if brightness_start is None:
                brightness_start = t

        else:

            if brightness_start is not None:

                brightness_events.append(
                    (
                        brightness_start,
                        t
                    )
                )

                brightness_start = None

        # ----------------------------------------------------
        # STUCK FRAME
        # ----------------------------------------------------

        frame_diff = None
        similar = False

        if previous_gray is not None:

            frame_diff = float(
                cv2.absdiff(
                    gray,
                    previous_gray
                ).mean()
            )

            similar = (
                frame_diff
                < STUCK_DIFF_THRESHOLD
            )

        if similar:

            if stuck_candidate_start is None:
                stuck_candidate_start = (
                    t - SAMPLE_INTERVAL
                )

            duration = (
                t
                + SAMPLE_INTERVAL
                - stuck_candidate_start
            )

            if (
                duration
                >= STUCK_MIN_DURATION
                and stuck_start is None
            ):
                stuck_start = (
                    stuck_candidate_start
                )

        else:

            if stuck_start is not None:

                stuck_events.append(
                    (
                        stuck_start,
                        t
                    )
                )

                stuck_start = None

            stuck_candidate_start = None

        previous_gray = gray.copy()

        # ----------------------------------------------------
        # PANDIT
        # ----------------------------------------------------

        h, w = image.shape[:2]

        detector.setInputSize(
            (w, h)
        )

        _, faces = detector.detect(
            image
        )

        present = False
        best_score = 0.0
        face_count = (
            0
            if faces is None
            else len(faces)
        )

        if faces is not None:

            for face in faces:

                try:

                    aligned = (
                        recognizer.alignCrop(
                            image,
                            face
                        )
                    )

                    feature = (
                        recognizer.feature(
                            aligned
                        )
                    )

                    scores = [
                        float(
                            recognizer.match(
                                feature,
                                reference,
                                cv2.FaceRecognizerSF_FR_COSINE
                            )
                        )
                        for reference
                        in references
                    ]

                    score = max(
                        scores
                    )

                    best_score = max(
                        best_score,
                        score
                    )

                    if (
                        score
                        >= MATCH_THRESHOLD
                    ):
                        present = True

                except Exception:
                    pass

        if present:

            pandit_present_seconds += (
                SAMPLE_INTERVAL
            )

            if pandit_absent_start is not None:

                pandit_events.append(
                    (
                        pandit_absent_start,
                        t
                    )
                )

                pandit_absent_start = None

        else:

            if pandit_absent_start is None:
                pandit_absent_start = t

        video_samples.append(
            [
                fmt(t),
                t,
                round(
                    brightness,
                    2
                ),
                (
                    "LOW"
                    if is_low
                    else "NORMAL"
                ),
                (
                    "PRESENT"
                    if present
                    else "ABSENT"
                ),
                round(
                    best_score,
                    4
                ),
                face_count,
                (
                    ""
                    if frame_diff is None
                    else round(
                        frame_diff,
                        4
                    )
                ),
                (
                    "YES"
                    if similar
                    else "NO"
                )
            ]
        )

        frame_number += 1

        if frame_number % 30 == 0:

            print(
                f"Processed {fmt(t)} | "
                f"Brightness {brightness:.1f} | "
                f"Pandit "
                f"{'PRESENT' if present else 'ABSENT'} | "
                f"Match {best_score:.3f}"
            )

video_process.stdout.close()
video_process.wait()

total_duration = (
    frame_number
    * SAMPLE_INTERVAL
)

# Close final video events.
if pandit_absent_start is not None:

    pandit_events.append(
        (
            pandit_absent_start,
            total_duration
        )
    )

if brightness_start is not None:

    brightness_events.append(
        (
            brightness_start,
            total_duration
        )
    )

if stuck_start is not None:

    stuck_events.append(
        (
            stuck_start,
            total_duration
        )
    )

# ============================================================
# AUDIO EVENTS
# ============================================================

audio_thread.join()

if audio_error:

    print(
        "WARNING: Audio analysis error:",
        audio_error[0]
    )

audio_silence_events = []

audio_silence_start = None
audio_silence_total = 0

for item in audio_results:

    if item["silent"]:

        audio_silence_total += (
            item["end"]
            - item["start"]
        )

        if audio_silence_start is None:
            audio_silence_start = (
                item["start"]
            )

    else:

        if audio_silence_start is not None:

            end = item["start"]

            if (
                end
                - audio_silence_start
                >= AUDIO_MIN_SILENCE
            ):
                audio_silence_events.append(
                    (
                        audio_silence_start,
                        end
                    )
                )

            audio_silence_start = None

if audio_silence_start is not None:

    end = (
        audio_results[-1]["end"]
        if audio_results
        else total_duration
    )

    if (
        end
        - audio_silence_start
        >= AUDIO_MIN_SILENCE
    ):
        audio_silence_events.append(
            (
                audio_silence_start,
                end
            )
        )

# ============================================================
# SUMMARY
# ============================================================

pandit_present_seconds = min(
    pandit_present_seconds,
    total_duration
)

pandit_absent_seconds = max(
    0,
    total_duration
    - pandit_present_seconds
)

stuck_seconds = sum(
    end - start
    for start, end
    in stuck_events
)

audio_silence_seconds = sum(
    end - start
    for start, end
    in audio_silence_events
)

pandit_present_percentage = pct(
    pandit_present_seconds,
    total_duration
)

pandit_absent_percentage = pct(
    pandit_absent_seconds,
    total_duration
)

brightness_percentage = pct(
    low_brightness_seconds,
    total_duration
)

stuck_percentage = pct(
    stuck_seconds,
    total_duration
)

audio_percentage = pct(
    audio_silence_seconds,
    total_duration
)

# ============================================================
# WRITE MASTER CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["MASTER STREAM SUMMARY"]
    )

    writer.writerow([
        "Total video duration",
        fmt(total_duration)
    ])

    writer.writerow([
        "Samples analyzed",
        frame_number
    ])

    writer.writerow([])

    # PANDIT
    writer.writerow(
        ["PANDIT SUMMARY"]
    )

    writer.writerow([
        "Present time",
        fmt(pandit_present_seconds)
    ])

    writer.writerow([
        "Present percentage",
        f"{pandit_present_percentage:.2f}%"
    ])

    writer.writerow([
        "Absent time",
        fmt(pandit_absent_seconds)
    ])

    writer.writerow([
        "Absent percentage",
        f"{pandit_absent_percentage:.2f}%"
    ])

    writer.writerow([
        "Absence incidents",
        len(pandit_events)
    ])

    writer.writerow([])

    # BRIGHTNESS
    writer.writerow(
        ["BRIGHTNESS SUMMARY"]
    )

    writer.writerow([
        "Threshold",
        "< 100"
    ])

    writer.writerow([
        "Low brightness time",
        fmt(low_brightness_seconds)
    ])

    writer.writerow([
        "Low brightness percentage",
        f"{brightness_percentage:.2f}%"
    ])

    writer.writerow([
        "Low brightness incidents",
        len(brightness_events)
    ])

    writer.writerow([])

    # STUCK
    writer.writerow(
        ["STUCK FRAME SUMMARY"]
    )

    writer.writerow([
        "Minimum duration",
        f"{STUCK_MIN_DURATION} seconds"
    ])

    writer.writerow([
        "Stuck time",
        fmt(stuck_seconds)
    ])

    writer.writerow([
        "Stuck percentage",
        f"{stuck_percentage:.2f}%"
    ])

    writer.writerow([
        "Stuck incidents",
        len(stuck_events)
    ])

    writer.writerow([])

    # AUDIO
    writer.writerow(
        ["AUDIO SUMMARY"]
    )

    writer.writerow([
        "RMS silence threshold",
        AUDIO_SILENCE_RMS
    ])

    writer.writerow([
        "Minimum silence duration",
        f"{AUDIO_MIN_SILENCE} seconds"
    ])

    writer.writerow([
        "Low/no audio time",
        fmt(audio_silence_seconds)
    ])

    writer.writerow([
        "Low/no audio percentage",
        f"{audio_percentage:.2f}%"
    ])

    writer.writerow([
        "Low/no audio incidents",
        len(audio_silence_events)
    ])

    writer.writerow([])

    # EVENTS
    writer.writerow(
        ["PANDIT ABSENCE PERIODS"]
    )

    writer.writerow([
        "start",
        "end",
        "duration_seconds",
        "duration"
    ])

    for start, end in pandit_events:

        writer.writerow([
            fmt(start),
            fmt(end),
            round(
                end - start,
                2
            ),
            fmt(
                end - start
            )
        ])

    writer.writerow([])

    writer.writerow(
        ["LOW BRIGHTNESS PERIODS"]
    )

    writer.writerow([
        "start",
        "end",
        "duration_seconds",
        "duration"
    ])

    for start, end in brightness_events:

        writer.writerow([
            fmt(start),
            fmt(end),
            round(
                end - start,
                2
            ),
            fmt(
                end - start
            )
        ])

    writer.writerow([])

    writer.writerow(
        ["STUCK FRAME PERIODS"]
    )

    writer.writerow([
        "start",
        "end",
        "duration_seconds",
        "duration"
    ])

    for start, end in stuck_events:

        writer.writerow([
            fmt(start),
            fmt(end),
            round(
                end - start,
                2
            ),
            fmt(
                end - start
            )
        ])

    writer.writerow([])

    writer.writerow(
        ["LOW AUDIO PERIODS"]
    )

    writer.writerow([
        "start",
        "end",
        "duration_seconds",
        "duration"
    ])

    for start, end in audio_silence_events:

        writer.writerow([
            fmt(start),
            fmt(end),
            round(
                end - start,
                2
            ),
            fmt(
                end - start
            )
        ])

    writer.writerow([])

    # SAMPLE DATA
    writer.writerow(
        ["VIDEO SAMPLE DATA"]
    )

    writer.writerow([
        "timestamp",
        "timestamp_seconds",
        "brightness",
        "brightness_status",
        "pandit_status",
        "pandit_confidence",
        "faces_detected",
        "frame_difference",
        "frame_similar_to_previous"
    ])

    writer.writerows(
        video_samples
    )

    writer.writerow([])

    writer.writerow(
        ["AUDIO SAMPLE DATA"]
    )

    writer.writerow([
        "start",
        "end",
        "rms",
        "status"
    ])

    for item in audio_results:

        writer.writerow([
            fmt(item["start"]),
            fmt(item["end"]),
            round(
                item["rms"],
                2
            ),
            (
                "SILENCE"
                if item["silent"]
                else "AUDIO"
            )
        ])

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==============================================")
print("          MASTER ANALYSIS COMPLETE")
print("==============================================")
print()
print(
    "Total duration:",
    fmt(total_duration)
)

print()
print("PANDIT")
print(
    "Present:",
    fmt(pandit_present_seconds),
    f"({pandit_present_percentage:.2f}%)"
)
print(
    "Absent:",
    fmt(pandit_absent_seconds),
    f"({pandit_absent_percentage:.2f}%)"
)
print(
    "Incidents:",
    len(pandit_events)
)

print()
print("BRIGHTNESS")
print(
    "Low brightness:",
    fmt(low_brightness_seconds),
    f"({brightness_percentage:.2f}%)"
)
print(
    "Incidents:",
    len(brightness_events)
)

print()
print("STUCK FRAMES")
print(
    "Stuck:",
    fmt(stuck_seconds),
    f"({stuck_percentage:.2f}%)"
)
print(
    "Incidents:",
    len(stuck_events)
)

print()
print("AUDIO")
print(
    "Low/no audio:",
    fmt(audio_silence_seconds),
    f"({audio_percentage:.2f}%)"
)
print(
    "Incidents:",
    len(audio_silence_events)
)

print()
print("CSV saved to:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)
print()
print("==============================================")
