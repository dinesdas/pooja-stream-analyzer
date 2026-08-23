import re
import sys
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ANALYZER = APP_DIR / "pooja_stream_master_analyzer.py"
OUTPUT = APP_DIR / "output" / "master_stream_report.csv"

st.set_page_config(page_title="Master QC", page_icon="🛕", layout="wide")
st.title("🛕 Master Stream QC")
st.caption("Pandit • Brightness • Stuck Frames • Low/No Audio")

url = st.text_area("M3U8 Stream URL", placeholder="Paste M3U8 URL here...", height=100)

def make_temp_analyzer(stream_url):
    source = ANALYZER.read_text(encoding="utf-8")
    pattern = r'VIDEO_URL\s*=\s*(?:\(\s*)?(?:"[^"]*"\s*)+\)?'
    replacement = "VIDEO_URL = " + repr(stream_url)
    updated, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        raise RuntimeError("Could not find VIDEO_URL in master analyzer.")
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="_master_gui_",
        dir=str(APP_DIR), delete=False, encoding="utf-8"
    )
    f.write(updated)
    f.close()
    return Path(f.name)

if st.button("▶ Run Master Analysis", type="primary", use_container_width=True):
    if not url.strip():
        st.error("Please paste an M3U8 URL.")
    elif not ANALYZER.exists():
        st.error("pooja_stream_master_analyzer.py not found.")
    else:
        temp = None
        try:
            temp = make_temp_analyzer(url.strip())
            st.info("Running Master Analysis. Long streams may take time.")
            process = subprocess.Popen(
                [sys.executable, "-W", "ignore::DeprecationWarning", "-u", str(temp)],
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            lines = []
            box = st.empty()
            for line in process.stdout:
                if line.strip():
                    lines.append(line.rstrip())
                    box.code("\n".join(lines[-40:]), language="text")
            rc = process.wait()
            if rc == 0:
                st.success("✅ Master analysis complete.")
            else:
                st.error("❌ Master analysis failed.")
        except Exception as exc:
            st.exception(exc)
        finally:
            if temp and temp.exists():
                temp.unlink()

if OUTPUT.exists():
    st.markdown("---")
    st.subheader("Master Report")
    try:
        raw = pd.read_csv(OUTPUT, header=None, dtype=str, keep_default_na=False)
        summary = {}
        for _, row in raw.iterrows():
            if len(row) >= 2 and row.iloc[0]:
                summary[str(row.iloc[0])] = str(row.iloc[1])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", summary.get("Total video duration", "—"))
        c2.metric("Pandit Present", summary.get("Present percentage", "—"))
        c3.metric("Low Brightness", summary.get("Low brightness percentage", "—"))
        c4.metric("Stuck Frames", summary.get("Stuck percentage", "—"))

        c5, c6 = st.columns(2)
        c5.metric("Low/No Audio", summary.get("Low/no audio percentage", "—"))
        c6.metric("Pandit Absent", summary.get("Absent percentage", "—"))
    except Exception:
        st.info("Report created; summary fields could not be parsed.")

    st.download_button(
        "⬇ Download Master CSV",
        OUTPUT.read_bytes(),
        file_name="master_stream_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Run the analysis to generate the Master CSV.")
