from pathlib import Path
import tempfile

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from camera import SignCamera
from model import Predictor, decide
from vision import extract_clip

st.set_page_config(page_title="ASL word recognizer", page_icon="👋", layout="centered")


@st.cache_resource
def load_predictor():
    return Predictor()


def ice_config():
    # Optional TURN credentials are supplied via Streamlit Secrets, not source code.
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        turn_url = st.secrets.get("TURN_URL")
    except StreamlitSecretNotFoundError:
        turn_url = None
    if turn_url:
        servers.append({"urls": [turn_url],
                        "username": st.secrets["TURN_USERNAME"],
                        "credential": st.secrets["TURN_CREDENTIAL"]})
    return {"iceServers": servers}


def render_result(ranked, threshold, margin):
    if not ranked:
        return
    word = decide(ranked, threshold, margin)
    if word == "Not recognized":
        st.warning("Not recognized with the current thresholds. Try again.")
    else:
        st.success(f"Predicted English word: **{word}**")
    st.dataframe([{"Word": r["word"], "Model score": round(r["score"], 4)}
                  for r in ranked], hide_index=True, use_container_width=True)
    st.caption("Scores are uncalibrated softmax outputs, not verified probabilities of correctness.")


st.title("👋 ASL word recognizer")
st.write("Recognize one sign at a time using a pretrained 100-word ASL model. No teaching or training needed.")
st.caption("Experimental word recognition. Continuous ASL sentences and unsupported signs may be misrecognized.")

with st.spinner("Loading pretrained model…"):
    try:
        predictor = load_predictor()
    except Exception as exc:
        st.error(f"Could not load pretrained model: {exc}")
        st.stop()

with st.sidebar:
    st.header("Recognition settings")
    threshold = st.slider("Minimum model score", 0.0, 1.0, 0.70, 0.05)
    margin = st.slider("Minimum gap between first and second choice", 0.0, 0.9, 0.15, 0.05)
    st.caption("These are heuristic rejection settings. Unknown gestures can still receive high scores.")
    with st.expander("All 100 supported words", expanded=True):
        st.write(", ".join(sorted(predictor.labels)))
    st.markdown("[Pretrained model and source](https://github.com/Abel1712/ASL-Recognition)")

mode = st.radio("Input", ["Webcam", "Upload video"], horizontal=True)
if mode == "Webcam":
    st.write("Start the camera, show your upper body and both hands, then click **Capture one sign**. "
             "There is a 1.5-second countdown, followed by 30 processed frames.")
    flip = st.checkbox("Flip camera pixels before recognition", value=False,
                       help="Training videos are unflipped; upstream webcam demo flips. Compare both orientations if needed.")
    ctx = webrtc_streamer(
        key="asl-camera", mode=WebRtcMode.SENDRECV,
        rtc_configuration=ice_config(),
        media_stream_constraints={"video": {"width": {"ideal": 640},
                                             "height": {"ideal": 480},
                                             "frameRate": {"ideal": 25, "max": 25}},
                                  "audio": False},
        video_processor_factory=lambda: SignCamera(predictor),
        async_processing=False,
    )
    if st.button("Capture one sign", type="primary", disabled=not ctx.state.playing):
        if ctx.video_processor:
            ctx.video_processor.capture(flip)

    @st.fragment(run_every=0.5)
    def live_result():
        if ctx.video_processor:
            status, ranked, count = ctx.video_processor.snapshot()
            st.info(status)
            st.progress(min(count / 30, 1.0))
            render_result(ranked, threshold, margin)
    live_result()
    st.caption("If the camera stays at Connecting, try Upload video. Hosted WebRTC may need a TURN relay; "
               "see README for configuration. Streaming frames are sent to the app server.")
else:
    st.write("Upload a clip of one sign. Pick the time where the sign begins; the model uses the next "
             "30 frames (about 1–1.2 seconds at 25–30 FPS), matching its training pipeline.")
    upload = st.file_uploader("Video (up to 25 MB)", type=["mp4", "webm", "mov"])
    start = st.number_input("Sign start time in seconds", min_value=0.0, max_value=60.0,
                            value=0.0, step=0.1)
    flip_upload = st.checkbox("Flip uploaded video pixels", value=False)
    if st.button("Recognize uploaded sign", type="primary", disabled=upload is None):
        st.session_state.pop("upload_result", None)
        if upload.size > 25 * 1024 * 1024:
            st.error("Choose a file smaller than 25 MB.")
        else:
            with st.spinner("Tracking hands and recognizing sign…"):
                try:
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / ("clip" + Path(upload.name).suffix.lower())
                        path.write_bytes(upload.getvalue())
                        sequence, preview, quality = extract_clip(path, start, flip_upload)
                        ranked = predictor.predict(sequence)
                        st.session_state.upload_result = (ranked, preview, quality)
                except Exception as exc:
                    st.error(str(exc))
    if "upload_result" in st.session_state:
        ranked, preview, quality = st.session_state.upload_result
        render_result(ranked, threshold, margin)
        st.image(preview, channels="BGR", caption=f"Tracking preview ({quality:.0%} of frames tracked)")

st.caption("Model: Abel1712/ASL-Recognition, WLASL100 LSTM–GRU, pinned revision 2e42f11. "
           "No live accuracy claim has been independently verified. Uploaded clips are processed in "
           "temporary server files, removed after processing; no training data is collected.")
