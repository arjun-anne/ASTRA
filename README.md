# Pretrained ASL word recognizer — Streamlit

This replaces the earlier teach-your-own-examples prototype. It ships **actual
pretrained weights and their exact 100-word mapping**. You do not collect examples,
train a model, download WLASL, or use a paid inference API.

## Run

Use **Python 3.11**. Newer Python 3.13/3.14 cannot install this pinned MediaPipe release.

```bash
cd /Users/atul/Data/Students/Arjun/asl-streamlit-pretrained
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 in Chrome and allow the camera. If using uv instead:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match
.venv/bin/streamlit run streamlit_app.py
```

## Test without training

1. Check **All 100 supported words** in the sidebar. Start with signs you know,
   for example yes, no, help, need, eat, drink. Have an ASL-fluent person verify
   unfamiliar signs. This app does not teach how to perform each sign.
2. Start the webcam. Show your face, upper body, and both hands, facing the camera.
3. Click **Capture one sign**, wait for the 1.5-second countdown, then sign once.
4. Capture collects 30 processed frames. Watch its progress; a busy CPU may make
   this take longer than a second. Review the predicted label and alternatives.
5. If streaming fails, record a short video elsewhere, choose **Upload video**,
   and set the sign start time. The next 30 consecutive frames are used.

The training extractor uses raw xyz coordinates for MediaPipe pose points 0–24,
then 21 left-hand points, then 21 right-hand points: 201 values per frame. Missing
groups are zeros; short clips are zero-padded. We preserve that layout and load
weights strictly. No undocumented shoulder normalization or arbitrary keypoint
mapping is used. Uploaded videos follow the upstream first-30-frame preparation.
Live webcam capture adapts that to 30 received/processed frames; actual rate may
differ from training. Upstream training is unflipped, whereas its webcam demo
flips input, so this app defaults to unflipped and exposes a flip option.

This is **isolated-word recognition**, not continuous ASL translation. It has no
trained unknown/no-sign class. Confidence/gap thresholds and a tracking-quality
gate reduce some bad results but cannot guarantee rejection of unsupported signs.
Model scores are uncalibrated. A confident wrong prediction remains possible.
Do not interpret the upstream reported accuracy as measured live app accuracy:
its published split code splits augmented samples, so independence of signers and
original clips in validation is not established. Independent signer-held-out and
webcam evaluation is needed. No production accuracy claim is made here.

## Deploy to Streamlit Community Cloud

1. Put the **contents of this folder** at the root of a GitHub repository,
   including `models/`, `THIRD_PARTY_LICENSE.txt`, `requirements.txt`,
   `packages.txt`, and `.streamlit/config.toml`. Do not upload `.venv/` or secrets.
2. In https://share.streamlit.io create an app from the repository.
3. Set the entrypoint to **streamlit_app.py** and select **Python 3.11** in
   Advanced settings, then deploy. No model token/API key is required.
4. First test **Upload video**, then webcam access. Runtime/cloud deployment
   should be verified on the actual account; this project is not predeployed.

CPU inference is sufficient for this small model (~1.4 MB weights). Runtime
libraries and landmark detection consume substantially more memory/CPU than the
weights. The Linux requirements select the CPU-only PyTorch wheel. Shared model
loading is cached, access is locked, video is capped at 640px width, audio is off,
and inference runs once per user-initiated capture. Shared hosting is appropriate
for a low-traffic demo; simultaneous video sessions can exceed resource limits.

### Webcam networking / TURN

`st.camera_input` captures pictures, so this app uses `streamlit-webrtc` for video.
Localhost and Community Cloud HTTPS support camera permissions. The default free
Google STUN endpoint may work for some networks; on Community Cloud or restrictive
networks a **TURN relay may be required**. Free hosting does not guarantee a free,
reliable video relay. Upload mode avoids WebRTC and needs no relay credentials.

If you have a TURN provider, enter the following in Community Cloud's **Secrets**
(or local `.streamlit/secrets.toml`). Never commit real credentials:

```toml
TURN_URL = "turns:your-relay.example:443?transport=tcp"
TURN_USERNAME = "your-username"
TURN_CREDENTIAL = "your-credential"
```

The app passes these ICE credentials to the browser as WebRTC requires. Use
short-lived credentials or an appropriately limited account. This basic example
does not automatically refresh expiring TURN credentials. Consider a provider's
free allowance for a demo; allowance and eligibility are provider-dependent.

### Privacy

Camera frames and uploaded clips go to the app server for processing. The app
does not deliberately persist recordings. Uploaded video uses a temporary folder
that is removed after inference; results/previews live in the browser's server
session. WebRTC buffers and detector state are per camera processor. There are
no training-data collection or external inference calls.

## Model provenance

- Model repository: https://github.com/Abel1712/ASL-Recognition
- Pinned revision: `2e42f117b322d6910f2846693bd7efd751036511`
- Checkpoint: `models/asl_model_best.pth`; 100-class LSTM–GRU, trained on WLASL100.
- Labels: `models/idx_to_word.json`, preserved verbatim from the same revision.
- Inference architecture/feature extraction adapted from `src/app.py`,
  `src/train_model_pytorch.py`, and `src/extract_keypoints.py`.
- Upstream MIT notice preserved in `THIRD_PARTY_LICENSE.txt`.
- SHA-256 for both model artifacts is validated in `model.py` before loading;
  `torch.load(..., weights_only=True)` and strict state-dict matching are used.

Selected supported everyday words: yes, no, help, want, need, eat, drink, work,
school, family, mother, doctor, medicine, book, chair, table, bed, go, give, what,
who, how, time, now, later. **Hello, please, thank you, bathroom, and water are not
classes in this checkpoint.** Expanding the vocabulary requires another trained
checkpoint or fine-tuning; it cannot be done by adding label names.

Dataset terms remain separate from software licensing; see the WLASL project's
terms before dataset reuse or wider deployment: https://github.com/dxli94/WLASL.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests validate feature layout, zero padding, real checkpoint loading, inference,
and rejection logic, plus blank-video rejection and Streamlit UI startup. All five
tests passed locally on macOS/Python 3.11 on November 14, 2026. The UI test mocks
only WebRTC's server-session introspection because AppTest has no live session
manager; it does not establish webcam transport. These checks do not measure ASL
accuracy. Actual camera recognition and Community Cloud deployment require
end-to-end testing in those environments.

References:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input
- https://github.com/whitphx/streamlit-webrtc#deploy-to-the-cloud
