import os
import uuid
import tempfile
import subprocess
import numpy as np
import torch
from PIL import Image
import gradio as gr
from sam2.build_sam import build_sam2_video_predictor

# ── 디버이스 설정 ───────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
config     = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor  = build_sam2_video_predictor(config, checkpoint, device=device)

# ── 세션 상태 저장 ───────────────────
SESSION_STATES = {}

# ── 비디오 로드 및 초기화 ───────────────────

def load_video_and_init(video_file):
    """업로드된 비디오를 프레임으로 분리하고 처음 프레임과 세션 ID를 반환"""
    tmp_dir = tempfile.mkdtemp()
    cmd = [
        "ffmpeg", "-i", video_file.name,
        "-q:v", "2", "-start_number", "0",
        os.path.join(tmp_dir, "%05d.jpg")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frames = sorted(
        [f for f in os.listdir(tmp_dir) if f.lower().endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    state = predictor.init_state(video_path=tmp_dir)

    base = np.array(Image.open(os.path.join(tmp_dir, frames[0])))

    session_id = str(uuid.uuid4())
    SESSION_STATES[session_id] = {
        "state": state,            # SAM2 predictor state
        "frames": frames,          # 프레임 파일명 리스트
        "tmp_dir": tmp_dir,        # 프레임이 저장된 임시 폴더
        "base": base,              # 처음 프레임 RGB 배열
        "mask_pos": np.zeros(base.shape[:2], dtype=bool),  # 차려시크 영역
        "mask_neg": np.zeros(base.shape[:2], dtype=bool),  # 빨간색 영역
        "obj_id_counter": 1,       # 객체 ID 칸터 (Positive, Negative 객체)
        "pending_points": []       # (자표, label) 쌍 목록
    }
    return Image.fromarray(base), session_id

# ── 클릭 저장 ───────────────────

def store_click(evt: gr.SelectData, label, session_id):
    """클릭 위치를 리케일(Positive/Negative)으로 pending 목록에 저장"""
    sess = SESSION_STATES[session_id]
    if label in ("Positive", "Negative"):
        lbl_val = 1 if label == "Positive" else 0
        sess["pending_points"].append((evt.index, lbl_val))
    SESSION_STATES[session_id] = sess
    return None  # 즉시 UI 변경 없음

# ── pending 포인트 적용 ───────────────────

def apply_pending_clicks(session_id):
    sess = SESSION_STATES[session_id]
    base = sess["base"]
    points = sess["pending_points"]

    if not points:
        return Image.fromarray(base)

    # 클릭들은 동일한 라벨(모드)로 입력되는다고 가정
    mode_label = points[0][1]  # 1: Positive, 0: Negative

    # SAM2는 최소 1개의 positive 포인트가 있어야 마스크 예측 가능하니 Negative 모드에서도 내부적으로는 positive(1)로 변환해 예측 후 red overlay로만 사용
    pts = np.array([pt for pt, _ in points], dtype=np.float32)
    labs = np.full(len(pts), 1, dtype=np.int32)  # 내부적으로 전부 1로 변환

    obj_id = sess["obj_id_counter"]
    _, _, out_logits = predictor.add_new_points_or_box(
        inference_state=sess["state"],
        frame_idx=0,
        obj_id=obj_id,
        points=pts,
        labels=labs,
    )

    logits = out_logits[0].cpu().numpy()
    if logits.ndim == 3:
        logits = logits[0]
    mask = logits > 0.0

    if mode_label == 1:
        sess["mask_pos"] = np.logical_or(sess["mask_pos"], mask)
    else:
        sess["mask_neg"] = np.logical_or(sess["mask_neg"], mask)

    # 다음 객체 ID 준비 & pending 초기화
    sess["obj_id_counter"] += 1
    sess["pending_points"] = []
    SESSION_STATES[session_id] = sess

    # ─ 시간화: 빨간 → 차려 순으로 오버레이 (차려는 최종 우선)
    overlay = base.copy().astype(float)
    if sess["mask_neg"].any():
        alpha_neg = sess["mask_neg"][..., None] * 0.6
        overlay = overlay * (1 - alpha_neg) + alpha_neg * np.array([255, 0, 0])
    if sess["mask_pos"].any():
        alpha_pos = sess["mask_pos"][..., None] * 0.6
        overlay = overlay * (1 - alpha_pos) + alpha_pos * np.array([0, 255, 0])

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

# ── 포인트 초기화 ───────────────────

def clear_points(session_id):
    sess = SESSION_STATES[session_id]
    # predictor state 리셋 (객체들 초기화)
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(sess["state"])
    else:
        sess["state"] = predictor.init_state(video_path=sess["tmp_dir"])

    sess.update(
        mask_pos=np.zeros(sess["base"].shape[:2], dtype=bool),
        mask_neg=np.zeros(sess["base"].shape[:2], dtype=bool),
        obj_id_counter=1,
        pending_points=[],
    )
    SESSION_STATES[session_id] = sess
    return Image.fromarray(sess["base"])

# ── 결과 전파 및 비디오 생성 ───────────────────
#  (전파 로직은 변경 없음 – 객체 ID 1(Positive) 결과만 사용)

def propagate_and_export(session_id):
    sess = SESSION_STATES[session_id]
    state = sess["state"]
    frames = sess["frames"]
    tmp_dir = sess["tmp_dir"]

    out_dir = tempfile.mkdtemp()
    for idx, _, logits in predictor.propagate_in_video(state):
        mask = (logits[0] > 0.0).cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        frame = np.array(Image.open(os.path.join(tmp_dir, frames[idx])))
        alpha = mask[..., None] * 0.6
        overlay = (frame * (1 - alpha) + alpha * np.array([0, 255, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(out_dir, f"{idx:05d}.jpg"))

    video_out = os.path.join(out_dir, "result.mp4")
    os.system(f"ffmpeg -framerate 30 -i {out_dir}/%05d.jpg -c:v libx264 -pix_fmt yuv420p {video_out}")
    return video_out

# ── Gradio UI 구성 ───────────────────

with gr.Blocks() as demo:
    gr.Markdown("## SAM2 Video Segmentation Demo")

    with gr.Row():
        video_in = gr.File(label="Upload Video (.mp4)")
        load_btn = gr.Button("Load Video")

    with gr.Row():
        with gr.Column():
            frame_disp = gr.Image(label="First Frame", interactive=True)
        with gr.Column():
            label_radio = gr.Radio(["Positive", "Negative"], value="Positive", label="Click Label")
            add_point_btn = gr.Button("Add Points")
            clear_btn = gr.Button("Clear Points")

    session_state = gr.State()
    propagate_btn = gr.Button("Propagate Segmentation")
    video_out = gr.Video(label="Segmented Video")

    load_btn.click(load_video_and_init, inputs=[video_in], outputs=[frame_disp, session_state])
    frame_disp.select(store_click, inputs=[label_radio, session_state], outputs=[])
    add_point_btn.click(apply_pending_clicks, inputs=[session_state], outputs=[frame_disp])
    clear_btn.click(clear_points, inputs=[session_state], outputs=[frame_disp])
    propagate_btn.click(propagate_and_export, inputs=[session_state], outputs=[video_out])

    demo.launch(server_name="0.0.0.0", server_port=8888, share=False)

