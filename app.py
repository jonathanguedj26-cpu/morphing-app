import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

SESSION_STORE = {}


def img_to_b64(img, quality=82):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def align_to_eyes(img, src_left, src_right, dst_left, dst_right, out_wh):
    sl = np.array(src_left, dtype=np.float64)
    sr = np.array(src_right, dtype=np.float64)
    dl = np.array(dst_left, dtype=np.float64)
    dr = np.array(dst_right, dtype=np.float64)

    src_d = sr - sl
    dst_d = dr - dl

    scale = np.linalg.norm(dst_d) / (np.linalg.norm(src_d) + 1e-8)
    angle = np.arctan2(dst_d[1], dst_d[0]) - np.arctan2(src_d[1], src_d[0])

    ca = np.cos(angle) * scale
    sa = np.sin(angle) * scale

    sc = (sl + sr) / 2
    dc = (dl + dr) / 2

    tx = dc[0] - (ca * sc[0] - sa * sc[1])
    ty = dc[1] - (sa * sc[0] + ca * sc[1])

    M = np.float32([[ca, -sa, tx], [sa, ca, ty]])
    return cv2.warpAffine(img, M, out_wh, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def morph_pair(a, b, n=24):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    params = dict(pyr_scale=0.5, levels=6, winsize=25,
                  iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, **params)
    bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, **params)

    h, w = a.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    frames = []
    for i in range(n):
        t = smoothstep((i + 1) / (n + 1))
        wa = cv2.remap(af, xs + fwd[..., 0] * t,       ys + fwd[..., 1] * t,       cv2.INTER_LINEAR)
        wb = cv2.remap(bf, xs + bwd[..., 0] * (1 - t), ys + bwd[..., 1] * (1 - t), cv2.INTER_LINEAR)
        blended = ((1 - t) * wa + t * wb).clip(0, 255).astype(np.uint8)
        frames.append(blended)
    return frames


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('photos')
    if not (2 <= len(files) <= 5):
        return jsonify(error='Envoyez 2 à 5 photos'), 400

    sid = str(uuid.uuid4())
    imgs = []
    previews = []

    for f in files:
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error='Image invalide'), 400

        h, w = img.shape[:2]
        if max(h, w) > 720:
            s = 720 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)

        imgs.append(img)
        previews.append({
            'b64': img_to_b64(img, 80),
            'w': img.shape[1],
            'h': img.shape[0]
        })

    SESSION_STORE[sid] = imgs
    return jsonify(sid=sid, images=previews)


@app.route('/generate', methods=['POST'])
def generate():
    body = request.get_json()
    sid = body.get('sid')
    eyes = body.get('eyes')

    imgs = SESSION_STORE.get(sid)
    if imgs is None:
        return jsonify(error='Session expirée, rechargez la page'), 400

    n = len(imgs)
    h0, w0 = imgs[0].shape[:2]

    dl = eyes[0]['left']
    dr = eyes[0]['right']
    aligned = [imgs[0].copy()]
    for i in range(1, n):
        al = align_to_eyes(imgs[i], eyes[i]['left'], eyes[i]['right'],
                           dl, dr, (w0, h0))
        aligned.append(al)

    PAUSE, MORPH = 6, 24
    frames = []
    frame_photo = []

    for _ in range(PAUSE):
        frames.append(img_to_b64(aligned[0]))
        frame_photo.append(0.0)

    for i in range(n - 1):
        mf = morph_pair(aligned[i], aligned[i + 1], MORPH)
        for j, frame in enumerate(mf):
            frames.append(img_to_b64(frame))
            frame_photo.append(i + smoothstep((j + 1) / (MORPH + 1)))
        for _ in range(PAUSE):
            frames.append(img_to_b64(aligned[i + 1]))
            frame_photo.append(float(i + 1))

    del SESSION_STORE[sid]
    return jsonify(frames=frames, frame_photo=frame_photo, n_photos=n)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
