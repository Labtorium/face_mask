import random
import cv2
import face_recognition
import numpy as np
import os
import time

def overlay_character(frame, char_img, face_box, scale=1.5):
    top, right, bottom, left = face_box
    w, h = right - left, bottom - top

    scaled_w, scaled_h = int(w * scale), int(h * scale)
    char_resized = cv2.resize(char_img, (scaled_w, scaled_h))

    offset_x = left - (scaled_w - w) // 2
    offset_y = top - (scaled_h - h) // 2 - 60

    y1 = max(offset_y, 0)
    y2 = min(offset_y + scaled_h, frame.shape[0])
    x1 = max(offset_x, 0)
    x2 = min(offset_x + scaled_w, frame.shape[1])

    char_y1 = y1 - offset_y
    char_y2 = char_y1 + (y2 - y1)
    char_x1 = x1 - offset_x
    char_x2 = char_x1 + (x2 - x1)

    if char_resized.shape[2] == 4:
        alpha_s = char_resized[char_y1:char_y2, char_x1:char_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha_s * char_resized[char_y1:char_y2, char_x1:char_x2, c] +
                alpha_l * frame[y1:y2, x1:x2, c]
            )
    else:
        frame[y1:y2, x1:x2] = char_resized[char_y1:char_y2, char_x1:char_x2]

# キャラクター画像の読み込み
character_dir = 'boonboon'
character_images = [
    cv2.imread(os.path.join(character_dir, f), cv2.IMREAD_UNCHANGED)
    for f in sorted(os.listdir(character_dir)) if f.lower().endswith(('.png', '.jpg'))
]
if not character_images:
    raise Exception("キャラクター画像が読み込めません")

# 背景画像の読み込み
background_frame = cv2.imread(os.path.join(character_dir, 'background/background.png'), cv2.IMREAD_UNCHANGED)
if background_frame is None:
    raise Exception("背景画像が読み込めません")

# トラッキング辞書: face_id -> (encoding, char_index, last_seen_time)
tracked_faces = {}
ID_TIMEOUT = 3.0  # 秒以内は同一人物とみなす

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    current_time = time.time()
    used_char_indices = set()
    new_tracked_faces = {}

    # 過去のトラッキングから有効なものを抽出
    valid_tracked_faces = {
        fid: (enc, char_index) for fid, (enc, char_index, last_seen) in tracked_faces.items()
        if current_time - last_seen <= ID_TIMEOUT
    }

    known_ids = list(valid_tracked_faces.keys())
    known_encodings = [valid_tracked_faces[fid][0] for fid in known_ids]

    unmatched_detected = list(range(len(face_encodings)))
    matches = {}

    # 顔マッチング（距離が最も近い組み合わせでペアリング）
    if known_encodings and face_encodings:
        known_encodings_np = np.array(known_encodings)
        face_encodings_np = np.array(face_encodings)

        distance_matrix = np.linalg.norm(
            np.expand_dims(known_encodings_np, axis=1) - np.expand_dims(face_encodings_np, axis=0),
            axis=2
        )

        used_known = set()
        used_detected = set()

        while True:
            min_val = np.min(distance_matrix)
            if min_val > 0.6:
                break

            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

            if i in used_known or j in used_detected:
                distance_matrix[i, j] = np.inf
                continue

            matched_id = known_ids[i]
            matches[j] = matched_id
            used_known.add(i)
            used_detected.add(j)
            used_char_indices.add(valid_tracked_faces[matched_id][1])
            distance_matrix[i, :] = np.inf
            distance_matrix[:, j] = np.inf

        unmatched_detected = [j for j in range(len(face_encodings)) if j not in matches]

    # 新規顔にキャラを割り当て
    for j in unmatched_detected:
        enc = face_encodings[j]
        new_id = enc.tobytes().hex()

        available_indices = [i for i in range(len(character_images)) if i not in used_char_indices]
        char_index = available_indices[0] if available_indices else random.randint(0, len(character_images) - 1)

        used_char_indices.add(char_index)
        matches[j] = new_id
        new_tracked_faces[new_id] = (enc, char_index, current_time)

    # 描画とトラッキング更新
    for j, face_encoding in enumerate(face_encodings):
        face_id = matches[j]
        if face_id in tracked_faces:
            char_index = tracked_faces[face_id][1]
        elif face_id in new_tracked_faces:
            char_index = new_tracked_faces[face_id][1]
        else:
            continue

        new_tracked_faces[face_id] = (face_encoding, char_index, current_time)

        top, right, bottom, left = face_locations[j]
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        char_img = character_images[char_index]
        overlay_character(frame, char_img, (top, right, bottom, left), scale=1.5)

    # 古いデータを破棄して更新
    tracked_faces = {
        fid: data for fid, data in {**tracked_faces, **new_tracked_faces}.items()
        if current_time - data[2] <= ID_TIMEOUT
    }

    # 背景合成
    if background_frame.shape[2] == 4:
        alpha_s = background_frame[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[:, :, c] = (
                alpha_s * background_frame[:, :, c] +
                alpha_l * frame[:, :, c]
            )

    cv2.imshow('Tracked Faces with Characters', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
