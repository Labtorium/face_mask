import random
import cv2
import face_recognition
import numpy as np
import os
import uuid

def overlay_character(frame, char_img, face_box, scale=1.5):
    top, right, bottom, left = face_box
    w, h = right - left, bottom - top

    # 拡大後のサイズ
    scaled_w, scaled_h = int(w * scale), int(h * scale)
    char_resized = cv2.resize(char_img, (scaled_w, scaled_h))

    # 中心に合わせる
    offset_x = left - (scaled_w - w) // 2
    offset_y = top - (scaled_h - h) // 2 - 60

    # 合成範囲（画面外に出ないようクリップ）
    y1 = max(offset_y, 0)
    y2 = min(offset_y + scaled_h, frame.shape[0])
    x1 = max(offset_x, 0)
    x2 = min(offset_x + scaled_w, frame.shape[1])

    # キャラクター画像の使用範囲（必要なら切り取り）
    char_y1 = y1 - offset_y
    char_y2 = char_y1 + (y2 - y1)
    char_x1 = x1 - offset_x
    char_x2 = char_x1 + (x2 - x1)

    if char_resized.shape[2] == 4:
        alpha_s = char_resized[char_y1:char_y2, char_x1:char_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * char_resized[char_y1:char_y2, char_x1:char_x2, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
    else:
        frame[y1:y2, x1:x2] = char_resized[char_y1:char_y2, char_x1:char_x2]


# キャラクター画像の読み込み
character_dir = 'boonboon'
character_images = [
    cv2.imread(os.path.join(character_dir, f), cv2.IMREAD_UNCHANGED)
    for f in sorted(os.listdir(character_dir)) if f.lower().endswith(('.png', '.jpg'))
]
if not character_images:
    raise Exception("キャラクター画像が読み込めませんでした！characters フォルダに画像を入れてください。")
random.shuffle(character_images)

# 背景画像の読み込み
background_frame = cv2.imread('background/background.png', cv2.IMREAD_UNCHANGED)
if background_frame is None:
    raise Exception("背景画像が読み込めませんでした！backgroundフォルダに画像を入れてください。")


# トラッキング情報（ID: ベクトル）
known_face_encodings = []
known_face_ids = []
face_id_to_char_index = {}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 顔の位置とエンコーディング（特徴ベクトル）取得
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_id = None

        if True in matches:
            idx = matches.index(True)
            face_id = known_face_ids[idx]
        else:
            face_id = str(uuid.uuid4())  # 新規ID
            known_face_encodings.append(face_encoding)
            known_face_ids.append(face_id)

            # 使用されていないキャラ画像を優先的に割り当てる
            used_indices = set(face_id_to_char_index.values())
            available_indices = [i for i in range(len(character_images)) if i not in used_indices]
            if available_indices:
                char_index = available_indices[0]
            else:
                char_index = random.randint(0, len(character_images) - 1)

            face_id_to_char_index[face_id] = char_index

        # 元画像の座標に戻す
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        w, h = right - left, bottom - top

        char_img = character_images[face_id_to_char_index[face_id]]
        overlay_character(frame, char_img, (top, right, bottom, left), scale=1.5)

    # 背景フレーム
    alpha_s = background_frame[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[:, :, c] = (alpha_s * background_frame[:, :, c] +
                          alpha_l * frame[:, :, c])

    cv2.imshow('Tracked Faces with Characters', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()