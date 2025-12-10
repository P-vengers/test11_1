import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import json

# ================== 설정값 ==================
NUM_LM = 478                 # FaceMesh 정제 모드 랜드마크 개수
MIN_DEPTH_M = 0.15           # 사용 깊이 범위 (m)
MAX_DEPTH_M = 1.0
MIN_SAMPLES_PER_POINT = 5    # 랜드마크별 최소 샘플 프레임 수
Z_VIS_SCALE = 2.0            # 시각화에서 깊이 과장 배수

SCAN_MAX_FRAMES = 90         # s 누른 후 스캔 프레임 수 (30fps 기준 3초 정도)


# ================== (0) 얼굴 부위 인덱스 & 안전 마스크 ==================

EYE_L_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

EYE_R_IDX = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466
]

MOUTH_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87
]

NOSE_IDX = [
    1, 2, 98, 327, 168, 195, 5, 4, 94, 370
]

FORBIDDEN_IDX = set(EYE_L_IDX + EYE_R_IDX + MOUTH_IDX + NOSE_IDX)


def build_safety_mask(fused_mask: np.ndarray) -> np.ndarray:
    safe_mask = fused_mask.copy()
    for idx in FORBIDDEN_IDX:
        if 0 <= idx < fused_mask.shape[0]:
            safe_mask[idx] = False
    return safe_mask


# ================== (1) 단일 View 스캔 (s 한 번 → 자동 종료) ==================
def scan_one_view(view_idx: int = 0):
    print(f"\n[뷰 {view_idx}] RealSense 초기화...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("[ERROR] RealSense pipeline 시작 실패:", e)
        return None, None

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    all_samples = [[] for _ in range(NUM_LM)]

    scanning = False
    has_data = False
    frame_count = 0
    quit_flag = False

    print(f"[뷰 {view_idx}] 준비 완료.")
    print("   - 's' : 스캔 시작 (한 번 누르면 자동으로 끝남)")
    print("   - 'q' 또는 ESC : 프로그램 종료")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            h, w, _ = color_img.shape

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            disp = color_img.copy()

            if results.multi_face_landmarks:
                lmks = results.multi_face_landmarks[0].landmark

                # 2D 랜드마크 표시
                for lm in lmks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(disp, (px, py), 1, (0, 255, 0), -1)

                if scanning:
                    intr = color_frame.profile.as_video_stream_profile().intrinsics

                    for idx, lm in enumerate(lmks):
                        if idx >= NUM_LM:
                            break

                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        if not (0 <= px < w and 0 <= py < h):
                            continue

                        d = depth_img[py, px]
                        if d == 0:
                            continue

                        d_m = d * depth_scale
                        if not (MIN_DEPTH_M <= d_m <= MAX_DEPTH_M):
                            continue

                        X, Y, Z = rs.rs2_deproject_pixel_to_point(
                            intr, [px, py], d_m
                        )  # meters
                        all_samples[idx].append([X, Y, Z])
                        has_data = True

                    cv2.putText(
                        disp,
                        f"[뷰 {view_idx}] SCANNING... ({frame_count+1}/{SCAN_MAX_FRAMES})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        disp,
                        f"[뷰 {view_idx}] Press 's' to scan (auto stop)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    disp,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(f"View {view_idx}", disp)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[사용자 종료 요청]")
                quit_flag = True
                break

            if (not scanning) and key == ord("s"):
                scanning = True
                frame_count = 0
                print(f"[뷰 {view_idx}] 스캔 시작 (자동 종료 모드)")

            if scanning:
                frame_count += 1
                if frame_count >= SCAN_MAX_FRAMES:
                    print(f"[뷰 {view_idx}] 스캔 자동 종료 (프레임 한도 도달)")
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if quit_flag or not has_data:
        return None, None

    # median 기반 outlier 제거 + 평균
    mean_pts = np.zeros((NUM_LM, 3), dtype=np.float64)
    counts = np.zeros((NUM_LM,), dtype=np.int32)

    for idx, samples in enumerate(all_samples):
        if len(samples) < MIN_SAMPLES_PER_POINT:
            continue

        arr = np.asarray(samples)
        z = arr[:, 2]
        z_med = np.median(z)

        good = np.abs(z - z_med) < 0.02  # 2cm 이내
        arr_good = arr[good]
        if arr_good.size == 0:
            continue

        mean_pts[idx] = arr_good.mean(axis=0)
        counts[idx] = arr_good.shape[0]

    valid_mask = counts > 0
    if not np.any(valid_mask):
        print("[뷰] 유효 포인트가 부족합니다.")
        return None, None

    print(f"[뷰 {view_idx}] 유효 랜드마크 수: {int(valid_mask.sum())}")
    print(
        f"[뷰 {view_idx}] 랜드마크당 평균 샘플 수: {counts[valid_mask].mean():.1f}"
    )

    return mean_pts, valid_mask


# ================== (2) 얼굴 평면(PCA) & 투영 ==================
def build_face_plane(points_3d: np.ndarray, mask: np.ndarray):
    valid_pts = points_3d[mask]
    if valid_pts.shape[0] < 3:
        center = points_3d.mean(axis=0)
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        return center, u_axis, v_axis, normal

    center = valid_pts.mean(axis=0)

    X = valid_pts - center
    cov = X.T @ X / len(valid_pts)

    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    u_axis = eigvecs[:, 1]
    v_axis = eigvecs[:, 2]

    return center, u_axis, v_axis, normal


def project_to_uv(points_3d: np.ndarray, origin, u_axis, v_axis):
    rel = points_3d - origin
    u = rel @ u_axis
    v = rel @ v_axis
    return u, v


# ================== (3) Z형 지그재그 경로 ==================
def build_zigzag_path(points_3d: np.ndarray,
                      safe_mask: np.ndarray,
                      row_step_mm: float = 5.0):
    if not np.any(safe_mask):
        return []

    origin, u_axis, v_axis, _ = build_face_plane(points_3d, safe_mask)
    u, v = project_to_uv(points_3d, origin, u_axis, v_axis)

    idx_valid = np.where(safe_mask)[0]
    u_valid = u[idx_valid]
    v_valid = v[idx_valid]

    v_min, v_max = v_valid.min(), v_valid.max()
    row_step = row_step_mm / 1000.0  # mm → m

    row_ids = np.floor((v_valid - v_min) / row_step).astype(int)
    max_row = row_ids.max()

    rows = [[] for _ in range(max_row + 1)]
    for row_id, idx, u_val in zip(row_ids, idx_valid, u_valid):
        rows[row_id].append((idx, u_val))

    ordered_indices = []

    for r, row in enumerate(rows):
        if not row:
            continue
        row = sorted(row, key=lambda x: x[1])  # u 기준
        if r % 2 == 1:
            row = row[::-1]  # 홀수 줄은 반전 → 지그재그
        ordered_indices.extend([idx for idx, _ in row])

    return ordered_indices


# ================== (4) 법선(normal) 추정 ==================
def estimate_normals(points_3d: np.ndarray,
                     mask: np.ndarray,
                     k: int = 20):
    N = points_3d.shape[0]
    normals = np.zeros_like(points_3d)

    valid_idx = np.where(mask)[0]
    valid_pts = points_3d[valid_idx]

    if valid_pts.shape[0] < 3:
        return normals

    for idx in valid_idx:
        p = points_3d[idx]
        diff = valid_pts - p
        dist2 = np.sum(diff ** 2, axis=1)
        order = np.argsort(dist2)

        if len(order) <= 1:
            continue

        neighbor_count = min(k + 1, len(order))
        neighbors = valid_pts[order[1:neighbor_count]]

        if neighbors.shape[0] < 3:
            continue

        X = neighbors - neighbors.mean(axis=0)
        cov = X.T @ X / len(neighbors)
        eigvals, eigvecs = np.linalg.eigh(cov)

        n = eigvecs[:, 0]
        n = n / (np.linalg.norm(n) + 1e-8)

        normals[idx] = n

    return normals


# ================== (5) JSONL 저장 ==================
def export_path_to_jsonl(fused_pts,
                         fused_mask,
                         out_path: str = "face_path_points.jsonl",
                         row_step_mm: float = 5.0):
    safe_mask = build_safety_mask(fused_mask)

    if not np.any(safe_mask):
        print("[WARN] 안전 영역에 유효한 포인트가 없습니다. 저장 중단.")
        return

    normals = estimate_normals(fused_pts, safe_mask, k=20)
    ordered_idx = build_zigzag_path(
        fused_pts,
        safe_mask,
        row_step_mm=row_step_mm
    )

    if not ordered_idx:
        print("[WARN] Zig-zag 경로에 포인트가 없습니다. 저장 중단.")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for idx in ordered_idx:
            p = fused_pts[idx]
            n = normals[idx]

            record = {
                "lm_idx": int(idx),
                "X_m": float(p[0]),
                "Y_m": float(p[1]),
                "Z_m": float(p[2]),
                "nx": float(n[0]),
                "ny": float(n[1]),
                "nz": float(n[2]),
            }
            f.write(json.dumps(record) + "\n")

    print(f"[SAVE] {out_path} 에 {len(ordered_idx)}개 포인트 저장 완료")


# ================== (6) 시각화 (선택) ==================
def visualize_single_model(pts, mask):
    valid = mask
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xyz = pts[valid]
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2] * Z_VIS_SCALE,
        s=5,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel(f"Z (m) x{Z_VIS_SCALE}")
    ax.set_title("Fused Face Landmarks")
    plt.show()


# ================== (7) main ==================
def main():
    # 1) 단일 뷰 스캔 (캘리브레이션 때 사용한 카메라 위치에서)
    pts, mask = scan_one_view(view_idx=0)
    if pts is None or mask is None:
        print("[ERROR] 스캔 실패")
        return

    fused_pts = pts
    fused_mask = mask

    print("[INFO] 얼굴 모델 생성 완료")

    # 2) (선택) 시각화
    visualize_single_model(fused_pts, fused_mask)

    # 3) 로봇용 경로 + 법선 JSONL 저장
    export_path_to_jsonl(
        fused_pts,
        fused_mask,
        out_path="face_path_points.jsonl",
        row_step_mm=5.0,
    )


if __name__ == "__main__":
    main()
