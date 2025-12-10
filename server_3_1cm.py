import socket
import numpy as np
import json
import math

# =========================================================
# [STEP 1] cali_3.py 에서 출력된 3x4 행렬을 여기에 복붙하세요!
#   - 단위: mm
#   - Robot(mm) = TRANSFORMATION_MATRIX @ [Cam(mm); 1]
# =========================================================
TRANSFORMATION_MATRIX = np.array([
    [0.01665455, 0.97822465, 0.02832482, 377.40874423],
    [0.99141691, -0.02245248, -0.00914220, 8.52529268],
    [-0.01228417, 0.00438147, -0.97705115, 390.87641972],
])
# =========================================================

HOST = "0.0.0.0"
PORT = 200

PATH_FILE = "face_path_points_10mm.jsonl"


# ---------------------------------------------------------
# JSONL 파일에서 1줄씩 읽기 위한 제너레이터
# ---------------------------------------------------------
def jsonl_reader(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------
# 카메라(m) → 로봇(mm) 변환 + normal 회전
# ---------------------------------------------------------
def transform_point_and_normal(cam_xyz_m, n_cam):
    """
    cam_xyz_m : (3,) in meters
    n_cam     : (3,) unit vector in camera frame
    return    : (p_robot_mm(3,), n_robot(3,))
    """
    R = TRANSFORMATION_MATRIX[:, :3]   # (3x3)
    t = TRANSFORMATION_MATRIX[:, 3]    # (3,)

    p_cam_mm = np.array(cam_xyz_m, dtype=float) * 1000.0
    p_robot = R @ p_cam_mm + t

    n_cam = np.array(n_cam, dtype=float)
    n_robot = R @ n_cam
    n_norm = np.linalg.norm(n_robot)
    if n_norm < 1e-8:
        n_robot = np.array([0.0, 0.0, 1.0])
    else:
        n_robot = n_robot / n_norm

    return p_robot, n_robot


def build_rot_from_normal(n):
    """
    n : (3,) unit vector (로봇 좌표계 기준, 툴 Z축 방향)
    회전행렬 R을 구성. R의 3번째 열이 n.
    """
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-8)

    # 기준 up 벡터 선택 (Z축과 너무 평행이면 Y축 사용)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])

    x = np.cross(up, n)
    x = x / (np.linalg.norm(x) + 1e-8)

    y = np.cross(n, x)
    y = y / (np.linalg.norm(y) + 1e-8)

    R = np.column_stack([x, y, n])  # [x y z] 컬럼
    return R


def euler_zyz_from_rotm(R):
    """
    Doosan이 사용하는 ZYZ 오일러(A,B,C) 추출
    R = Rz(A) * Ry(B) * Rz(C)
    return: (A,B,C) in radians
    """
    r = R
    cB = max(min(r[2, 2], 1.0), -1.0)
    B = math.acos(cB)
    sB = math.sin(B)

    if abs(sB) < 1e-8:
        # 특이점: B ≈ 0 또는 π
        # 간단하게 A=0, C는 XY 평면 회전으로 처리
        A = 0.0
        C = math.atan2(r[1, 0], r[0, 0])
    else:
        A = math.atan2(r[1, 2], r[0, 2])
        C = math.atan2(r[2, 1], -r[2, 0])

    return A, B, C


# ---------------------------------------------------------
# 서버 메인 루프
# ---------------------------------------------------------
def start_server():
    data_iter = jsonl_reader(PATH_FILE)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("\n=========================================")
    print(f"[SERVER] 얼굴 경로 전송 서버 시작 (PORT: {PORT})")
    print(f"[SERVER] 경로 파일: {PATH_FILE}")
    print("[INFO] 로봇이 'shot'을 보내면 JSONL에서 다음 점을 읽어 전송합니다.")
    print("=========================================\n")

    conn, addr = server.accept()
    print(f"[SERVER] 로봇 접속됨 → {addr}")

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break

            msg = data.decode().strip()
            if not msg:
                continue

            print(f"\n[FROM ROBOT] 수신 메시지: {msg}")

            if msg == "shot":
                try:
                    # JSONL 한 줄 읽기
                    entry = next(data_iter)

                    cx = float(entry["X_m"])
                    cy = float(entry["Y_m"])
                    cz = float(entry["Z_m"])

                    nx = float(entry.get("nx", 0.0))
                    ny = float(entry.get("ny", 0.0))
                    nz = float(entry.get("nz", 1.0))

                    # 카메라 → 로봇 좌표 변환
                    p_robot, n_robot = transform_point_and_normal(
                        (cx, cy, cz),
                        (nx, ny, nz),
                    )
                    rx_mm, ry_mm, rz_mm = p_robot

                    # normal을 Z축으로 하는 회전행렬 & ZYZ Euler(A,B,C)
                    R_tool = build_rot_from_normal(n_robot)
                    A_rad, B_rad, C_rad = euler_zyz_from_rotm(R_tool)

                    # rad → deg (Doosan A,B,C는 deg 단위) :contentReference[oaicite:2]{index=2}
                    A_deg = math.degrees(A_rad)
                    B_deg = math.degrees(B_rad)
                    C_deg = math.degrees(C_rad)

                    print("---------------------------------")
                    print(f"📂 Camera (m): {cx:.6f}, {cy:.6f}, {cz:.6f}")
                    print(f"🤖 Robot  (mm): {rx_mm:.2f}, {ry_mm:.2f}, {rz_mm:.2f}")
                    print(
                        f"🔺 Normal(base): {n_robot[0]:.6f}, {n_robot[1]:.6f}, {n_robot[2]:.6f}"
                    )
                    print(
                        f"🎯 Euler ZYZ(A,B,C)[deg]: {A_deg:.3f}, {B_deg:.3f}, {C_deg:.3f}"
                    )
                    print("---------------------------------")

                    # *** 로봇 전송 패킷 ***
                    # 형식: x, y, z, A, B, C  (단위: mm, deg)
                    send_str = (
                        f"{rx_mm:.3f},{ry_mm:.3f},{rz_mm:.3f},"
                        f"{A_deg:.3f},{B_deg:.3f},{C_deg:.3f}"
                    )

                    conn.sendall((send_str + "\r\n").encode())
                    print(f"[TO ROBOT] 전송 완료 → {send_str}")

                except StopIteration:
                    print("⚠️ JSONL 파일에 더 이상 데이터가 없습니다!")
                    conn.sendall(("EOF\r\n").encode())

            else:
                print("[INFO] 알 수 없는 명령, 무시합니다.")

        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
            break

    conn.close()
    server.close()
    print("[SERVER] 연결 종료")


if __name__ == "__main__":
    start_server()
