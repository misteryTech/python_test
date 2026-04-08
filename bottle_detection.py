import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import serial
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH           = "runs/detect/bottle_qc_final9/weights/best.pt"
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
CONFIDENCE_THRESHOLD = 0.70
SAVE_REJECTS         = True
REJECT_FOLDER        = "rejected_bottles"

# Arduino
ARDUINO_PORT         = "COM6"
BAUD_RATE            = 115200

# Frames needed before confirming detection
CONFIRM_FRAMES       = 5

# Label colors (BGR)
LABEL_COLORS = {
    "good_bottle":  (0, 200, 0),
    "broken_glass": (0, 0, 220),
    "dirty_bottle": (0, 165, 255),
}

# ─────────────────────────────────────────────
# BEHAVIOR SUMMARY
# ─────────────────────────────────────────────
# good_bottle  → Relay ON 15s, Servo triggered
# dirty_bottle → Motor ON 3s then STOP, Servo 180°
# broken_glass → Motor ON 4s then STOP, Servo 180°
# no bottle    → All OFF (only if Arduino is NOT busy)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# ARDUINO CONNECTION
# ─────────────────────────────────────────────
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[✓] Arduino connected on {ARDUINO_PORT}")
except Exception as e:
    arduino = None
    print(f"[!] Arduino not found ({e}) — running without hardware")

# ─────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────
def setup_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"[✗] Cannot open camera at index {index}")
    print(f"[✓] Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap

# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────
def load_model(path):
    if not os.path.exists(path):
        print(f"[!] Model '{path}' not found — loading fallback yolov8n.pt")
        return YOLO("yolov8n.pt")
    model = YOLO(path)
    print(f"[✓] Model loaded: {path}")
    return model

# ─────────────────────────────────────────────
# FOLDER SETUP
# ─────────────────────────────────────────────
def setup_save_folder(folder):
    if SAVE_REJECTS and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[✓] Reject folder created: {folder}/")

# ─────────────────────────────────────────────
# SEND COMMAND TO ARDUINO
# ─────────────────────────────────────────────
def send_command(cmd, label=""):
    if arduino:
        try:
            arduino.write(cmd)
            print(f"  [→] Arduino command: {cmd} {label}")
        except Exception as e:
            print(f"  [!] Serial error: {e}")

# ─────────────────────────────────────────────
# SAVE REJECTED IMAGE
# ─────────────────────────────────────────────
def save_rejected(frame, label):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{REJECT_FOLDER}/{label}_{ts}.jpg"
    cv2.imwrite(filename, frame)
    print(f"  💾 Saved: {filename}")

# ─────────────────────────────────────────────
# DRAW BOUNDING BOXES
# ─────────────────────────────────────────────
def draw_detections(frame, results, conf_threshold):
    detected_labels = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            detected_labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = LABEL_COLORS.get(label, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label}  {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame, detected_labels

# ─────────────────────────────────────────────
# DRAW HUD
# ─────────────────────────────────────────────
def draw_hud(frame, detected_labels, fps,
             relay_state, motor_state, servo_state,
             confirm_count, confirm_needed,
             action_timer, action_duration,
             arduino_busy):                          # ← new param
    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    a_text  = "Arduino: CONNECTED" if arduino else "Arduino: NOT FOUND"
    a_color = (0, 255, 0) if arduino else (0, 0, 255)
    cv2.putText(frame, a_text, (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, a_color, 2)

    # Relay
    r_text  = f"Relay:  {'ON ' if relay_state == 'ON' else 'OFF'}"
    r_color = (0, 255, 0) if relay_state == "ON" else (0, 0, 255)
    cv2.putText(frame, r_text, (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, r_color, 2)

    # Motor
    m_text  = f"Motor:  {'ON ' if motor_state == 'ON' else 'OFF'}"
    m_color = (0, 220, 255) if motor_state == "ON" else (100, 100, 100)
    cv2.putText(frame, m_text, (10, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, m_color, 2)

    # Servo
    s_text  = f"Servo:  {'180°' if servo_state == 'ON' else '0°  '}"
    s_color = (0, 165, 255) if servo_state == "ON" else (100, 100, 100)
    cv2.putText(frame, s_text, (10, 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 2)

    # ── BUSY indicator ──                           # ← new
    busy_text  = "ARDUINO: BUSY - waiting..." if arduino_busy else "ARDUINO: READY"
    busy_color = (0, 200, 255) if arduino_busy else (100, 100, 100)
    cv2.putText(frame, busy_text, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, busy_color, 1)

    # Action countdown timer bar
    if action_timer is not None and action_duration > 0:
        elapsed   = time.time() - action_timer
        remaining = max(0.0, action_duration - elapsed)
        pct       = remaining / action_duration
        bar_w, bar_h = 160, 10
        filled = int(pct * bar_w)
        cv2.rectangle(frame, (10, 160), (10 + bar_w, 160 + bar_h), (60, 60, 60), -1)
        bar_color = (0, 255, 120) if relay_state == "ON" else (0, 165, 255)
        cv2.rectangle(frame, (10, 160), (10 + filled, 160 + bar_h), bar_color, -1)
        cv2.putText(frame, f"Timer: {remaining:.1f}s / {action_duration}s",
                    (10, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Confirmation bar
    bar_w2 = 160
    filled2 = int((min(confirm_count, confirm_needed) / confirm_needed) * bar_w2)
    cv2.rectangle(frame, (10, 194), (10 + bar_w2, 206), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 194), (10 + filled2, 206), (0, 220, 255), -1)
    cv2.putText(frame, f"Confirm: {min(confirm_count, confirm_needed)}/{confirm_needed}",
                (10, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # ── Status banner top right ──
    if arduino_busy:
        # Show what's currently running even if camera sees nothing
        status_text  = "ARDUINO BUSY - TASK RUNNING..."
        banner_color = (100, 80, 0)
    elif not detected_labels:
        status_text  = "NO BOTTLE"
        banner_color = (50, 50, 50)
    elif all(l == "good_bottle" for l in detected_labels):
        status_text  = "GOOD  |  RELAY 15s + SERVO"
        banner_color = (0, 140, 0)
    elif "broken_glass" in detected_labels:
        status_text  = "BROKEN  |  MOTOR 4s + SERVO 180"
        banner_color = (0, 0, 170)
    else:
        status_text  = "DIRTY  |  MOTOR 3s + SERVO 180"
        banner_color = (0, 100, 200)

    (bw, bh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (w - bw - 28, 0), (w, 40), banner_color, -1)
    cv2.putText(frame, status_text, (w - bw - 14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bottle QC Detection System")
    print("  good_bottle  → Relay ON 15s + Servo triggered")
    print("  dirty_bottle → Motor ON 3s STOP + Servo 180°")
    print("  broken_glass → Motor ON 4s STOP + Servo 180°")
    print("  no bottle    → All OFF (only when Arduino is free)")
    print("  Press Q to quit")
    print("=" * 60)

    setup_save_folder(REJECT_FOLDER)
    cap   = setup_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    model = load_model(MODEL_PATH)

    prev_time      = datetime.now()
    fps            = 0.0

    # Detection state
    prev_status    = None
    pending_status = None
    confirm_count  = 0

    # Hardware display state
    relay_state     = "OFF"
    motor_state     = "OFF"
    servo_state     = "OFF"
    action_timer    = None
    action_duration = 0

    # ── BUSY FLAG ──────────────────────────────
    # True = Arduino is executing a task.
    # Python will NOT send any new command until
    # the full action_duration has elapsed.
    arduino_busy    = False                          # ← new
    # ───────────────────────────────────────────

    send_command(b'N', "startup - all OFF")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        # ── Inference ──
        results = model(frame, verbose=False)
        frame, detected_labels = draw_detections(frame, results, CONFIDENCE_THRESHOLD)

        # ── FPS ──
        now       = datetime.now()
        elapsed_t = (now - prev_time).total_seconds()
        fps       = 1.0 / elapsed_t if elapsed_t > 0 else fps
        prev_time = now

        # ── Raw status this frame ──
        if not detected_labels:
            raw_status = "NONE"
        elif all(l == "good_bottle" for l in detected_labels):
            raw_status = "GOOD"
        elif "broken_glass" in detected_labels:
            raw_status = "BROKEN"
        else:
            raw_status = "DIRTY"

        # ── Confirmation counter ──
        if raw_status == pending_status:
            confirm_count += 1
        else:
            pending_status = raw_status
            confirm_count  = 1

        # ── Check if Arduino finished its task ──  # ← new block
        if arduino_busy and action_timer is not None:
            if time.time() - action_timer >= action_duration:
                # Task fully done — unlock Python
                arduino_busy    = False
                relay_state     = "OFF"
                motor_state     = "OFF"
                servo_state     = "OFF"
                action_timer    = None
                action_duration = 0
                prev_status     = None  # reset so next bottle triggers fresh
                confirm_count   = 0
                pending_status  = None
                print("  [✓] Arduino task done — ready for next bottle")

        # ── Only act if Arduino is FREE ──         # ← guard
        if not arduino_busy:

            # ── Trigger on confirmed status change ──
            if confirm_count >= CONFIRM_FRAMES and raw_status != prev_status:
                prev_status = raw_status

                if raw_status == "GOOD":
                    print("  ✅ GOOD BOTTLE — Relay ON 15s + Servo triggered")
                    send_command(b'G', "Relay ON 15s + Servo")
                    relay_state     = "ON"
                    motor_state     = "OFF"
                    servo_state     = "ON"
                    action_timer    = time.time()
                    action_duration = 15
                    arduino_busy    = True           # ← lock

                elif raw_status == "DIRTY":
                    print("  🟠 DIRTY BOTTLE — Motor 3s STOP + Servo 180°")
                    send_command(b'D', "Motor 3s + Servo 180")
                    relay_state     = "OFF"
                    motor_state     = "ON"
                    servo_state     = "ON"
                    action_timer    = time.time()
                    action_duration = 3
                    arduino_busy    = True           # ← lock
                    if SAVE_REJECTS:
                        save_rejected(frame, "dirty_bottle")

                elif raw_status == "BROKEN":
                    print("  ❌ BROKEN GLASS — Motor 4s STOP + Servo 180°")
                    send_command(b'B', "Motor 4s + Servo 180")
                    relay_state     = "OFF"
                    motor_state     = "ON"
                    servo_state     = "ON"
                    action_timer    = time.time()
                    action_duration = 4
                    arduino_busy    = True           # ← lock
                    if SAVE_REJECTS:
                        save_rejected(frame, "broken_glass")

                elif raw_status == "NONE":
                    print("  ⬜ No bottle — All OFF")
                    send_command(b'N', "All OFF")
                    relay_state     = "OFF"
                    motor_state     = "OFF"
                    servo_state     = "OFF"
                    action_timer    = None
                    action_duration = 0

        # ── Draw HUD ──
        frame = draw_hud(frame, detected_labels, fps,
                         relay_state, motor_state, servo_state,
                         confirm_count, CONFIRM_FRAMES,
                         action_timer, action_duration,
                         arduino_busy)               # ← pass busy flag

        cv2.imshow("Bottle QC System  |  Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[✓] Quit requested. Exiting...")
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()

    if arduino:
        send_command(b'N', "exit - all OFF")
        time.sleep(0.3)
        arduino.close()
        print("[✓] Arduino disconnected. All OFF.")


if __name__ == "__main__":
    main()