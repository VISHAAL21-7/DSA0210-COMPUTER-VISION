import cv2

cap = cv2.VideoCapture(
    r"C:/Users/msvis/Pictures/Camera Roll/WIN_20250921_20_27_41_Pro.mp4"
)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

slow_repeat = 3  # how many times to repeat frame for slow motion
slow_count = 0
slow_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (400, 300))

    # ---------- SLOW MOTION ----------
    if slow_count == 0:
        slow_frame = frame.copy()

    slow_display = slow_frame.copy()
    slow_count = (slow_count + 1) % slow_repeat

    # ---------- FAST MOTION ----------
    cap.read()  # skip 1 frame
    cap.read()  # skip 2 frames
    fast_display = frame.copy()

    # ---------- SIDE BY SIDE ----------
    combined = cv2.hconcat([slow_display, fast_display])

    cv2.putText(combined, "SLOW MOTION", (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(combined, "FAST MOTION", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Slow vs Fast Motion", combined)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
