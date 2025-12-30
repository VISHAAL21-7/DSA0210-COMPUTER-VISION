import cv2

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

slow_repeat = 4   # Higher value = slower video
slow_count = 0
slow_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for better display
    frame = cv2.resize(frame, (400, 300))

    # -------- SLOW MOTION --------
    if slow_count == 0:
        slow_frame = frame.copy()

    slow_display = slow_frame.copy()
    slow_count = (slow_count + 1) % slow_repeat

    # -------- FAST MOTION --------
    cap.read()  # skip frame
    cap.read()  # skip frame
    fast_display = frame.copy()

    # -------- SIDE BY SIDE --------
    combined = cv2.hconcat([slow_display, fast_display])

    # Labels
    cv2.putText(combined, "SLOW MOTION", (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(combined, "FAST MOTION", (450, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam Slow vs Fast Motion", combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
