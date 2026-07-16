import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils
drawing_style = mp.solutions.drawing_utils.DrawingSpec(
    color=(52, 172, 76), thickness=1, circle_radius=1)
finger_points = {
    "Zeigefinger": (6, 8),
    "Mittelfinger": (10, 12),
    "Ringfinger": (14, 16),
    "kleiner Finger": (18, 20),
}
finger_count = 0
while True:
    camera_working, frame = camera.read()
    if not camera_working:
        print("Kamera funktioniert nicht!")
        break

    frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_result = hands.process(frame_converted)

    finger_count = 0

    if hands_result.multi_hand_landmarks is not None:

        for hand in hands_result.multi_hand_landmarks:

            drawing.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS, drawing_style, drawing_style)

            for finger, (point_joint, point_tip) in finger_points.items():

                if hand.landmark[point_tip].y < hand.landmark[point_joint].y:
                    finger_count += 1
                    print(f" {finger} ist ausgestreckt")

    cv2.putText(frame, f"{finger_count} Finger ausgestreckt!",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 172, 76), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Finger counting", frame)

camera.release()
cv2.destroyAllWindows()
