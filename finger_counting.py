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
    "Daumen": (3, 4)
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
    is_finger_extended = False

    if hands_result.multi_hand_landmarks is not None:
        for hand, hand_side in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
            drawing.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS, drawing_style, drawing_style)

            for finger, (point_joint, point_tip) in finger_points.items():
                is_finger_extended = False
                if finger == "Daumen":
                    if hand_side.classification[0].label == "Right":
                        if hand.landmark[point_tip].x > hand.landmark[point_joint].x:
                            is_finger_extended = True

                    else:
                        if hand.landmark[point_tip].x < hand.landmark[point_joint].x:
                            is_finger_extended = True

                else:
                    if hand.landmark[point_tip].y < hand.landmark[point_joint].y:
                        is_finger_extended = True

                if is_finger_extended:
                    finger_count += 1
                    print(f"{finger} ist ausgestreckt!")

    cv2.putText(frame, f"{finger_count} Finger ausgestreckt!",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 172, 76), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Finger counting", frame)

camera.release()
cv2.destroyAllWindows()
