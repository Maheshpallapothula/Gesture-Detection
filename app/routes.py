from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
import mediapipe as mp

router = APIRouter()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Define the list of positive gestures
POSITIVE_GESTURES = {
    "thumbs_up": "👍",
    "peace": "✌️",
    "wave": "🖐️", #NW
    "heart": "💖",
    "super": "🤟",
    "ok": "👌" #NW
}


@router.post("/recognize-gesture/")
async def recognize_gesture(file: UploadFile = File(...)):
    """
    Recognize hand gestures (e.g., thumbs up, peace, wave, heart gesture, super gesture, OK).
    """
    # Read and decode the uploaded image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format")

    if image is None:
        raise HTTPException(status_code=400, detail="Unable to process the image")

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return {"gesture": "No hand detected", "symbol": "❌"}

        # If two hands are detected, check for combined gestures like heart
        if len(results.multi_hand_landmarks) == 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            # Check for heart gesture
            if detect_heart_gesture(hand1, hand2):
                return {"gesture": "heart", "symbol": "💖"}

        # Handle single-hand gestures
        for hand_landmarks in results.multi_hand_landmarks:
            gesture_name = detect_single_hand_gesture(hand_landmarks)

            if gesture_name in POSITIVE_GESTURES:
                return {"gesture": gesture_name, "symbol": POSITIVE_GESTURES[gesture_name]}

        # If no positive gestures match
        return {"gesture": "Unrecognized gesture", "symbol": "❓"}


def detect_heart_gesture(hand1, hand2):
    """
    Detect a heart gesture made with two hands.
    """
    # Extract key landmarks from both hands
    hand1_thumb_tip = hand1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    hand1_index_tip = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    hand2_thumb_tip = hand2.landmark[mp_hands.HandLandmark.THUMB_TIP]
    hand2_index_tip = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the thumbs and index fingers are close enough to form a heart shape
    thumb_distance = np.linalg.norm([
        hand1_thumb_tip.x - hand2_thumb_tip.x,
        hand1_thumb_tip.y - hand2_thumb_tip.y
    ])
    index_distance = np.linalg.norm([
        hand1_index_tip.x - hand2_index_tip.x,
        hand1_index_tip.y - hand2_index_tip.y
    ])

    # Adjust threshold to detect proximity of tips (tuned based on testing)
    if thumb_distance < 0.05 and index_distance < 0.05:
        return True

    return False


def detect_single_hand_gesture(hand_landmarks):
    """
    Detect gestures based on a single hand's landmarks.
    """
    # Extract key landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # 1. Gesture: Thumbs Up
    if thumb_tip.y < thumb_mcp.y and all(
        finger.y > thumb_mcp.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]
    ):
        return "thumbs_up"

    # 2. Gesture: Peace
    if index_tip.y < index_pip.y and middle_tip.y < middle_pip.y and all(
        finger.y > middle_pip.y for finger in [ring_tip, pinky_tip]
    ):
        return "peace"

    # 3. Gesture: Wave
    wave = all(
        finger.y < index_pip.y for finger in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    ) and all(
        abs(finger.x - middle_tip.x) < 0.2 for finger in [index_tip, ring_tip, pinky_tip]
    )
    if wave:
        return "wave"

    # 4. Gesture: Super Gesture (🤟)
    super_gesture = (
        thumb_tip.y < thumb_mcp.y and  # Thumb extended
        index_tip.y < index_pip.y and  # Index extended
        pinky_tip.y < pinky_pip.y and  # Pinky extended
        middle_tip.y > middle_pip.y and  # Middle folded
        ring_tip.y > ring_pip.y  # Ring folded
    )
    if super_gesture:
        return "super"

    # 5. Gesture: OK Gesture (👌)
    ok_gesture = (
        np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]) < 0.05 and  # Thumb & index form a circle
        all(finger.y > index_pip.y for finger in [middle_tip, ring_tip, pinky_tip])  # Other fingers extended or folded
    )
    if ok_gesture:
        return "ok"

    return "unrecognized"
