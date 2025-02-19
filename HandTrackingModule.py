import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, detectionCon=0.5, maxHands=2):
        self.detectionCon = detectionCon
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []  # List to store landmark positions

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized landmark coordinates to pixels
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb (Landmarks 4 and 3)
        if self.lmList[4][1] > self.lmList[3][1]:  # Check if the thumb is "up"
            fingers.append(1)
        else:
            fingers.append(0)

        # Index Finger (Landmarks 8 and 7)
        if self.lmList[8][2] < self.lmList[6][2]:  # Check if the index finger is "up"
            fingers.append(1)
        else:
            fingers.append(0)

        # Middle Finger (Landmarks 12 and 11)
        if self.lmList[12][2] < self.lmList[10][2]:  # Check if the middle finger is "up"
            fingers.append(1)
        else:
            fingers.append(0)

        # Ring Finger (Landmarks 16 and 15)
        if self.lmList[16][2] < self.lmList[14][2]:  # Check if the ring finger is "up"
            fingers.append(1)
        else:
            fingers.append(0)

        # Pinky Finger (Landmarks 20 and 19)
        if self.lmList[20][2] < self.lmList[18][2]:  # Check if the pinky is "up"
            fingers.append(1)
        else:
            fingers.append(0)

        return fingers
