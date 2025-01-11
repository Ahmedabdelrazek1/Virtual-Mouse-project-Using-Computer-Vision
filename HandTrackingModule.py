import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.prev_index_position = None  
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    detector = handDetector()
    screenWidth, screenHeight = pyautogui.size() 
    frameWidth, frameHeight = 640, 480  
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img)
        if len(lmList) != 0:
            
            x, y = lmList[8][1], lmList[8][2]  
            screenX = np.interp(x, (0, frameWidth), (0, screenWidth))
            screenY = np.interp(y, (0, frameHeight), (0, screenHeight))
            pyautogui.moveTo(screenX, screenY, duration=0.1)

            current_index_y = lmList[8][2]  
            if detector.prev_index_position is not None:
                prev_index_y = detector.prev_index_position  

                if current_index_y < prev_index_y:  
                    pyautogui.click()  

            detector.prev_index_position = current_index_y

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
