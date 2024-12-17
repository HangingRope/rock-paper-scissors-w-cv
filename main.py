import cv2
import random
import mediapipe as mp
from utils import get_gesture, puttext

class RockPaperScissorsGame:
    def __init__(self):
        self.options = ['Rock', 'Paper', 'Scissors']
        self.last_pos = "None"
        self.FONT_SIZE = 2  
        self.FONT_THICKNESS = 3  

    def run(self):
        cap = cv2.VideoCapture(0)

        # Set the window size (for example, 800x600)
        cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediaPipe Hands', 800, 600)

        with mp.solutions.hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                success, image = cap.read()

                image = cv2.flip(image, 1)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = hands.process(image)

                image.flags.writeable = True

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        lmList = []
                        for id, lm in enumerate(hand_landmarks.landmark):
                            h, w, c = image.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([id, cx, cy])
                        res = get_gesture(lmList)
                        if res != self.last_pos:
                            ans = self.options[random.randint(0, len(self.options) - 1)]
                            self.last_pos = res
                        
                        cv2.putText(image, ("You: "+ str(res)), (400, 100), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (0, 0, 0), self.FONT_THICKNESS+2)  # Shadow
                        cv2.putText(image, ("You: "+ str(res)), (400, 100), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (255, 255, 255), self.FONT_THICKNESS)  # Text
                        
                        cv2.putText(image, ("CPU: "+ str(ans)), (60, 100), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (0, 0, 0), self.FONT_THICKNESS+2)  # Shadow
                        cv2.putText(image, ("CPU: "+ str(ans)), (60, 100), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (255, 0, 255), self.FONT_THICKNESS)  # Text
                        
                        result_text = str(puttext(res, ans))
                        cv2.putText(image, result_text, (100, 400), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (0, 0, 0), self.FONT_THICKNESS+2)  # Shadow
                        cv2.putText(image, result_text, (100, 400), cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (255, 0, 255), self.FONT_THICKNESS)  # Text

                cv2.imshow('MediaPipe Hands', image)

                if (cv2.waitKey(5) & 0xFF == 27): 
                    break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    game = RockPaperScissorsGame()
    game.run()
