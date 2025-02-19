import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm
import math

def virtual_Painter():
    brushThickness = 5
    eraserThickness = 50  # Eraser thickness
    folderPath = "Header"

    # Load all images from the folder for the toolbar
    myList = os.listdir(folderPath)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    header = overlayList[0]  # Default header
    drawColor = (255, 0, 255)  # Default drawing color (Purple)
    shape = 'freestyle'  # Default shape mode

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera")
        return

    cap.set(3, 1280)  # Width
    cap.set(4, 720)  # Height

    detector = htm.handDetector(detectionCon=0.85, maxHands=1)
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    imgCanvasWhite = np.ones((720, 1280, 3), np.uint8) * 255  # Blank white canvas

    history = []  # Undo stack
    redo_stack = []  # Redo stack

    selected_tool_x = None  # Variable to track selected tool for underline
    shape_size = 50  # Default size for circle and square

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image")
            break

        img = cv2.flip(img, 1)  # Flip the image horizontally

        # 1. Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip

            fingers = detector.fingersUp()

            # Calculate the distance between the index and middle finger to adjust shape size
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            shape_size = int(distance / 2)  # Adjust size of shapes based on finger distance
            shape_size = max(20, shape_size)  # Ensure minimum size of 20 pixels

            # 2. If Selection Mode - Two fingers are up (select tool/color/shape)
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0  # Reset previous positions

                # Selection mode for color or shape
                if y1 < 120:
                    if 150 < x1 < 300:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)  # Purple
                        selected_tool_x = 225
                        shape = 'freestyle'  # Freestyle drawing
                    elif 350 < x1 < 500:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)  # Blue
                        selected_tool_x = 425
                        shape = 'freestyle'
                    elif 550 < x1 < 700:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)  # Green
                        selected_tool_x = 625
                        shape = 'freestyle'
                    elif 750 < x1 < 900:
                        header = overlayList[3]
                        shape = 'rectangle'  # Rectangle shape
                        selected_tool_x = 825
                    elif 950 < x1 < 1100:
                        header = overlayList[4]
                        shape = 'circle'  # Circle shape
                        selected_tool_x = 1025
                    elif 1150 < x1 < 1300:
                        header = overlayList[5]
                        drawColor = (0, 0, 0)  # Eraser
                        selected_tool_x = 1225
                        shape = 'freestyle'

                # Highlight the selected tool
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # 3. Drawing Mode - Only Index Finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Draw based on selected shape mode
                if shape == 'freestyle':
                    if drawColor == (0, 0, 0):  # Eraser
                        cv2.line(img, (xp, yp), (x1, y1), (255, 255, 255), eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
                        cv2.line(imgCanvasWhite, (xp, yp), (x1, y1), (255, 255, 255), eraserThickness)
                    else:
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.line(imgCanvasWhite, (xp, yp), (x1, y1), drawColor, brushThickness)

                elif shape == 'rectangle':
                    if xp and yp:
                        side_length = shape_size  # Use the calculated shape size for the square
                        cv2.rectangle(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                        cv2.rectangle(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                elif shape == 'circle':
                    radius = shape_size  # Use the calculated size for the circle
                    cv2.circle(img, (xp, yp), radius, drawColor, brushThickness)
                    cv2.circle(imgCanvas, (xp, yp), radius, drawColor, brushThickness)

                xp, yp = x1, y1

        # 4. Update the history for undo/redo
        if len(history) == 0 or not (np.array_equal(imgCanvasWhite, history[-1][1]) and np.array_equal(imgCanvas, history[-1][0])):
            history.append((imgCanvas.copy(), imgCanvasWhite.copy()))
            redo_stack.clear()  # Clear redo stack when new drawing is done

        # 5. Resize and set the header
        header_resized = cv2.resize(header, (1280, 120))
        img[0:120, 0:1280] = header_resized

        # Underline the selected tool
        if selected_tool_x is not None:
            cv2.line(img, (selected_tool_x - 50, 115), (selected_tool_x + 50, 115), (255, 255, 255), 5)

        # Combine canvas and image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Show live feed and canvas
        cv2.imshow("Virtual Painter", img)
        cv2.imshow("White Canvas", imgCanvasWhite)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f"canvas_{timestamp}.png", imgCanvasWhite)
            print(f"Saved drawing as canvas_{timestamp}.png")

        # Undo and redo actions
        if key == ord('u') and len(history) > 1:
            redo_stack.append(history.pop())
            if len(history) > 0:
                imgCanvas, imgCanvasWhite = history[-1]
            print("Undo Action Triggered")

        if key == ord('r') and len(redo_stack) > 0:
            imgCanvas, imgCanvasWhite = redo_stack.pop()
            history.append((imgCanvas.copy(), imgCanvasWhite.copy()))
            print("Redo Action Triggered")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_Painter()
