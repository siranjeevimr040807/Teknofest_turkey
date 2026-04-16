import cv2
import numpy as np

#reading image
#image = cv2.imread(r"C:\Users\MMMMsmmmVJCXJgnh\Downloads\openCVexample2.jpg")
#image = cv2.imread(r"C:\Users\MMMMsmmmVJCXJgnh\Downloads\openCVexample.jpg")
#image = cv2.imread(r"C:\Users\MMMMsmmmVJCXJgnh\Downloads\ChatGPT Image Mar 7, 2026, 10_50_09 AM.png")
#image = cv2.imread(r"C:\Users\MMMMsmmmVJCXJgnh\Downloads\ChatGPT Image Mar 9, 2026, 11_14_44 PM.png")
#image = cv2.imread(r"C:\Users\MMMMsmmmVJCXJgnh\Downloads\Symmetrical geometric shapes in bold colours.png")

# -------------------------
# OPEN VIDEO
# -------------------------
cap = cv2.VideoCapture(0)   # or 0 for webcam

while True:

    ret, image = cap.read()

    if not ret:
        break

    image = cv2.resize(image, (1366,768))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -------------------------
    # RED COLOR MASK
    # -------------------------

    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask_red1 + mask_red2

    # -------------------------
    # BLUE COLOR MASK
    # -------------------------

    lower_blue = np.array([100,120,70])
    upper_blue = np.array([140,255,255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # -------------------------
    # MORPHOLOGY
    # -------------------------

    kernel = np.ones((5,5),np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)


    red_mask = cv2.GaussianBlur(red_mask,(7,7),0)
    blue_mask = cv2.GaussianBlur(blue_mask,(7,7),0)

    # -------------------------
    # RED TRIANGLE DETECTION
    # -------------------------

    contours,_ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 100:

            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 3:

                cv2.drawContours(image, [approx], 0, (0,0,0), 3)

                x,y,w,h = cv2.boundingRect(approx)

                cv2.putText(image,"Red Triangle",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

                M = cv2.moments(cnt)

                if M["m00"] != 0:

                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    cv2.circle(image,(cx,cy),6,(0,0,0),-1)
                    

    # -------------------------
    # BLUE HEXAGON DETECTION
    # -------------------------

    contours,_ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 100:

            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 6:

                cv2.drawContours(image,[approx],0,(0,0,0),3)

                x,y,w,h = cv2.boundingRect(approx)

                cv2.putText(image,"Blue Hexagon",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

                M = cv2.moments(cnt)

                if M["m00"] != 0:

                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    cv2.circle(image,(cx,cy),6,(0,0,0),-1)

    # -------------------------
    # SHOW RESULT
    # -------------------------

    cv2.imshow("result", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



