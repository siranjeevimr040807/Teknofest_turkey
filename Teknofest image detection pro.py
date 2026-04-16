import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# -------------------------
# KALMAN FILTER FOR RED
# -------------------------
kf_red = cv2.KalmanFilter(4,2)

kf_red.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kf_red.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kf_red.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03


# -------------------------
# KALMAN FILTER FOR BLUE
# -------------------------
kf_blue = cv2.KalmanFilter(4,2)

kf_blue.measurementMatrix = np.array([[1,0,0,0],
                                      [0,1,0,0]], np.float32)

kf_blue.transitionMatrix = np.array([[1,0,1,0],
                                     [0,1,0,1],
                                     [0,0,1,0],
                                     [0,0,0,1]], np.float32)

kf_blue.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03


while True:

    ret, image = cap.read()
    if not ret:
        break

    image = cv2.resize(image,(800,600))

    h,w = image.shape[:2]
    cam_center = (w//2 , h//2)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -------------------------
    # RED MASK
    # -------------------------
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask_red = cv2.inRange(hsv,lower_red1,upper_red1) + \
               cv2.inRange(hsv,lower_red2,upper_red2)

    # -------------------------
    # BLUE MASK
    # -------------------------
    lower_blue = np.array([100,120,70])
    upper_blue = np.array([140,255,255])

    mask_blue = cv2.inRange(hsv,lower_blue,upper_blue)

    # -------------------------
    # MORPHOLOGY
    # -------------------------
    kernel = np.ones((5,5),np.uint8)

    mask_red = cv2.morphologyEx(mask_red,cv2.MORPH_OPEN,kernel)
    mask_red = cv2.morphologyEx(mask_red,cv2.MORPH_CLOSE,kernel)

    mask_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_OPEN,kernel)
    mask_blue = cv2.morphologyEx(mask_blue,cv2.MORPH_CLOSE,kernel)

    # -------------------------
    # RED TRIANGLE DETECTION
    # -------------------------
    contours,_ = cv2.findContours(mask_red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

            epsilon = 0.04 * cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            if len(approx) == 3:

                cv2.drawContours(image,[approx],0,(0,0,255),3)

                M = cv2.moments(cnt)

                if M["m00"] != 0:

                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    measurement = np.array([[np.float32(cx)],
                                            [np.float32(cy)]])

                    kf_red.correct(measurement)

    # -------------------------
    # BLUE HEXAGON DETECTION
    # -------------------------
    contours,_ = cv2.findContours(mask_blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

            epsilon = 0.04 * cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            if len(approx) == 6:

                cv2.drawContours(image,[approx],0,(255,0,0),3)

                M = cv2.moments(cnt)

                if M["m00"] != 0:

                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    measurement = np.array([[np.float32(cx)],
                                            [np.float32(cy)]])

                    kf_blue.correct(measurement)

    # -------------------------
    # KALMAN PREDICTION
    # -------------------------
    pred_red = kf_red.predict()
    pred_blue = kf_blue.predict()

    red_point = (int(pred_red[0]), int(pred_red[1]))
    blue_point = (int(pred_blue[0]), int(pred_blue[1]))

    # -------------------------
    # DRAW CAMERA CENTER
    # -------------------------
    cv2.circle(image,cam_center,6,(0,255,0),-1)

    # -------------------------
    # DRAW RED TRACK
    # -------------------------
    cv2.circle(image,red_point,7,(0,0,255),-1)

    cv2.line(image,
             red_point,
             cam_center,
             (0,0,255),
             2)

    cv2.putText(image,
                f"Red Triangle: {red_point}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2)

    # -------------------------
    # DRAW BLUE TRACK
    # -------------------------
    cv2.circle(image,blue_point,7,(255,0,0),-1)

    cv2.line(image,
             blue_point,
             cam_center,
             (255,0,0),
             2)

    cv2.putText(image,
                f"Blue Hexagon: {blue_point}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2)

    # -------------------------
    # SHOW RESULT
    # -------------------------
    cv2.imshow("result",image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
