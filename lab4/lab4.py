import cv2
import numpy as np


def MOG2Tracking(cap):
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    cv2.namedWindow("Select ROI")
    x, y, w, h = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    roi_coords = (x, y, w, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = bg_sub.apply(frame)

        rx, ry, rw, rh = roi_coords

        roi_mask = mask[ry:ry + rh, rx:rx + rw]

        roi_mask = cv2.threshold(roi_mask, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 150:
                xx, yy, ww, hh = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (rx + xx, ry + yy), (rx + xx + ww, ry + yy + hh), (0, 255, 0), 2)

        cv2.imshow("MOG2 Tracking", frame)

        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



def PyramidalLKOopticalFlow(cap):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    cv2.namedWindow("Select ROI")
    x, y, w, h = cv2.selectROI("Select ROI", old_frame, False)
    cv2.destroyWindow("Select ROI")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    roi_gray = old_gray[y:y + h, x:x + w]

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
    if p0 is None:
        print("Не знайдено точок для відстеження")
        return

    p0[:, 0, 0] += x
    p0[:, 0, 1] += y

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]

            if len(good_new) > 0:
                x_min, y_min = good_new.min(axis=0).astype(int)
                x_max, y_max = good_new.max(axis=0).astype(int)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.imshow('Pyramidal Lucas-Kanade Optical Flow', frame)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def KalmanFilterTracking(cap):
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    cv2.namedWindow("Select ROI")
    bbox = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = bbox

    template = frame[y:y + h, x:x + w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    cx = x + w / 2
    cy = y + h / 2

    kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
    kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    search_window = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sx1 = max(0, pred_x - w // 2 - search_window)
        sy1 = max(0, pred_y - h // 2 - search_window)
        sx2 = min(frame.shape[1], pred_x + w // 2 + search_window)
        sy2 = min(frame.shape[0], pred_y + h // 2 + search_window)

        search_region = frame_gray[sy1:sy2, sx1:sx2]

        if search_region.shape[0] >= template_gray.shape[0] and \
           search_region.shape[1] >= template_gray.shape[1]:

            res = cv2.matchTemplate(search_region, template_gray, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)

            mx = sx1 + max_loc[0] + w // 2
            my = sy1 + max_loc[1] + h // 2

            measurement = np.array([[mx], [my]], np.float32)
            corrected = kalman.correct(measurement)

            cx, cy = int(corrected[0]), int(corrected[1])

            cv2.rectangle(frame, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), (0, 0, 255), 2)

        cv2.imshow("Kalman Filter Tracking", frame)

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    print("Оберіть метод:")
    print("1 - MOG2 Background Subtraction=")
    print("2 - Lucas-Kanade Optical Flow")
    print("3 - Kalman Filter Tracking")

    mode = int(input("mode: "))

    if mode == 1:
        MOG2Tracking(cap)
    elif mode == 2:
        PyramidalLKOopticalFlow(cap)
    elif mode == 3:
        KalmanFilterTracking(cap)
    else:
        print("Невірний вибір методу!")
