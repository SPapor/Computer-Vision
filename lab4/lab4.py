import cv2
import numpy as np


def BasicLKOopticalFlow(cap):
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    ret, old_frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    x, y, w, h = cv2.selectROI("Select ROI", old_frame, False)

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

            cv2.imshow('Basic Lucas-Kanade Optical Flow', frame)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            print("Точки втрачено")
            break

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


def PyramidalLKOopticalFlow(cap):
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    ret, old_frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    x, y, w, h = cv2.selectROI("Select ROI", old_frame, False)

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
        else:
            print("Точки втрачено")
            break

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


def KalmanFilterTracking(cap):
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося зчитати відео")
        return

    bbox = cv2.selectROI("Select ROI", frame, False)

    x, y, w, h = bbox

    template = frame[y:y + h, x:x + w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    center_x = x + w / 2
    center_y = y + h / 2
    kalman.statePre = np.array([[np.float32(center_x)], [np.float32(center_y)], [0], [0]], np.float32)
    kalman.statePost = np.array([[np.float32(center_x)], [np.float32(center_y)], [0], [0]], np.float32)

    search_window = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        search_x1 = max(0, pred_x - w // 2 - search_window)
        search_y1 = max(0, pred_y - h // 2 - search_window)
        search_x2 = min(frame.shape[1], pred_x + w // 2 + search_window)
        search_y2 = min(frame.shape[0], pred_y + h // 2 + search_window)

        search_region = frame_gray[search_y1:search_y2, search_x1:search_x2]

        if search_region.shape[0] >= template_gray.shape[0] and search_region.shape[1] >= template_gray.shape[1]:
            result = cv2.matchTemplate(search_region, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            match_x = search_x1 + max_loc[0] + w // 2
            match_y = search_y1 + max_loc[1] + h // 2

            measurement = np.array([[np.float32(match_x)], [np.float32(match_y)]])

            corrected = kalman.correct(measurement)
            corr_x, corr_y = int(corrected[0]), int(corrected[1])

            corr_x1 = int(corr_x - w // 2)
            corr_y1 = int(corr_y - h // 2)
            cv2.rectangle(frame, (corr_x1, corr_y1), (corr_x1 + w, corr_y1 + h), (0, 0, 255), 2)

        cv2.imshow('Kalman Filter Tracking', frame)

        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    print('Оберіть метод object tracking :')
    print('1 - Lucas-Kanade Optical Flow (Basic KLT)')
    print('2 - Pyramidal LK (Multi-level KLT)')
    print('3 - Kalman Filter Tracking')

    mode = int(input('mode:'))

    if mode == 1:
        print('1 - Lucas-Kanade Optical Flow (Basic KLT)')
        BasicLKOopticalFlow(cap)

    elif mode == 2:
        print('2 - Pyramidal LK (Multi-level KLT)')
        PyramidalLKOopticalFlow(cap)

    elif mode == 3:
        print('3 - Kalman Filter Tracking')
        KalmanFilterTracking(cap)

    else:
        print('Невірний вибір методу. Доступні лише 1, 2 та 3.')