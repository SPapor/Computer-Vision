import cv2

video = 0
W = 640
H = 480
min = 5000
kcf_or_kfc = True

cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

bg = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=40,detectShadows=False)

def create_tracker():
    if kcf_or_kfc:
        try:
            return cv2.TrackerKCF_create()
        except:
            pass
    try:
        return cv2.TrackerCSRT_create()
    except:
        pass
    return cv2.TrackerMIL_create()

tracker = None
tracking = False
bbox = None
lost_frames = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))
    fgmask = bg.apply(frame)
    fgmask = cv2.erode(fgmask, None, iterations=1)
    fgmask = cv2.dilate(fgmask, None, iterations=4)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    best_bbox = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area and area > min:
            largest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.3 < w/h < 3.0:
                largest_area = area
                best_bbox = (x, y, w, h)

    if best_bbox is not None:
        x, y, w, h = best_bbox

        if not tracking:
            tracker = create_tracker()
            tracker.init(frame, (x, y, w, h))
            tracking = True
            lost_frames = 0

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, f"TRACKING ", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if tracking:
        success, new_bbox = tracker.update(frame)
        if success:
            lost_frames = 0
            x, y, w, h = [int(v) for v in new_bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "TRACKING", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            lost_frames += 1
            cv2.putText(frame, "LOST", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if lost_frames > 30:
            tracking = False
            tracker = None

    if best_bbox is None and not tracking:
        cv2.putText(frame, "Waiting ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    cv2.imshow("", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()