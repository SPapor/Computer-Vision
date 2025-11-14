import cv2
import numpy as np

IMAGE_PATH = 'img.png'
WEAK_THRESHOLD = 121
STRONG_THRESHOLD = 132
MIN_AREA = 900
MAX_AREA = 42000

def resize_for_display(img, max_width=1000, max_height=800):
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def find_buildings(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    _, strong = cv2.threshold(L, STRONG_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, weak = cv2.threshold(L, WEAK_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(weak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros(L.shape, dtype=np.uint8)

    for cnt in contours:
        temp = np.zeros(L.shape, dtype=np.uint8)
        cv2.drawContours(temp, [cnt], -1, 255, thickness=cv2.FILLED)
        intersection = cv2.bitwise_and(temp, strong)
        if cv2.countNonZero(intersection) > 0:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    count = 0
    for cnt in final_contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(result, f'{count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return result, count


image = cv2.imread(IMAGE_PATH)



result, count = find_buildings(image)


cv2.imwrite('result.jpg', result)
display = resize_for_display(result)

cv2.imshow('', display)
cv2.waitKey(0)
cv2.destroyAllWindows()
