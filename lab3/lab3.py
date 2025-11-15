import cv2
import numpy as np

IMG_PATH = 'kpi_bing_map.png'

SATURATION_LIMIT = 38
MIN_BUILDING_AREA = 1100
MAX_BUILDING_AREA = 32000
MIN_SOLIDITY = 0.01

OPEN_KERNEL = np.ones((4, 4), np.uint8)
CLOSE_KERNEL = np.ones((12, 12), np.uint8)
DILATE_KERNEL = np.ones((2, 2), np.uint8)

CANNY_LOW = 130
CANNY_HIGH = 160

CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)


def detect_buildings(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, building_mask = cv2.threshold(saturation, SATURATION_LIMIT, 255, cv2.THRESH_BINARY_INV)

    opened = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    building_mask_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced_gray = clahe.apply(gray)

    smooth = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(smooth, CANNY_LOW, CANNY_HIGH)

    inv_saturation_mask = cv2.bitwise_not(building_mask)
    edges_on_buildings = cv2.bitwise_and(edges, inv_saturation_mask)

    combined_mask = cv2.bitwise_or(building_mask_closed, edges_on_buildings)

    thick_mask = cv2.dilate(combined_mask, DILATE_KERNEL, iterations=1)

    contours, _ = cv2.findContours(thick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = image.copy()
    valid_buildings = 0
    filled_mask = np.zeros_like(thick_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BUILDING_AREA < area < MAX_BUILDING_AREA):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity >= MIN_SOLIDITY:
            valid_buildings += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return (
        result_img, valid_buildings,
        building_mask, building_mask_closed,
        enhanced_gray, edges, edges_on_buildings,
        combined_mask, thick_mask, filled_mask
    )


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        return

    (
        final_img, count,
        mask1, mask2, gray_enh, canny, canny_masked,
        union_mask, dilated, filled
    ) = detect_buildings(img)

    cv2.putText(final_img, f'Buildings: {count}', (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    stages = [
        ("", mask1),
        ("", mask2),
        ("", gray_enh),
        ("", canny),
        ("", canny_masked),
        ("", union_mask),
        ("", dilated),
        ("", filled),
        ("", final_img)
    ]

    for title, stage_img in stages:
        display = cv2.resize(stage_img, (900, 900)) if stage_img.ndim == 2 else stage_img
        display = cv2.resize(display, (900, 900))
        cv2.imshow(title, display)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()