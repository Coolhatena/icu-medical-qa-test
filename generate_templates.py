import cv2
import numpy as np
import os

LOW = np.array([110, 0, 0])
UPP = np.array([180, 255, 255])
def hsv_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW, UPP)
	# filtered = cv2.bitwise_and(img, img, mask= msk)
	return msk
      
base_template_path = "images/cap1.jpg"
output_dir = "images/template1"
os.makedirs(output_dir, exist_ok=True)

rotations = [0, 45, 90, 135, 180, 225, 270, 315]


template = cv2.imread(base_template_path, cv2.IMREAD_UNCHANGED)
template = hsv_filter(template)

if template is None:
    raise FileNotFoundError(f"No se encontró la imagen: {base_template_path}")

(h, w) = template.shape[:2]
center = (w // 2, h // 2)

for angle in rotations:
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(template, M, (new_w, new_h), borderValue=(0, 0, 0))

    out_path = os.path.join(output_dir, f"template_{angle}.png")
    cv2.imwrite(out_path, rotated)
    print(f"Guardada plantilla rotada: {out_path}")
