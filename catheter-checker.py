import cv2
import os
import math
import numpy as np
// TODO: Setup on station with camera
LOW = np.array([95, 0, 0])
UPP = np.array([180, 255, 255])

LOW_tag = np.array([115, 100, 0])
UPP_tag = np.array([180, 255, 255])
q_unicode = ord('q')

template1_dir = "images/template1"
template2_dir = "images/template2"
rotations = [0, 45, 90, 135, 180, 225, 270, 315]
matching_threshold = 0.5

def rotarGrad(originx, originy, angle, rad):
	return (round(originx+rad*math.cos(-math.radians(angle))), round(originy+rad*math.sin(-math.radians(angle))))


def hsv_filter(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW, UPP)
	return msk


def filter_tag(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW_tag, UPP_tag)
	return msk	


def detect_best_template(scene_mask, template_dir):
	best_val = -1
	best_angle = None
	best_top_left = None
	best_w, best_h = 0, 0
	for angle in rotations:
		path = os.path.join(template_dir, f"template_{angle}.png")
		template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if template is None:
			continue

		res = cv2.matchTemplate(scene_mask, template, cv2.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		if max_val > best_val:
			best_val = max_val
			best_angle = angle
			best_top_left = max_loc
			best_w, best_h = template.shape[::-1]

	return {
		"score": best_val,
		"angle": best_angle,
		"top_left": best_top_left,
		"w": best_w,
		"h": best_h
	} if best_val > matching_threshold else None


images = ["img5", "img7", "img8", "img10", "img11"]

for img_name in images:
	img_scene_color = cv2.imread(f"images/{img_name}.jpg")
	if img_scene_color is None:
		break

	img_scene_mask = hsv_filter(img_scene_color)
	h_scene, w_scene = img_scene_mask.shape

	result1 = detect_best_template(img_scene_mask, template1_dir)

	if result1:
		top_left1 = result1["top_left"]
		w1, h1 = result1["w"], result1["h"]
		cx1 = top_left1[0] + w1 // 2
		cy1 = top_left1[1] + h1 // 2
		region = "abajo" if cy1 < h_scene // 2 else "arriba"

		bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
		figure_center1 = (int(top_left1[0] + w1/2), int(top_left1[1] + h1/2))

		if region == "abajo":
			mask_region2 = img_scene_mask[h_scene // 2:, :]  # Bottom side
			offset_y = h_scene // 2
			color = (0, 255, 0) if result1["angle"] not in [135, 180, 225, 270] else (0, 0, 255)
		else:
			color = (0, 255, 0)
			mask_region2 = img_scene_mask[:h_scene // 2, :]  # upper side
			offset_y = 0

		cv2.rectangle(img_scene_color, top_left1, bottom_right1, color, 2)
		cv2.line(img_scene_color, figure_center1, rotarGrad(figure_center1[0], figure_center1[1], result1["angle"] + 90, 120), color, 2)
		cv2.putText(img_scene_color, f"T1 {result1['angle']}", (top_left1[0], top_left1[1]-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
		
		result2 = detect_best_template(mask_region2, template2_dir)

		if result2:
			top_left2 = (result2["top_left"][0], result2["top_left"][1] + offset_y)
			w2, h2 = result2["w"], result2["h"]

			bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
			figure_center2 = (int(top_left2[0] + w2/2), int(top_left2[1] + h2/2))
			cv2.rectangle(img_scene_color, top_left2, bottom_right2, (0, 255, 0), 2)
			cv2.line(img_scene_color, figure_center2, rotarGrad(figure_center2[0], figure_center2[1], result2["angle"] + 90, 120), (0, 255, 0), 2)
			cv2.putText(img_scene_color, f"T2 {result2['angle']}", (top_left2[0], top_left2[1]-10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		

		tag_img = filter_tag(img_scene_color)
		contornos, _ = cv2.findContours(tag_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if contornos:
			contorno_max = max(contornos, key=cv2.contourArea)
			x, y, w, h = cv2.boundingRect(contorno_max)
			cv2.rectangle(img_scene_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	cv2.imshow("Deteccion",  cv2.resize(img_scene_color, (720, 960)))
	# cv2.imshow("mask", img_scene_mask)
	if cv2.waitKey(6000 if img_name == "img5" else 3000) == q_unicode:
		break

cv2.destroyAllWindows()
