#! python3
import cv2
import argparse


def blur_face(img, x1, y1, x2, y2, pad):
	ih, iw, _ = img.shape
	w, h = x2-x1, y2-y1
	w, h = int(w*(1+pad/100)), int(h*(1+pad/100))
	x, y = (x1+x2)//2, (y1+y2)//2
	x1, y1 = max(0, x-w//2), max(0, y-h//2)
	x2, y2 = min(iw, x1+w), min(ih, y1+h)
	k = 5+2*(min(w, h)//4)
	img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)
	return img


def get_faces(img, net):
	ih, iw, _ = img.shape
	blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
		104, 117, 123], False, False)

	net.setInput(blob)
	detections = net.forward()
	conf_threshold = 0.5
	faces = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			faces.append(
				(
					int(detections[0, 0, i, 3] * iw),
					int(detections[0, 0, i, 4] * ih),
					int(detections[0, 0, i, 5] * iw),
					int(detections[0, 0, i, 6] * ih),
				)	
			)
	return faces


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"image",
		help="Path to image file"
	)
	args = parser.parse_args()
	img = cv2.imread(args.image, cv2.IMREAD_COLOR)

	modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
	configFile = "deploy.prototxt"
	net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

	faces = get_faces(img, net)

	for face in faces:
		img = blur_face(img, *face, 50)

		res = img
	cv2.imwrite("fo_"+args.image, res)


if __name__ == '__main__':
	main()
