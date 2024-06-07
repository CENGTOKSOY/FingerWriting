import cv2
import numpy as np

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        exit()
    return cap

def create_empty_drawing(frame):
    return np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        return (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    return None

def draw_line(drawing, start_point, end_point, color):
    cv2.line(drawing, start_point, end_point, color, 2)

def toggle_writing(defects, contour):
    global writing
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        area = 0.5 * b * c * np.sin(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
        if area < 1000:
            writing = not writing
            print("Yazı yazma durumu:", "Başla" if writing else "Dur")
            break

def clear_screen(key, drawing):
    if key == ord('c'):
        drawing[:] = 0
        print("Ekran temizlendi.")

def change_color(key):
    global color
    if key == ord('r'):
        color = (0, 0, 255) # Kırmızı
    elif key == ord('g'):
        color = (0, 255, 0) # Yeşil
    elif key == ord('b'):
        color = (255, 0, 0) # Mavi
    print(f"Kalem rengi değiştirildi: {color}")

# HSV'de el rengi aralığı
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
kernel = np.ones((3, 3), np.uint8)

writing = False
last_point = None
color = (255, 255, 255)  # Beyaz başlangıç rengi

cap = initialize_camera()
ret, frame = cap.read()
if not ret:
    print("Kamera görüntüsü okunamıyor, program sonlandırılıyor.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

drawing = create_empty_drawing(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü okunamıyor, döngüden çıkılıyor.")
        break

    mask = process_frame(frame)
    largest_contour = find_largest_contour(mask)

   
 if largest_contour is not None:
        center = get_contour_center(largest_contour)
        if center:
            if writing and last_point:
                draw_line(drawing, last_point, center, color)
            last_point = center

            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)
            if defects is not None:
                toggle_writing(defects, largest_contour)


    combined = cv2.add(frame, drawing)
    cv2.imshow('frame', combined)
    cv2.imshow('drawing', drawing)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    clear_screen(key, drawing)
    change_color(key)


