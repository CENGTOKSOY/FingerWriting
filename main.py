import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# El tanıma için renk aralığını ayarla (HSV formatında)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Ekrana yazı yazmak için boş bir görüntü oluştur
ret, test_frame = cap.read()
if ret:
    drawing = np.zeros((test_frame.shape[0], test_frame.shape[1], 3), np.uint8)
else:
    print("Kamera görüntüsü alınamıyor, lütfen kamera bağlantısını kontrol edin.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Yazı yazma durumunu kontrol etmek için bir değişken
writing = False
last_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü okunamıyor, döngüden çıkılıyor.")
        break

    frame = cv2.flip(frame, 1)

    # Renk uzayını BGR'dan HSV'ye dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # El rengi için maske oluştur
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Gürültüyü azalt ve el sınırlarını belirginleştir
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Konturları bul ve en büyüğünü seç
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # El merkezini bul
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Eğer yazı yazma modu aktifse ve bir önceki nokta varsa, çizgi çiz
            if writing and last_point is not None:
                cv2.line(drawing, last_point, (cx, cy), (255, 255, 255), 2)
            last_point = (cx, cy)

            # Parmak uçlarını algılamak için konturun dışbükey kabuğunu hesapla
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)

            # Başparmak ve işaret parmağının birleşip birleşmediğini kontrol et
            # ...
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(largest_contour[s][0])
                    end = tuple(largest_contour[e][0])
                    far = tuple(largest_contour[f][0])
                    # Üçgen alanını hesapla
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    area = 0.5 * b * c * np.sin(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                    # Eğer alan belirli bir eşiğin altındaysa, parmaklar birleşmiş demektir
                    if area < 1000:
                        writing = not writing  # Yazı yazma durumunu değiştir
                        if writing:
                            print("Yazı yazmaya başla.")
                        else:
                            print("Yazı yazmayı durdur.")
                        break


    # Çizim ekranını ana görüntüye ekle
    combined = cv2.add(frame, drawing)

    # Görüntüleri göster
    cv2.imshow('frame', combined)
    cv2.imshow('drawing', drawing)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
