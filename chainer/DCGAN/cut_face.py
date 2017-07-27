import cv2
import numpy as np
import os

root = './konachan/'
save_root = './face/'
img_list = os.listdir(root)
cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

try:
    os.mkdir(save_root)
except OSError:
    pass

len_list = len(img_list)
c = 1
for filename in img_list:
    if c%100 == 0:
        print('cut image: %s (%d/%d)' % (filename, c, len_list))

    try:
        img = cv2.imread(root + filename)
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_g = cv2.equalizeHist(img_g)
    except:
        print("error in %s" % filename)
        continue

    face = cascade.detectMultiScale(img_g, scaleFactor=1.1, minNeighbors=5, minSize=(96, 96))
    n = 0
    for (x, y, w, h) in face:
        img_f = img[y:y+h, x:x+w]
        cv2.imwrite(save_root + str(n) + "_" + filename, img_f)
        n += 1
    c += 1

print('cut images finshed!')

face_list = os.listdir(save_root)
len_data = len(face_list)

print('get %d faces!' % len_data)

data = np.zeros((len_data, 3, 96, 96), dtype=np.uint8)
for i in range(len_data):
    if i%100 == 0:
        print("read face: %d/%d" % (i, len_data))
    data[i] = cv2.resize(cv2.imread(save_root + face_list[i]), (96, 96)).transpose((2,0,1))

np.save('data_face.npy', data)

print('all finished!')
