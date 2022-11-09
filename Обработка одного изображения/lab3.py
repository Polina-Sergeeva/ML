import numpy as np
import cv2
import matplotlib.pyplot as plt

# Загрузка изображения в цветовом пространстве RGB
original_image = cv2.imread("C:/a/test2.jpeg")

# Преобразование цветового пространства RGB в HSV
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))

# Преобразовать np.float32
vectorized = np.float32(vectorized)

# Здесь мы применяем кластеризацию k-средних, чтобы пиксели вокруг цвета были согласованы и давали одинаковые значения RGB/HSV.
# определить критерии, количество кластеров и применить kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 10
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

# Теперь конвертируем обратно в uint8
# теперь нам нужно получить доступ к меткам, чтобы восстановить кластеризованное изображение
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#res2 — это результат кадра, подвергшегося кластеризации методом k-средних.
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(res2)
plt.title('K = %i' % K), plt.xticks([]), plt.yticks([])

#обнаружение краев
edges = cv2.Canny(img,100,200)
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()