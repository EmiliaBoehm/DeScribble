# Dokumentation / Notizen zu OpenCV
https://docs.opencv.org/4.8.0/d6/d00/tutorial_py_root.html
```python
import cv2
```

## VErfahren: Boxing, dann masking

https://stackoverflow.com/questions/57858944/opencv-python-border-removal-preprocessing-for-ocr

```
import cv2
import numpy as np

image = cv2.imread('1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = np.zeros(image.shape, dtype=np.uint8)

cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cv2.fillPoly(mask, cnts, [255,255,255])
mask = 255 - mask
result = cv2.bitwise_or(image, mask)

cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey(0)
```

## Dilation = Streckung, Dilatation
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

 - Erosion = Noise entfernen, schrumpfen
 - Dilation = Strecken
 - Opening = Erst erosion, dann dilation
 - Closing = Erst dilation, dann erosion 
 - Siehe Link oben für weitere Operationen. 

Für all diese Transformationen braucht es einen "Kernel", also eine
Matrix, die auf das Bild faltend angewendet wird. Um eine
vorinitialisierte Matrix zu erhalten, verwendet man
`getStructuringElement` mit typischen Formen:

  - `cv.MORPH_RECT`
  - `cv.MORPH_ELLIPSE`
  - `cv.MORPH_CROSS`
 
 Zur Wahl des Kernels, siehe
 https://stackoverflow.com/questions/67117928/how-to-decide-on-the-kernel-to-use-for-dilations-opencv-python
 
## Bilder Größe herausfinden

```
img.shape
```

## Bilder sind "mat arrays"
https://docs.opencv.org/4.8.0/d3/d63/classcv_1_1Mat.html#details

> The class Mat represents an n-dimensional dense numerical
> single-channel or multi-channel array. It can be used to store real
> or complex-valued vectors and matrices, grayscale or color images,
> voxel volumes, vector fields, point clouds, tensors, histograms
> (though, very high-dimensional histograms may be better stored in a
> SparseMat ).

## Rechteck zeichnen
https://docs.opencv.org/4.8.0/dc/da5/tutorial_py_drawing_functions.html

```python
cv.rectangle(img, left-top-corner, bottom-right-corner, color-tuple, thickness)
```

## Thresholding
https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

`cv.threshold(img-in-grayscale!, schwellenwert, max, cv.THRESH_....)`

 - Bild in Grauwerte, da nur dann werte von 0-255
 - Schwellenwert und Maximalwert 0-255.
 - ACHTUNG: Gibt ein TUPEL zurück: `(ret, img)`. `ret` ist der
   benutzte Schwellenwert. 
 

 - Mögliche Flags:
    - `cv.THRESH_BINARY`, `cv.THRESH_BINARY_INV`: Unter Schwellenwert
      ist 0, drüber ist 1; bzw. umgekehrt.
    - `cv.ADAPTIVE_THRESH_MEAN_C` - Schwellenwert regional; siehe Dokumentation.
    - `cv.ADAPTIVE_THRESH_GAUSSIAN_C`
    - `cv.THRESH_OTSU` - Klingt am besten.

## Farbkonversion mit cvtColor()

`cvtColor(src, dest, code, nChannels)`

 - Konversion wird bestimmt mit dem Flag cv.COLOR_.....
   - cv.COLOR_BGR2GRAY -> BGR zu Grauskalen
   - Einlesen mit  `cv2.imread` per default BGR.
   - "BGR" ist RGB mit anderer Reihenfolge (blau zuerst)
   - In Pillow, the order of colors is assumed to be RGB (red, green,
     blue).
 - nChannels kann ausgelassen werden, wird dann automatisch bestimmt. 

## Konturen finden: cv2.findContours
https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

 - VERÄNDERT das Bild in OpenCV < 3.2! Also eine Kopie übergeben.
 - Rückgabewerte: `contours, hierarchy`
   - "contours" ist eine Liste von Punkten (x,y) (numpy arrays). Liste
     kann mit `drawContours` gezeichnet werden.
   - Zur hierarchie: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
 - Erkennt die 'weißen' Konturen; Bild muss also invertiert
   binarisiert werden.
   
 
## Konturen finden: convexHull?

https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/

Sieht ganz interessant aus

## Bilder schreiben
```python
cv2.imwrite(path, mat-array)
```
## cv2.imshow()
https://www.delftstack.com/howto/python/opencv-imshow-in-python/

### Ein Bild anzeigen
```python
import cv2

img = cv2.imread("deftstack.png")
window_name = "Image title"
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# oder cv2.destroyWindow(window_name)
```

### Ein numpy Array anzeigen

300 x 300 x 3 (=300x300 RGB) Array

```python-mode
import cv2
import numpy as np

# Fill numpy array with 125 = #7d7d7d
ndarray = np.full((300,300,3), 125, dtype=np.uint8)
 
# Show image
cv2.imshow('Example - Show image in window', ndarray)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
```

