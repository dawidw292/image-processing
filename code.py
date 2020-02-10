from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def myConvolve(picture, kernel):
    kRows, kCols = kernel.shape
    kCenterX = int(kCols / 2)
    kCenterY = int(kRows / 2)
    rows, cols = picture.shape
    out = np.zeros(picture.shape)
    m = np.arange(0, kRows)
    n = np.arange(0, kCols)
    picSet = np.array(np.meshgrid(np.arange(0, rows), np.arange(0, cols)))
    s1, s2, s3 = np.shape(picSet)
    picSet = picSet.reshape((s1, s2 * s3))
    allSet = np.array(np.meshgrid(np.arange(0, rows), np.arange(0, cols), m, n))
    s1, s2, s3, s4, s5 = allSet.shape
    allSet = allSet.reshape(s1, s2*s3*s4*s5)
    allSet = np.vstack((allSet, np.zeros((1, s2*s3*s4*s5), dtype = np.int32)))
    ii = allSet[0] + (kCenterY - allSet[2])
    jj = allSet[1] + (kCenterX - allSet[3])
    index = np.logical_and(ii >= 0, ii < rows) * np.logical_and(jj >= 0, jj < cols)
    allSet[4][index] = out[allSet[0][index], allSet[1][index]] + picture[ii[index], jj[index]] * kernel[allSet[2][index], allSet[3][index]]
    values = np.transpose(np.vstack((allSet[0], allSet[1], allSet[4])))
    s1, s2, s3 = np.shape(np.array(np.meshgrid(np.arange(0, rows), np.arange(0, cols))))
    out[picSet[0], picSet[1]] = np.sum(values[:, 2].reshape(s2 * s3, kCols * kRows), axis = 1)

    return out


def generateGauss(size, sigma=2):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel


# grey image
image = Image.open('picture.jpg')
imageA = np.array(image)
greyImageA = 0.2126 * imageA[:, :, 0] + 0.7152 * imageA[:, :, 1] + 0.0722 * imageA[:, :, 2]
greyImageA2 = (greyImageA / 4).astype(int)

fig1 = plt.figure("gray")
plt.imshow(Image.fromarray(greyImageA))

# histogram
values, count = np.unique(greyImageA2, return_counts = True)
fig2 = plt.figure("histogram")
plt.bar(values, count)

# gaussian blur
gauss = generateGauss(9)
blur = myConvolve(greyImageA, gauss)

fig3 = plt.figure("blur")
plt.imshow(Image.fromarray(blur))

# gradient

kX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

sX = myConvolve(blur, kX)
sY = myConvolve(blur, kY)

gradient = np.hypot(sX, sY)
gradient = gradient / gradient.max() * 255
theta = np.arctan2(sY, sX)

fig4 = plt.figure("sobel")
plt.imshow(Image.fromarray(gradient))

# non-maximum suppression
rows, cols = gradient.shape

angle = theta * 180. / np.pi
angle[angle < 0] += 180

supp = np.copy(gradient)

angle0 = np.array(np.where(np.logical_or(np.logical_and(angle >= 0, angle < 22.5), np.logical_and(angle >= 157.5, angle <= 180))))
angle45 = np.array(np.where(np.logical_and(angle >= 22.5, angle < 67.5)))
angle90 = np.array(np.where(np.logical_and(angle >= 67.5, angle < 112.5)))
angle135 = np.array(np.where(np.logical_and(angle >= 112.5, angle < 157.5)))

next0 = np.copy(angle0)
next0[1] = next0[1] + 1
prev0 = np.array(angle0)
prev0[1] = prev0[1] - 1

next45 = np.copy(angle45)
next45[0] = next45[0] + 1
next45[1] = next45[1] - 1
prev45 = np.copy(angle45)
prev45[0] = prev45[0] - 1
prev45[1] = prev45[1] + 1

next90 = np.copy(angle90)
next90[0] = next90[0] + 1
prev90 = np.copy(angle90)
prev90[0] = prev90[0] - 1

next135 = np.copy(angle135)
next135[0] = next135[0] + 1
next135[1] = next135[1] + 1
prev135 = np.copy(angle135)
prev135[0] = prev135[0] - 1
prev135[1] = prev135[1] - 1

index0 = np.logical_and(next0[1] < cols, prev0[1] >= 0)
index45 = np.logical_and(np.logical_and(next45[0] < rows, next45[1] >= 0), np.logical_and(prev45[0] >= 0, prev45[1] < cols))
index90 = np.logical_and(next90[0] < rows, prev90[0] >= 0)
index135 = np.logical_and(np.logical_and(next135[0] < cols, next135[1] < rows), np.logical_and(prev135[0] >= 0, prev135[1] >= cols))

index1 = np.logical_or(supp[angle0[0, index0], angle0[1, index0]] < supp[next0[0, index0], next0[1, index0]], supp[angle0[0, index0], angle0[1, index0]] < supp[prev0[0, index0], prev0[1, index0]])
index2 = np.where(np.logical_or(supp[angle45[0, index45], angle45[1, index45]] < supp[next45[0, index45], next45[1, index45]], supp[angle45[0, index45], angle45[1, index45]] < supp[prev45[0, index45], prev45[1, index45]]))
index3 = np.where(np.logical_or(supp[angle90[0, index90], angle90[1, index90]] < supp[next90[0, index90], next90[1, index90]], supp[angle90[0, index90], angle90[1, index90]] < supp[prev90[0, index90], prev90[1, index90]]))
index4 = np.where(np.logical_or(supp[angle135[0, index135], angle135[1, index135]] < supp[next135[0, index135], next135[1, index135]], supp[angle135[0, index135], angle135[1, index135]] < supp[prev135[0, index135], prev135[1, index135]]))

step = supp[angle0[0, index0], angle0[1, index0]]
step[index1] = 0
supp[angle0[0, index0], angle0[1, index0]] = step

step = supp[angle45[0, index45], angle45[1, index45]]
step[index2] = 0
supp[angle45[0, index45], angle45[1, index45]] = step

step = supp[angle90[0, index90], angle90[1, index90]]
step[index3] = 0
supp[angle90[0, index90], angle90[1, index90]] = step

step = supp[angle135[0, index135], angle135[1, index135]]
step[index4] = 0
supp[angle135[0, index135], angle135[1, index135]] = step

fig5 = plt.figure("suppresion")
plt.imshow(Image.fromarray(supp))

# dobule threshold

lowRat = 0.05
highRat = 0.09

high = supp.max() * highRat
low = high * lowRat

threshold = np.zeros(supp.shape)
weak = np.int32(30)
strong = np.int32(255)

threshold[supp >= high] = strong
threshold[np.logical_and(supp < high, supp >= low)] = weak

fig6 = plt.figure("double treshold")
plt.imshow(Image.fromarray(threshold))

# hysteresis
rows, cols = np.shape(threshold)
hys = np.zeros((rows + 2, cols + 2))
hys[1:rows+1, 1:cols+1] = threshold

hys1 = hys[2:rows+2, 1:cols+1]
hys2 = hys[1:rows+1, 2:cols+2]
hys3 = hys[0:rows, 1:cols+1]
hys4 = hys[1:rows+1, 0:cols]
hys5 = hys[2:rows+2, 0:cols]
hys6 = hys[0:rows, 2:cols+2]
hys7 = hys[2:rows+2, 2:cols+2]
hys8 = hys[0:rows, 0:cols]

hys[1:rows+1, 1:cols+1] = hys[1:rows+1, 1:cols+1] + hys1 + hys2 + hys3 + hys4 + hys5 + hys6 + hys7 + hys8
hys = hys[1:rows+1, 1:cols+1]
hys[hys <= 9*weak] = 0
hys[hys > 9*weak] = strong

fig7 = plt.figure("hysteresis")
plt.imshow(Image.fromarray(hys))

plt.show()
