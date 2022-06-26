#UFUK PALPAS 21702958
import os
import cv2
import numpy as np

fishbowl_dp = "fishbowl"
goldengate_dp = "goldengate"
fishbowl_imgs = [] #13 images
goldengate_imgs = [] #6 images

f = open('fishBowlImageNames.txt')
lines = [line.strip() for line in f.readlines()]

f2 = open('goldengateImageNames.txt')
lines2 = [line.strip() for line in f2.readlines()]

for root,dirs, files in os.walk(fishbowl_dp):
    for file in files:
        path = os.path.join(root,file)
        for l in lines:
            if l == file:
                fishbowl_imgs.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

fishbowl_imgs.reverse()
for root,dirs, files in os.walk(goldengate_dp):
    for file in files:
        path = os.path.join(root,file)
        for l in lines2:
            if l == file:
                goldengate_imgs.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

def local_features(imgArr):
    i = 0
    keyAndDes = []
    while i < len(imgArr):
        sift = cv2.SIFT_create()
        keyPoints, descriptors = sift.detectAndCompute(imgArr[i], None)
        keyAndDes.append([keyPoints, descriptors])
        i += 1
    return keyAndDes

f_keyAndDesc = local_features(fishbowl_imgs) #x,y = keyPoints[0].pt ,angle = orientation, octave = scale
g_keyAndDesc = local_features(goldengate_imgs)

#image=cv2.drawKeypoints(fishbowl_imgs[0],f_keyAndDesc[0][0],fishbowl_imgs[0],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('fishbowl_sift_keypoints.jpg',image)

# 3.3 part 2 buraya gelecek

bowlIndex = len(fishbowl_imgs) - 1
gateIndex = len(goldengate_imgs) - 1
if bowlIndex < 0 or gateIndex < 0:
    print("Please add image names to both txts")
    exit()
bowlMiddleImg = fishbowl_imgs[bowlIndex]
gateMiddleImg = goldengate_imgs[gateIndex]

def feature_match(descriptor_1,descriptor_2,ratio=0.5):
    m1=[]
    m2=[]
    for i in range(descriptor_1.shape[0]):
        if np.std(descriptor_1[i,:])!=0:
            des=descriptor_2-descriptor_1[i,:]
            des=np.linalg.norm(des, axis=1)
            orders =np.argsort(des).tolist()
            if des[orders[0]]/des[orders[1]]<=ratio:
                m1.append((i,orders[0]))
    for i in range(descriptor_2.shape[0]):
        if np.std(descriptor_2[i,:])!=0:
            des=descriptor_1-descriptor_2[i,:]
            des=np.linalg.norm(des, axis=1)
            orders =np.argsort(des).tolist()
            if des[orders[0]]/des[orders[1]]<=ratio:
                m2.append((orders[0],i))     
    match = list(set(m1).intersection(set(m2)))
    return match
    
def stitchImgs(ran, index, imgs, middleimg, gateOrFish, rightOrNot=False):
    if rightOrNot:
        getRight = True
    else:
        getRight = False
    newKeyAndDesc = None
    useDesc = f_keyAndDesc[index]
    currentRightIndex = index + 1
    currentLeftIndex = index - 1
    if gateOrFish:    
        stickedName = "goldengate"
        useDesc = g_keyAndDesc[index]
        keyAndDesc = g_keyAndDesc
    else:
        stickedName = "fishbowl"
        keyAndDesc = f_keyAndDesc
    for i in range(ran):
        if getRight:
            if currentRightIndex >= 13:
                break
            if i != 0:
                useDesc = newKeyAndDesc[0]
            print("Adding right| image no:", currentRightIndex)
            img = feature_match(useDesc[1],keyAndDesc[currentRightIndex][1],ratio=0.5)
            matches = [cv2.DMatch(i[0],i[1],1) for i in img]
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                points1[i, :] = useDesc[0][match.queryIdx].pt
                points2[i, :] = keyAndDesc[currentRightIndex][0][match.trainIdx].pt
            ##RANSAC for alignment
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            stitch = cv2.warpPerspective(imgs[currentRightIndex], h, (middleimg.shape[1] + imgs[currentRightIndex].shape[1], middleimg.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
            name = stickedName + "alignmentRight.jpeg"
            cv2.imwrite(name, stitch)
            for i in range(img[currentRightIndex].shape[0]):
                for j in range(img[currentRightIndex].shape[1]):
                    if stitch[i,j] == 0:
                        stitch[i,j] = middleimg[i,j]
                    else:
                        stitch[i,j] = (int(middleimg[i,j]) + int(stitch[i,j]))/2                 
            name = stickedName + "_Stiched.jpeg"  
            cv2.imwrite(name, stitch)
            newKeyAndDesc = local_features([stitch])
            image=cv2.drawKeypoints(stitch,newKeyAndDesc[0][0],stitch,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            name = stickedName + 'sift_keypoints.jpg'
            cv2.imwrite(name,image)
            middleimg = stitch
            currentRightIndex += 1 
            if currentLeftIndex >= 0: 
                getRight = True        
        else:
            if currentLeftIndex < 0:
                break
            print("Adding left| image no: ", currentLeftIndex)
            if i != 0:
                useDesc = newKeyAndDesc[0]
            img = feature_match(keyAndDesc[currentLeftIndex][1], useDesc[1], ratio=0.5)
            matches = [cv2.DMatch(i[0],i[1],1) for i in img]
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                points1[i, :] = keyAndDesc[currentLeftIndex][0][match.queryIdx].pt
                points2[i, :] = useDesc[0][match.trainIdx].pt

            ##RANSAC for alignment
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            stitch = cv2.warpPerspective(middleimg, h, (imgs[currentLeftIndex].shape[1] + middleimg.shape[1], imgs[currentLeftIndex].shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
            name = stickedName + "alignment.jpeg"
            cv2.imwrite(name, stitch)

            print(middleimg.shape[0], middleimg.shape[1], imgs[currentLeftIndex].shape[0], imgs[index - 1].shape[1])

            for i in range(middleimg.shape[0]):
                for j in range(middleimg.shape[1]):
                    if j < imgs[currentLeftIndex].shape[1]:
                        if stitch[i,j] == 0:
                            stitch[i,j] = imgs[currentLeftIndex][i,j]
                        else:
                            stitch[i,j] = (int(imgs[currentLeftIndex][i,j]) + int(stitch[i,j]))/2    
            name = stickedName + "_Stiched.jpeg"
            cv2.imwrite(name, stitch)
            newKeyAndDesc = local_features([stitch])
            image=cv2.drawKeypoints(stitch,newKeyAndDesc[0][0],stitch,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            name = stickedName + '_sift_keypoints_1.jpg'
            cv2.imwrite(name,image)
            middleimg = stitch
            currentLeftIndex -= 1
            if currentRightIndex < 13:
                getRight = False

stitchImgs(bowlIndex, bowlIndex, fishbowl_imgs, bowlMiddleImg, False, rightOrNot=False)
stitchImgs(gateIndex, gateIndex, goldengate_imgs, gateMiddleImg, True, rightOrNot=False)