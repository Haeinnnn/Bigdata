from PIL import Image
import numpy as np
import os
curPath=os.getcwd()
fileName = curPath+"\\ch012\\tower.jpg" # curpath 현재 디렉토리 가져오기
# 1. 이미지 데이터를 Average Hash로 변환하기
def average_hash(fname, size = 16) :
    img = Image.open(fname) # 2. 이미지 데이터 열기
    img = img.convert("L") # 3. 그레이스케일로 변환하기
    img = img.resize((size, size), Image.ANTIALIAS) # 4. 리사이즈하기
    pixel_data = img.getdata() # 5. 픽셀 데이터 가져오기
    pixels = np.array(pixel_data) # 6. Numpy 배열로변환하기
    pixels = pixels.reshape((size, size)) # 7. 2차원 배열로 변환하기
    avg = pixels.mean() # 8. 평균 구하기
    diff = 1 * (pixels > avg) # 평균보다 크면 1, 작으면 0으로 변환하기
    return diff

# 10. 이진 해시로 변환하기
def np2hash(ahash) :
    bhash = []
    for nl in ahash.tolish():
        sl = [str(i) for i in nl]
        s2 = "".join(sl)
        i = int(s2, 2) # 이진수를 정수로 변환하기
        bhash.append("%04x" % i)
    return "".join(bhash)

# Average Hash 출력하기
ahash = average_hash(fileName)
print(ahash)
print(np2hash(ahash))
