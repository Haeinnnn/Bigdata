from PIL import Image # 비트맵 형태로 처리해주는거
import numpy as np
import os, re

curPath=os.getcwd()

# 파일 경로 지정하기
# 이따가 와서 치기!
search_dir = curPath+"\\ch012\\image\\101_ObjectCategories"
cache_dir = curPath+"\\ch012\\image\\cache_avhash"
html_file = curPath+"\\ch012\\avhash_search_output.html"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
# 1. 이미지 데이터를 Average Hash 로 변환하기
def average_hash(fname, size = 16):
    fname2 = fname[len(search_dir):]
    # 이미지 캐시하기
    cache_file = cache_dir + "\\" + fname2.replace('\\', '_') + ".csv"
    if not os.path.exists(cache_file): # 해시 생성하기
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.ANTIALIAS)
        pixels = np.array(img.getdata()).reshape((size, size))
        avg = pixels.mean()
        px = 1 * (pixels > avg)
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else: # 캐시돼 있다면 읽지 않기
        px = np.loadtxt(cache_file, delimiter=",")
    return px

# 2. 해밍 거리 구하기
def hamming_dist(a, b):
    aa = a.reshape(1, -1) # 1차원 배열로 변환하기
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist
# 3. 모든 폴더에 처리 적용하기
def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root, f)
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname
# 4. 이미지 찾기
def find_image(fname, rate) :
    src = average_hash(fname)
    for fname in enum_all_files(search_dir):
        dst = average_hash(fname)
        diff_r = hamming_dist(src, dst) / 256
        # print("[check] ", fname)
        if diff_r < rate:
            yield (diff_r, fname)
# 5. 찾기
srcfile = search_dir + "\\chair\\image_0016.jpg"
html = ""
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x:x[0])
for r, f in sim:
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>'+ \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>'+ \
        '</a></p></div>'
    html += s
# HTML로 출력하기
html = """<html><head><meta charset="utf8"></head>
<body><h3> 원래 이미지 </h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html)
print("OK")
