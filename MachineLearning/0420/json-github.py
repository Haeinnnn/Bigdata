import urllib.request as req
import os.path, random
import json

# 1. JSON 데이터 내려받기
url = "https://api.github.com/repositories"
savename = "repo.json"
if not os.path.exists(savename):
    req.urlretrieve(url, savename)
    
# 2. JSON 파일 분석하기
items = json.load(open(savename, "r", encoding="utf-8"))
# 또는
# s = open(savename, "r", encoding="utf-8").read()
# items = json.loads(s)

# 3. 출력하기
for item in items:
    print(item["name"] + " - " + item["owner"]["login"])
