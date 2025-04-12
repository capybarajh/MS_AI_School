from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service

import time
import os
import urllib, requests

query = "mango"
service = Service('./chromedriver_win32/chromedriver')
driver = webdriver.Chrome(service=service)

driver.get("https://www.google.com/imghp?hl=ko&tab=ri&ogbl")
keyword = driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea'
    )
keyword.send_keys(query)

driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button'
).click()

print(f"{query} 스크롤 내리는 중...")
elem = driver.find_element_by_tag_name("body")
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try:
    driver.find_element_by_xpath(
        '/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[2]/div[2]/input'
        ).click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except:
    pass

links = []
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
# images = driver.find_elements_by_xpath(
#     "/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]/div[1]/span/div[1]/div[1]/div[*]/div[*]/a[1]/div[1]/img"
# )

for image in images:
    if image.get_attribute('src') != None:
        links.append(image.get_attribute('src'))
    elif image.get_attribute("data-src") != None:
        links.append(image.get_attribute("data-src"))
    elif image.get_attribute("data-iurl") != None:
        links.append(image.get_attribute("data-iurl"))

print(f"{query} 찾은 이미지 개수: {len(links)}")
time.sleep(1)

count = 0
for i in links:
    start = time.time()
    url = i
    os.makedirs(f"./{query}_img_dataset/", exist_ok=True)
    while True:
        try:
            urllib.request.urlretrieve(url, f"./{query}_img_dataset/{count:04}_{query}.png")
            print(f"{count + 1} / {len(links)} / {query} / 다운로드 시간: {time.time() - start} 초")
            break
        except Exception as e:
            print(f"HTTPError 발생 ({e}): 재시도중...")
            time.sleep(5)

        if time.time() - start > 60:
            print(f"{query} 이미지 다운로드 실패")
            break

    count = count + 1

print(f"{query} 다운로드 완료")
driver.close()