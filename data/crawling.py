from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time
import csv 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import urllib.request
from tqdm import tqdm


# CSV 
data_file = "korean_landmarks.csv"
data = []

with open(data_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        data.append(row)


# 이미지 저장 디렉토리 
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)


# 크롤링 함수
def download_images():

    chromeDriverPath = '/home/sion/chromedriver-linux64/chromedriver'

    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')  # 필요한 경우 추가

    service = Service(chromeDriverPath)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    for landmark in tqdm(data, desc="Processing Landmarks", unit="landmark"):
        name, _, _ = landmark

        print(f"{name} 이미지 크롤링 ...")

        search_query = f"{name} 이미지"
        
        # Google 이미지 검색
        driver.get(f"https://www.google.com/search?q={search_query}&tbm=isch")
        time.sleep(2)  # 페이지 로드 대기


        img_element = driver.find_elements(By.CSS_SELECTOR, ".F0uyec")
        time.sleep(3)

        cnt = 0
    
        for img in img_element:
            if cnt >= 10:
                break

            try:
                img.click()
                driver.implicitly_wait(5)

                img_ele = driver.find_element(By.XPATH, '//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]')

                img_url = img_ele.get_attribute('src')

                urllib.request.urlretrieve(img_url, f"./images/{name}_{cnt+1}.jpg")
                cnt += 1
                
            except:
                pass


# 크롤링 실행
download_images()