from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time
import csv 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


# CSV 파일에서 문화재 데이터 읽기
data_file = "korean_landmarks.csv"
data = []

with open(data_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        data.append(row)


# 이미지 저장 디렉토리 생성
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Selenium을 이용한 이미지 크롤링
def download_images():
    # ChromeDriver 절대 경로 설정
    chromeDriverPath = '/home/sion/chromedriver-linux64/chromedriver'

    # 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')  # 필요한 경우 추가

    # ChromeDriver 초기화
    service = Service(chromeDriverPath)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    for landmark in data:
        name, location, description = landmark
        search_query = f"{name} {location}"
        
        # Google 이미지 검색
        driver.get(f"https://www.google.com/search?q={search_query}&tbm=isch")
        time.sleep(2)  # 페이지 로드 대기


        images = driver.find_elements(By.CSS_SELECTOR, ".F0uyec")
        links = [image.get_attribute('src') for image in images if image.get_attribute('src') is not None]
        print('찾은 이미지의 개수 : ', len(links))


#         # 이미지 요소 수집
#         images = driver.find_elements(By.CSS_SELECTOR, "img")
#         count = 0

#         for img in images:
#             if count >= 10:
#                 break

#             try:
#                 src = img.get_attribute("src")
#                 if src and src.startswith("http"):
#                     # 이미지 저장
#                     file_name = f"{name}_{count + 1}.jpg"
#                     file_path = os.path.join(output_dir, file_name)
                    
#                     with open(file_path, "wb") as file:
#                         file.write(driver.execute_script("return fetch(arguments[0]).then(res => res.arrayBuffer()).then(buff => new Uint8Array(buff));", src))

#                     print(f"Saved {file_name}")
#                     count += 1
#             except Exception as e:
#                 print(f"Error downloading image for {name}: {e}")

#     driver.quit()

# 크롤링 실행
download_images()