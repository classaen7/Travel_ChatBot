import csv

# 문화재 및 유명 건축물 데이터 준비
data = [
    ["경복궁", "서울특별시 종로구 세종로", "조선시대의 대표적인 궁궐로, 광화문과 근정전이 유명"],
    ["창덕궁", "서울특별시 종로구 와룡동", "유네스코 세계유산에 등재된 조선시대 궁궐"],
    ["덕수궁", "서울특별시 중구 정동", "대한제국의 역사를 간직한 궁궐로 석조전이 인상적"],
    ["창경궁", "서울특별시 종로구 창경궁로", "조선의 왕비를 위해 건립된 궁궐"],
    ["경희궁", "서울특별시 종로구 새문안로", "조선 후기의 별궁으로 사용된 궁궐"],
    ["불국사", "경상북도 경주시 진현동", "석굴암과 함께 유네스코 세계유산으로 등재된 사찰"],
    ["석굴암", "경상북도 경주시 불국로", "불교 예술의 정수로 꼽히는 석굴 사원"],
    ["해인사", "경상남도 합천군 가야면", "팔만대장경을 보관하고 있는 사찰"],
    ["통도사", "경상남도 양산시 하북면", "부처님의 진신사리를 모신 사찰"],
    ["화엄사", "전라남도 구례군 마산면", "백제 시대에 창건된 아름다운 사찰"],
    ["송광사", "전라남도 순천시 송광면", "한국 불교의 3보사찰 중 하나"],
    ["봉정사", "경상북도 안동시 서후면", "가장 오래된 목조 건축물인 극락전이 있는 사찰"],
    ["수원 화성", "경기도 수원시 장안구", "조선 정조 시대에 건립된 군사 방어 시설"],
    ["남한산성", "경기도 광주시 남한산성면", "한국의 전통 산성으로 유네스코 세계유산"],
    ["고창 고인돌 유적", "전라북도 고창군 고창읍", "선사 시대의 대표적인 고인돌 유적"],
    ["제주 성산일출봉", "제주특별자치도 서귀포시 성산읍", "제주도의 상징적인 화산 지형"],
    ["한라산", "제주특별자치도 제주시", "제주도의 중심에 있는 화산으로 최고봉"],
    ["남산타워", "서울특별시 용산구 남산공원길", "서울의 랜드마크로 야경이 아름다움"],
    ["롯데타워", "서울특별시 송파구 올림픽로", "한국에서 가장 높은 건축물"],
    ["부산 타워", "부산광역시 중구 용두산길", "부산을 대표하는 전망 타워"],
    ["광안대교", "부산광역시 수영구", "부산의 해상 교량으로 야경이 유명"],
    ["독립문", "서울특별시 서대문구 현저동", "한국의 독립운동을 기념하는 문"],
    ["흥인지문", "서울특별시 종로구 흥인지문로", "서울의 동쪽 성문으로 동대문으로도 불림"],
    ["남대문", "서울특별시 중구 세종대로", "서울의 대표적인 성문으로 숭례문으로도 불림"],
    ["한옥마을", "전라북도 전주시 완산구", "전통 한옥의 아름다움을 간직한 마을"],
    ["강릉 선교장", "강원도 강릉시 운정길", "조선 시대의 양반 주택으로 유명"],
    ["오죽헌", "강원도 강릉시 율곡로", "율곡 이이가 태어난 역사적인 장소"],
    ["양동마을", "경상북도 경주시 강동면", "유네스코 세계유산에 등재된 전통 마을"],
    ["하회마을", "경상북도 안동시 풍천면", "한국 전통 문화를 간직한 마을"],
    ["대구 팔공산", "대구광역시 동구 팔공산로", "대구의 대표적인 자연 명소"],
    ["속초 설악산", "강원도 속초시 설악산로", "한국의 대표적인 산악 관광지"],
    ["춘천 남이섬", "강원도 춘천시 남산면", "드라마 촬영지로 유명한 관광지"],
    ["경주 대릉원", "경상북도 경주시 황남동", "신라 시대의 고분 유적지"],
    ["월정교", "경상북도 경주시 교동", "복원된 신라 시대의 아름다운 교량"],
    ["안압지", "경상북도 경주시 원화로", "신라 왕궁의 연못으로 야경이 아름다움"],
    ["전주 경기전", "전라북도 전주시 완산구", "조선 태조 이성계의 초상화를 모신 전각"],
    ["판문점", "경기도 파주시 군내면", "남북 분단의 상징적인 장소"],
    ["인천 차이나타운", "인천광역시 중구 차이나타운로", "중국 문화가 어우러진 특색 있는 거리"],
    ["설악산 케이블카", "강원도 속초시 설악산로", "설악산의 풍경을 한눈에 볼 수 있는 케이블카"],
    ["강릉 경포대", "강원도 강릉시 경포로", "동해의 아름다운 해변과 함께하는 명소"],
    ["서울 올림픽 공원", "서울특별시 송파구 올림픽로", "1988년 올림픽을 기념하는 공원"],
    ["부산 해운대", "부산광역시 해운대구", "한국에서 가장 유명한 해변"],
    ["제주 오름", "제주특별자치도 제주시", "제주도의 독특한 지형인 오름들"],
    ["울릉도 독도", "경상북도 울릉군 울릉읍", "한국 영토의 동쪽 끝 섬"],
    ["광주 국립아시아문화전당", "광주광역시 동구 문화전당로", "아시아 문화를 교류하는 복합 문화 공간"],
    ["익산 미륵사지", "전라북도 익산시 금마면", "백제 시대의 거대한 사찰 유적"],
    ["강화도 고려궁지", "인천광역시 강화군 강화읍", "고려 왕조의 잠시 수도였던 역사적 장소"],
    ["진도 운림산방", "전라남도 진도군 의신면", "남도 화가들의 작업 공간이었던 곳"],
    ["청주 상당산성", "충청북도 청주시 상당구", "조선 시대의 군사 방어 시설"],
    ["태백산", "강원도 태백시", "한국의 대표적인 명산 중 하나"],
    ["울산 간절곶", "울산광역시 울주군", "해돋이가 아름다운 명소로 유명"],
]

# CSV 파일로 저장
with open("korean_landmarks.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["이름", "지역", "설명"])
    writer.writerows(data)