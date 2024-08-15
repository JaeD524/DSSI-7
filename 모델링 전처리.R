library(readxl)
library(dplyr)
#유명 지역
popular1 <- read_xlsx("/Users/bagjaeyong/Desktop/대학교/DSSI/last_popular.xlsx")
popular2 <- read_xlsx("/Users/bagjaeyong/Desktop/대학교/DSSI/popular_food.xlsx")
names(popular2) <- names(popular1)


popular <- rbind(popular1, popular2)
View(popular1)
# 600만 개 항목
popular1$가중치 <- 1 / popular1$rank

# 가중치의 합 계산
total_weight <- sum(popular1$가중치)

# 관광지별 방문객 수 계산
popular1$방문객수 <- round((popular1$가중치 / total_weight) * 6000000)

popular2$가중치 <- 1 / popular2$rank

# 가중치의 합 계산
total_weight <- sum(popular2$가중치)

# 관광지별 방문객 수 계산
popular2$방문객수 <- round((popular2$가중치 / total_weight) * 6000000)

#관광객 전기차 현황
electronic_car_path <-"/Users/bagjaeyong/Desktop/대학교/DSSI/한국전력공사_지역별 전기차 현황정보_20230331.csv" 
electronic_D1 <- read.csv(electronic_car_path,fileEncoding="EUC-KR")

entrance1 <- read.csv("/Users/bagjaeyong/Desktop/대학교/DSSI/20240810150421_유입 출발지 - 유출 목적지 분포.csv",fileEncoding = "EUC-KR")
View(entrance1)
entrance <- entrance1[17:32,]
View(entrance)
entrance$electronic_car <- c(6685,705,3504,1025,3889,468,1284,289,357,178,179,173,140,49,29,0)
18832*35.5/100
3343*21.1/100
16765*20.9/100
15070*6.8/100
84533*4.6/100
21278*2.2/100
61123*2.1/100
18041*1.6/100
25495*1.4/100
14823*1.2/100
25535*0.7/100
24676*0.7/100
27840*0.5/100
9761*0.5/100
5864*0.5/100
sum <- (entrance$electronic_car)
#대전 전기차 충전소
charging_station <- read_xlsx("/Users/bagjaeyong/Desktop/대학교/DSSI/daejeon_charging_round.xlsx")
charging_station$Latitude[1]
charging_station$위도
cha
length(charging_station$id)
View(charging_station)
str(charging_station)


# 대전 위 경도

Daejeon_location <- read_xlsx("/Users/bagjaeyong/Desktop/대학교/DSSI/daejeon_loc_round.xlsx")

# 위도 경도 일치한 것은 1, 아닌것은 0
Daejeon_location$y_n <- 0  # 기본값을 0으로 설정

# 조건에 맞는 경우 y_n 컬럼을 1로 변경
Daejeon_location$y_n <- ifelse(
  paste(Daejeon_location$LongitudeRound, Daejeon_location$LatitudeRound) %in% 
    paste(charging_station$LongitudeRound, charging_station$LatitudeRound),
  1,  # 조건에 맞으면 1
  0   # 조건에 맞지 않으면 0
)
View(Daejeon_location)
str(Daejeon_location)
sum(Daejeon_location$y_n==1)


Daejeon_location
  # 다 합치기
final_data
