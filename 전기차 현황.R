install.packages("extrafont")
library(extrafont)
font_import()
library(ggplot2)
theme_set(theme_grey(base_family="AppleGothic"))


electronic_car_path <-"/Users/bagjaeyong/Desktop/대학교/DSSI/한국전력공사_지역별 전기차 현황정보_20230331.csv" 
electronic_car <- read.csv(electronic_car_path,fileEncoding="EUC-KR")
electronic_car$Date <- as.Date(electronic_car$기준일)
str(electronic_car)
ggplot(electronic_car,aes(x=Date,y=합계))+geom_line()+labs(x="날짜",y="전기차 수",title="국내 전기차 현황")


