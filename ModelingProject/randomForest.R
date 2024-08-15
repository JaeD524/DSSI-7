# 필요한 패키지 로드
library(readxl)
library(dplyr)
library(randomForest)
library(caret)

# 데이터 로드
data <- read_xlsx("C:/Users/user/Desktop/대학교/2024 여름계절학기/DSSI/대회 준비/final_data.xlsx")

# 데이터 확인 및 전처리
View(data)
str(data)

# 종속 변수 y_n을 factor로 변환
data$y_n <- as.factor(data$y_n)

# 데이터 분할 (90% 훈련 데이터, 10% 테스트 데이터)
set.seed(1004)
idx <- createDataPartition(data$y_n, p = 0.9, list = FALSE)
X_train <- data[idx, ]
X_test <- data[-idx, ]

# 모델 훈련
ctrl <- trainControl(method = "cv", number = 10)
rf_model <- train(
  x = X_train %>% select(-y_n), 
  y = X_train$y_n,
  method = "rf",
  trControl = ctrl,
  tuneLength = 30
)

# 모델 요약 출력
print(rf_model)

# 모델 예측 및 평가
rf_pred <- predict(rf_model, X_test %>% select(-y_n))
Acc1 <- confusionMatrix(rf_pred, X_test$y_n)
print(Acc1)
