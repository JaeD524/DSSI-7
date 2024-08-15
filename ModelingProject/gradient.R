library(caret)
library(xgboost)
library(readxl)

# 데이터 로드 및 전처리
data <- read_xlsx("C:/Users/user/Desktop/대학교/2024 여름계절학기/DSSI/대회 준비/my_data.xlsx")
View(data)
data$y_n <- as.factor(data$y_n)

# 데이터 분할
set.seed(1004)
idx <- createDataPartition(data$y_n, p = 0.8, list = FALSE)
X_train <- data1[idx, ]
X_test <- data1[-idx, ]

#독립변수와 종속변수 설정
train_independent_vars <- X_train[,c(1,2,5,7)]
train_dependent_var <- X_train$y_n
test_independent_vars <- X_test[,c(1,2,5,7)]
test_dependent_var <- X_test$y_n

# 독립변수를 숫자형 행렬로 변환
train_independent_matrix <- as.matrix(train_independent_vars)
test_independent_matrix <- as.matrix(test_independent_vars)

# 종속변수를 숫자형 벡터로 변환
train_dependent_vector <- as.numeric(train_dependent_var) - 1
test_dependent_vector <- as.numeric(test_dependent_var) - 1

# str(train_independent_matrix)

# 문자형 데이터를 숫자형으로 변환
train_independent_matrix <- apply(train_independent_matrix, 2, as.numeric)
test_independent_matrix <- apply(test_independent_matrix, 2, as.numeric)

# 문제가 된 열의 데이터 확인
na_indices <- which(is.na(test_independent_matrix), arr.ind = TRUE)
test_independent_vars[na_indices[,1], ]

# xgb.DMatrix 객체 생성
train_matrix <- xgb.DMatrix(data = train_independent_matrix, label = train_dependent_vector)
test_matrix <- xgb.DMatrix(data = test_independent_matrix, label = test_dependent_vector)

# XGBoost 하이퍼파라미터 설정
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 4,
  gamma = 1,
  min_child_weight = 2
)

# 모델 훈련
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix),
  verbose = 1
)

# 모델 요약 출력
print(xgb_model)

# 테스트 데이터에 대한 예측
xgb_pred_prob <- predict(xgb_model, test_matrix)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_pred <- as.factor(xgb_pred)

# xgb_pred와 test_dependent_var를 모두 factor로 변환하고, 동일한 레벨 순서로 설정
xgb_pred <- factor(xgb_pred, levels = levels(test_dependent_var))

# 혼동 행렬 계산
conf_matrix <- confusionMatrix(xgb_pred, test_dependent_var)
print(conf_matrix)

# 정확도 출력
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Accuracy:", accuracy))
