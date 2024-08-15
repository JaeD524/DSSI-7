library(lightgbm)
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
# na_indices <- which(is.na(test_independent_matrix), arr.ind = TRUE)
test_independent_vars[na_indices[,1], ]

# 데이터 전처리
# 데이터의 클래스 불균형을 해결하기 위해 scale_pos_weight 파라미터를 설정합니다.
pos_weight <- sum(train_dependent_vector == 0) / sum(train_dependent_vector == 1)

# 데이터를 LightGBM에서 사용할 수 있는 형태로 변환
train_matrix_lgb <- lgb.Dataset(data = train_independent_matrix, label = train_dependent_vector)
test_matrix_lgb <- lgb.Dataset(data = test_independent_matrix, label = test_dependent_vector, free_raw_data = FALSE)

# LightGBM 하이퍼파라미터 설정 (튜닝 포함)
params_lgb <- list(
  objective = "binary",
  metric = "binary_logloss",
  learning_rate = 0.05,  # 낮은 학습률
  num_leaves = 31,       # 잎사귀 수
  max_depth = 7,         # 최대 깊이
  min_child_samples = 20, # 자식 노드가 되기 위한 최소 샘플 수
  subsample = 0.8,       # 샘플 비율
  colsample_bytree = 0.8, # 특성 비율
  scale_pos_weight = pos_weight # 클래스 불균형 조정
)

# 모델 훈련
lgb_model <- lgb.train(
  params = params_lgb,
  data = train_matrix_lgb,
  nrounds = 1000, # 더 많은 라운드를 사용하여 성능 향상
  valids = list(test = test_matrix_lgb),
  early_stopping_rounds = 50 # 조기 중단
)

# 모델 요약 출력
print(lgb_model)

# 테스트 데이터에 대한 예측
lgb_pred_prob <- predict(lgb_model, test_independent_matrix)
lgb_pred <- ifelse(lgb_pred_prob > 0.5, 1, 0)
lgb_pred <- as.factor(lgb_pred)

# xgb_pred와 test_dependent_var를 모두 factor로 변환하고, 동일한 레벨 순서로 설정
lgb_pred <- factor(lgb_pred, levels = levels(test_dependent_var))

# 혼동 행렬 계산
conf_matrix_lgb <- confusionMatrix(lgb_pred, test_dependent_var)
print(conf_matrix_lgb)

# 정확도 출력
accuracy_lgb <- conf_matrix_lgb$overall['Accuracy']
print(paste("Accuracy (LightGBM):", accuracy_lgb))