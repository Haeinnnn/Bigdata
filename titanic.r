# 1. 데이터 불러오기
titanic = read.csv("C:/BigData/work/R_Project/ch011/titanic3.csv") # 파일 읽어오기
titanic <- titanic[, !names(titanic) %in% c("home.dest", "boat", "body")]
# 불필요한 컬럼 삭제
str(titanic) # 변수의 속성과 길이, 그리고 미리보기 값 출력

# 2. 데이터 타입 지정
titanic$pclass <- as.factor(titanic$pclass) # pclass(int)의 타입을 범주형인 factor로 바꿈 (정순데 연산관계존재x)
titanic$name <- as.character(titanic$name) # name(factor)의 타입을 문자열 char로 바꿈 (레벨의 수가 많아서)
titanic$ticket <- as.character(titanic$ticket) # 위와 동일하게 단순문자열 char로 바꿈
titanic$cabin <- as.character(titanic$cabin) # 위와 동일하게 단순문자열 char로 바꿈
titanic$survived <- factor(titanic$survived, levels=c(0,1), labels=c("dead", "survived"))
# survived(int)의 타입을 범주형인 factor로 바꿈 (정수로 두면 회귀분석 알고리즘을 수행함)
str(titanic) # 변수의 속성과 길이, 미리보기 값을 출력
titanic$sex <- as.factor(titanic$sex)
titanic$embarked <- as.factor(titanic$embarked)
# !! 책이랑 다른거 오류 수정해줌 위에 이렇게 두줄 추가해주기 !!

levels(titanic$embarked) # 미리 정의된 레벨 알아보기
table(titanic$embarked) # 데이터의 빈도 수를 보여줌 "" 값이 2개 존재함

levels(titanic$embarked) [1] <- NA # "" 데이터를 NA로 바꿔줌
table(titanic$embarked, useNA = "always") # 데이터의 빈도수를 보여줌 ""가 NA로 바뀜

titanic$cabin <- ifelse(titanic$cabin == "", NA, titanic$cabin) # cabin의 빈 문자열도 바꿈
# 문자열이라 직접 수정함
str(titanic) # 수정된내용 확인

# 3. 테스트 데이터의 분리
# createDataPartition를 사용해 y값을 고려한 훈련과 테스트 데이터의 분할을 할 수 있다
library(caret)
set.seed(137) # 항상 같은 데이터가 훈련데이터와 테스트로 분리될 수 있게
test_idx <- createDataPartition(titanic$survived, p=0.1)$Resample1 # 생존:사망자 일정하게유지
titanic.test <- titanic[test_idx, ]
titanic.train <- titanic[-test_idx, ] # 남은 값은 훈련데이터로
NROW(titanic.test) # 행의 수 출력
prop.table(table(titanic.test$survived)) # 비율로 테이블을 출력해줌
NROW(titanic.train)
prop.table(table(titanic.train$survived))

save(titanic, titanic.test, titanic.train, file="titanic.RData") # 사용이 편리하도록 저장

# 4. 교차 검증 준비
# 10겹 교차 검증을 수행하기로 하고 caret 패키지의 createFolds()를 사용해 분리
createFolds(titanic.train$survived, k=10) # 데이터 번호가 저장된 10개의 fold가 생성됨

create_ten_fold_cv <- function() { # 10겹 교차 데이터를 만드는 함수
  set.seed(137) # 항상 같은 결과가 나올 목적
  lapply(createFolds(titanic.train$survived, k=10), function(idx) {
    return(list(train=titanic.train[-idx, ],
                validation=titanic.train[idx, ]))
  })
}

x <- create_ten_fold_cv() # 방금 만든 함수를 사용
str(x) # 변수의 속성과 길이, 미리보기 값을 출력
head(x$Fold01$train) # 값이 짤리므로 훈련데이터 직접 가져오기 # 또는 x1$train 첫번째폴드호출

# 5. 데이터 탐색
# 모델을 작성하기 전 데이터의 모습을 알면 모델을 세울때 많이 도움이 된다
# 또한 데이터를 불러들일 때 혹시 에러가 있지 않았는지도 알게 된다
library(Hmisc)

data <- create_ten_fold_cv()[[1]]$train
# 교차 검증의 첫 번째 분할에서의 훈련 데이터를 사용한 데이터 탐색
summary(survived ~ pclass + sex + age + sibsp +
        parch + fare + embarked, data=data, method="reverse")
# method="reverse"는 종속 변수 survived에 따라 독립 변수들을 분할하여 보여줌
# 표가 생성됨 pclass, sex, age ... 등 변숫값에 따라 생존율의 분포가 나온다

# caret::featurePlot()을 사용해 데이터를 시각화하기
# 생존 또는 사망여부를 Y, 예측 변수를 X로 해서 호출함
# featurePlot() 사용시 NA 값이 있으면 결과값이 정확치않아 complete.cases()를 써서 테스트함
data <- create_ten_fold_cv()[[1]]$train
data.complete <- data[complete.cases(data), ]
featurePlot(
    data.complete[,
        sapply(names(data.complete),
             function(n) { is.numeric(data.complete [, n]) })],
    data.complete [, c("survived")], "ellipse")

# sapply()의 names는 data.complete의 컬럼명을 반환한다
# (pclass, survived,, name, sex ... 등을)
# sapply는 이 각각의 이름을 function()에 n으로 넘김
# 호출된 함수는 data.complete에서 해당 컬럼이 숫자형데이터인지를 is.number()로 테스트해서 반환
# 따라서 sapply()의 최종 결과는 TRUE또는 FALSE가 저장된 벡터

# install.packages("ellipse")
# library(ellipse)
# !! 설치가 안되어서 제대료 표가 안나왔었음!!

mosaicplot(survived ~ pclass + sex, data=data, color=TRUE, main="pclass and sex")
# 팩터 데이터 타입의 차트에 모자이크 플룻을 사용 가능

xtabs(~ sex + pclass, data=data) # xtabs로 분할표작성  plcass, sex의 전체 탑승자수

xtabs(survived == "survived" ~ sex + pclass, data=data) # survived 컬럼의값으로 생존자 수

xtabs(survived == "survived" ~ sex + pclass, data=data) / xtabs( ~ sex + pclass, data=data)
# 두 결과를 조합해 생존율을 구하기 생존자 수 / 탑승자 수

# 6. 평가 메트릭
# 생존 여부 예측 모델의 성능은 정확도로 하기로 한다
# 정확도는 예측한 값이 true든 false든 상관없이 정확히 예측한 값의 비율을 뜻한다
predicted <- c(1, 0, 0, 1, 1) # ex) 예측값으로 predicted를 씀
actual <- c(1, 0, 0, 0, 0) # ex) 실제값으로 actual을 씀
sum (predicted == actual) / NROW(predicted) # 실제값예측값이 같은것의 합 / 행의 수

# 7. 의사결정 나무 모델
# 기계 학습 모델을 데이터에 적용해 모델을 만들어본다
# 의사결정 나무 모델은 다양한 변수의 상호 작용을 잘 표현해준다
# 사용이 편리하고 변수 간 상호 작용을 잘 다뤄주는 모델인 의사결정나무모델 rpart와
# 조건부 추론 나무 ctree를 모델로 사용한다 (NA 값을 rpart가 대리변수로 처리해줌)
# 대리 변수란 노드에서 가지치기를 할 때 사용된 변수를 대신할 수 있는 다른 변수를 뜻함
# ex) height < 140이면 age < 10, height >= 140 이면 age >= 10 이므로 age가 NA여도 height로 대체

library(rpart)
m <- rpart( # name,, ticket,, cbine은 모델에 적합해보이지 않아 제외하고 모델 생성
    survived ~ pclass + sex + age + sibsp + parch + fare + embarked, data=titanic.train)
p <- predict(m, newdata=titanic.train, type="class")
head(p)

# 8. rpart의 교차 검증
# 교차 테스트 데이터에 대한 예측값을 구한다
# create_ten_fold_cv( )는 train에 훈련 데이터를
# validation에 검증 데이터를 담은 리스트의 리스트를 반환한다.
library(rpart)
library(foreach)
folds <- create_ten_fold_cv()
rpart_result <- foreach(f=folds) %do% # foreach는 리스트의 Fold01, 등을 f라는 변수로 받는다.
{ 
  model_rpart <- rpart(
    survived ~ pclass + sex + age + sibsp + parch + fare + embarked, data=f$train) 
    predicted <- predict(model_rpart, newdata=f$validation, type="class")
    return(list(actual=f$validation$survived, predicted=predicted))
} # f$train과 f$validation을 사용해 rpart(), predict()를 수행한다.
# 결과는 actual에 생존 여부의 실제 값, predicted에 생존 여부의 예측값을 저장한 리스트로 반환된다
# 마지막으로 foreach는 folds 전체에 대한 결과를 또 다시 리스트로 묶는다.
head(rpart_result) # 이해를 돕기 위해 씀

# 9. 정확도 평가
# 반환 값이 여러 폴드에 대한 결과를 저장한 리스트고
# 각 폴드의 결과는 actual과 predicted에 저장된 리스트임을 감안해 정확도 계산 함수를 재작성한다
evaluation <- function(lst) # rpart_result를 입력으로 받아 sapply 수행
{
  accuracy <- sapply(lst, function(one_result) { # sapply는 각 폴드의 결과에 대한 정확도 계산 후 벡터로 묶음
    return(sum(one_result$predicted == one_result$actual) / NROW(one_result$actual))})
    print(sprintf("MEAN +/- SD: %.3f +/- %.3f", mean(accuracy), sd(accuracy)))
    return(accuracy) # 정확도의 평균과 표준 편차를 출력한 뒤 정확도 벡터를 결과로 반환
}
(rpart_accuracy <- evaluation(rpart_result)) # 인수로 넘어온 함수를 그대로 실행하는 함수
# 정확도 계산 결과 rpart 모델의 성능은 80.1% 로 나왔다 (정확도의 분산은 0.029)

# 10. 조건부 추론 나무
# 계속적으로 모델을 향상시키는 것으로 1. 다른 모델링 기법을 적용하거나
# 2. 데이터 내에 숨겨진 쓸 만한 다른 특징 값을 찾는 방법이 있다
# 1. 을 사용하기 위해 조건부 추론 나무를 적용하여 ctree()를 사용한 교차 검증을 함
library(party)
ctree_result <- foreach(f=folds) %do% {
 model_ctree <- ctree(
    survived ~ pclass + sex + age + sibsp + parch + fare + embarked, data=f$train)
  predicted <- predict(model_ctree, newdata=f$validation, type="response") # rpart와 달리 type 지정해야 반환됨
  return(list(actual=f$validation$survived, predicted=predicted))}
(ctree_accuracy <- evaluation(ctree_result))
# 정확도 계산 결과 ctree 모델의 성능은 80.4%로 나왔다 (정확도의 분산은 0.027)

plot(density(rpart_accuracy), main="rpart VS ctree") # 정확도 벡터에서 밀도 그림 그려 분포 보기
lines(density(ctree_accuracy), col="red", lty="dashed")

# 11. 또 다른 특징의 발견
# 정확도를 향상시켰으니 데이터에 숨겨진 또 다른 특징을 발견해보자
# 11-1. 티켓을 사용한 가족 선별
# sibsp 속성은 함께 탑승한 형제 또는 배우자 , parch는 함께 탑승한 부모 또는 자녀
# carbin 선실번호, embarked 탑승한 곳 name 이름, ticktet으로 가족을 찾아보기
View(titanic.train[order(titanic.train$ticket),
    c("ticket", "parch", "name", "cabin", "embarked")]) # 티켓에 따라 정리

sum(is.na(titanic.train$ticket))
sum(is.na(titanic.train$embarked))
sum(is.na(titanic.train$cabin))
# ticket의 값은 0, embarked의 값은 2개, cabin에는 912개 행에 NA가 저장되어 있다.
# NA 값이 없는 ticket 속성이 더 정확하고 사용하기 편리함을 알 수 있다.

# 6시까지 작성 계속 추가 중!! 현재 560
