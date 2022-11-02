.libPaths()
.libPaths("C:/project")
# cran 미러설정
options(repos = c(CRAN = "http://cran.rstudio.com"))

#1. 라이브러리 설치 및 infile
install.packages("dplyr")
install.packages("nnet")
install.packages("randomForest")
install.packages("tidyverse")
install.packages("mgsub")
install.packages("caret")

library(dplyr) 
library(tidymodels)
library(nnet)
library(randomForest)
library(tidyverse)
library(mgsub)
library(caret)

df <- read.csv("C:/project/sampling.csv", header=TRUE, sep=",",fill=TRUE) 
head(df, 5)
# 110790 obs. of 10 variables

#2. 데이터 EDA 및 전처리
# 데이터 확인 
names(df)
summary(df)
sum(is.na(df))
table(is.na(df$재무등급))
table(is.na(df$모형))

# 결측치 제거 
df1<-na.omit(df)
sum(is.na(df1$재무등급))
df1 <- df1[!(df1$최종등급 == "20100331" ), ]
df1 <- df1[!(df1$재무등급 == "NG" ), ]

# 사업자번호 제거 
df1<-df1[,c(-1,-3)]

head(df1)

# 등급 ->숫자로 표현 
df1$재무등급<-mgsub::mgsub(df1$재무등급, c("D", "CCC","CC","C","B","BB","BBB","A","AA","AAA"), c(0,1,2,3,4,5,6,7,8,9))
df1$최종등급<-mgsub::mgsub(df1$최종등급, c("D", "CCC","CC","C","B","BB","BBB","A","AA","AAA"), c(0,1,2,3,4,5,6,7,8,9))
df1$재무등급<-as.numeric(df1$재무등급)
df1$최종등급<-as.numeric(df1$최종등급)


# 데이터 split
set.seed(71)
divide=sample(c(rep(0,0.7*nrow(df1)),rep(1,0.3*nrow(df1))))
train_data=df1[divide==0,]
test_data=df1[divide==1,]

str(train_data)


# 3. 모델링 
# 3-1 로지스틱  
model_lg<- multinom(최종등급 ~., data=train_data)
summary(mlogit)
predict1=predict(mlogit, test_data)
confusionMatrix(table(predict1, test_data$최종등급))

# 3-2 랜덤포레스트
model_rf=randomForest(as.factor(최종등급) ~.,data=train_data)
print(model_rf)
predict2<-predict(model_rf, test_data[,-8])
confusionMatrix(table(predict2, test_data$최종등급))
# 중요 변수 확인
importance(model_rf)
varImpPlot(model_rf)


# submission 제출
submission <- read.csv("C:/project/submission.csv", header=TRUE, sep=",") 
submission <- rbind(submission,predict2)

write.csv(submission, '경로\파일명.csv')
 