library(caret)
# set seed
set.seed(42)

# train.control
control <- trainControl(method = 'cv', number = 3, classProbs = T, summaryFunction = twoClassSummary)
# train
model <- train(class.true ~ ., data = train, method = 'ranger', trControl = control)
class.pred <- predict(model, newdata = eval) %>% as.numeric() * -1 + 2

# final result
report <- data.table(example_id = docvars(dfm.eval, 'docid'), label = class.pred)
fwrite(report, 'report.csv')