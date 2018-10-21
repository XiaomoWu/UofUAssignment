library(quanteda)
library(caret)
#library(spacyr)
ld(corpus, T)

# dfm
dfm <- dfm(corpus, remove = stopwords("english"), remove_punct = T, stem = T)
dfm.trim <- dfm_trim(dfm, termfreq_type = 'rank', min_termfreq = 1000, docfreq_type = 'count', min_docfreq = 100) %>% dfm_tfidf()
dfm.train <- dfm_subset(dfm.trim, type == 'train')
dfm.eval <- dfm_subset(dfm.trim, type == 'eval')


# create train set
train <- convert(dfm.train, to = 'data.frame') %>% setDT()
train[, ':='(class.true = ifelse(docvars(dfm.train, 'label') == '1', 'P', 'N'), document = NULL)]
#train[, ':='(class.true = as.factor(docvars(dfm.train, 'label')), document = NULL)]

eval <- convert(dfm.eval, to = 'data.frame') %>% setDT()
eval[, ':='(class.true = ifelse(docvars(dfm.eval, 'label') == '1', 'P', 'N'), document = NULL)]

