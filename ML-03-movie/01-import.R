library(quanteda)

# set data dir
data.dir.tf <- 'C:/Users/rossz/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-03-movie'
data.dir.sp <- 'C:/Users/Yu Zhu/OneDrive/Academy/the U/Assignment/AssignmentSln/ML-03-movie'
data.dir <- ifelse(file.exists(data.dir.tf), data.dir.tf, data.dir.sp)
rm(data.dir.tf, data.dir.sp)

# get review id
train.id <- fread(file.path(data.dir, 'data/data-splits/data.train.id'), header = F, col.names = 'id')
test.id <- fread(file.path(data.dir, 'data/data-splits/data.test.id'), header = F, col.names = 'id')
eval.id <- fread(file.path(data.dir, 'data/data-splits/data.eval.anon.id'), header = F, col.names = 'id')

# get label 
train.label <- fread(file.path(data.dir, 'data/data-splits/data.train'), header = F, col.names = 'label', sep = NULL)[, ":="(label = str_sub(label, 1, 1))]
test.label <- fread(file.path(data.dir, 'data/data-splits/data.test'), header = F, col.names = 'label', sep = NULL)[, ":="(label = str_sub(label, 1, 1))]
eval.label <- fread(file.path(data.dir, 'data/data-splits/data.eval.anon'), header = F, col.names = 'label', sep = NULL)[, ":="(label = str_sub(label, 1, 1))]

# create train/test/eval data.table
train <- fread(file.path(data.dir, 'data/raw-data/train.rawtext'), header = F, col.names = 'text', sep = NULL)[, ':='(type = 'train', rowid = 1:.N, docid = train.id$id, label = train.label$label)]
test <- fread(file.path(data.dir, 'data/raw-data/test.rawtext'), header = F, col.names = 'text', sep = NULL)[, ':='(type = 'test', rowid = 1:.N, docid = test.id$id, label = test.label$label)]
eval <- fread(file.path(data.dir, 'data/raw-data/test.rawtext'), header = F, col.names = 'text', sep = NULL)[, ':='(type = 'eval', rowid = 1:.N, docid = eval.id$id)]
rm(train.label, test.label, eval.label)
rm(train.id, test.id, eval.id)

# create corpus from data.table
corpus.train <- corpus(train)
corpus.test <- corpus(test)
corpus.eval <- corpus(eval)
rm(train, test, eval)

# merge into single corpus
corpus <- corpus.train + corpus.test + corpus.eval

# rm duplicate
rm(corpus.train, corpus.test, corpus.eval)

# save to disk
sv(corpus)

