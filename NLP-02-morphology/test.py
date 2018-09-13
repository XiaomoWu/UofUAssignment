# rules
with open(DATA_DIR + 'rules.txt') as f:
    lines = f.readlines()
rules = []
for line in lines:
    line = line.strip().lower().split()
    if len(line) == 7:
        rules.append({'type': line[0], 'affix': line[1], 'replace': line[2].replace(), 'pos_origin': line[3], 'pos_derived': line[5]})

# dict
with open(DATA_DIR + 'dict.txt') as f:
    lines = f.readlines()
dict = []
for line in lines:
    line = line.strip().lower().split()
    if len(line) == 2:
        dict.append({'word': line[0], 'pos': line[1], 'root': line[0]})
    elif len(line) == 4:
        dict.append({'word': line[0], 'pos': line[1], 'root': line[3]})

#########################
x = PARSER()
x.parse('reviewed')
x.output

d = list({'dressed adjective ROOT=dress SOURCE=morphology',
 'dressed noun ROOT=dressed SOURCE=default',
 'dressed verb ROOT=dress SOURCE=morphology'})

[x for x in d if x.find('SOURCE=default') < 0]

d[0].find('SOURCE=default')


word = 'viewness'

s = set()
s.add(1)
s