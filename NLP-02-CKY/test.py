cky = CKY()
cky.parse(prob = 'prob')
o = sorted(['%s[%s] (%1.4f)' % (i['left'], i['right'], i['prob']) for i in chart.iloc[0, 1]])
all_s = [i for i in o if i[0:2] == 'S_']
