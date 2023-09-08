items = []

with open('data/Sunspots.csv') as f:
    for lix, line in enumerate(f):
        if not line.startswith('#') and lix > 0:
            print(line)
            items.append(line.split(',')[-1][:-1])

print('total months: %s' % len(items))

with open('data/points.txt', mode='w') as f:
    for ix, item in enumerate(items):
        if ix >= 12:
            f.write('%s, %s\n' % (ix - 12, ', '.join(items[ix - 12:ix])))

