total = 3253

with open('data/relations.txt', mode='w') as f:
    for ix in range(total):
        for jx in range(1, 13):
            if ix - jx > 0:
                f.write('%s, %s, %s\n' % (ix, ix - jx, -jx))
        for jx in range(1, 13):
            if ix + jx < total:
                f.write('%s, %s, %s\n' % (ix, ix + jx, jx))
