from random import choice

total = 3253
relations = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

with open('data/walks.txt', mode='w') as f:
    for ix in range(110000):
        items = []
        cur = choice(range(total))
        items.append(str(cur))
        for jx in range(12):
            jump = choice(relations)
            items.append(str(jump))
            cur = cur + jump
            if cur < 0 or cur >= total:
                break
            items.append(str(cur))
        if len(items) == 25:
            f.write('%s\n' % (', '.join(items)))
