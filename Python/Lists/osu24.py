items = [("Mike Gundy", 1),
         ("John Smith", 2),
         ("Steve Lutz", 5),
         ("Kenny Gajewski", 4),
         ("Colin Carmichael", 6),
         ("Josh Holliday", 3),
         ]

# just get the rank
rank = []
for item in items:
    rank.append(item[1])

# another way to do this
x = map(lambda item: item[1], items)
for item in x:
    print(item)

# still another way
rank2 = list(map(lambda item: item[1], items))
print(rank2)

print(rank)
