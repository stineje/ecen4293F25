items = [("Mike Gundy", 1),
         ("John Smith", 2),
         ("Steve Lutz", 5),
         ("Kenny Gajewski", 4),
         ("Colin Carmichael", 6),
         ("Josh Holliday", 3),
         ]

# just get the rank
x = filter(lambda item: item[1] <= 3, items)
print(list(x))
