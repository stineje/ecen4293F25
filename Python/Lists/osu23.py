items = [("Mike Gundy", 1),
         ("John Smith", 2),
         ("Steve Lutz", 5),
         ("Kenny Gajewski", 4),
         ("Colin Carmichael", 6),
         ("Josh Holliday", 3),
         ]


# lambda uses parameters:expression
items.sort(key=lambda item: item[1])
print(items)
