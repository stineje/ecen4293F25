items = [("Mike Gundy", 1),
         ("John Smith", 2),
         ("Steve Lutz", 5),
         ("Kenny Gajewski", 4),
         ("Colin Carmichael", 6),
         ("Josh Holliday", 3),
         ]


def sort_item(item):
    return item[1]


items.sort(key=sort_item)
print(items)
