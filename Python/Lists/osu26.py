items = [("Mike Gundy", 1),
         ("John Smith", 2),
         ("Steve Lutz", 5),
         ("Kenny Gajewski", 4),
         ("Colin Carmichael", 6),
         ("Josh Holliday", 3),
         ]

# Lets use List Comprehension
rank2 = list(map(lambda item: item[1], items))
rank1 = [item[1] for item in items]
# Can also use filtering
filtered = [item for item in items if item[1] <= 3]

print(rank1)
print(rank2)
print(filtered)
