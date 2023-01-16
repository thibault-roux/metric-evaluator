
with open("hats.txt", "r", encoding="utf8") as file:
    nbr = 0
    next(file)
    for ligne in file:
        line = ligne.split("\t")
        nbr += int(line[2]) + int(line[4])

print(nbr)
print(nbr/50)