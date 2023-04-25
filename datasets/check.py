
distrib = dict()

with open("hats.txt", "r", encoding="utf8") as file:
    nbr = 0
    next(file)
    for ligne in file:
        line = ligne.split("\t")
        nbr += int(line[2]) + int(line[4])
        if (int(line[2]) + int(line[4])) not in distrib:
            distrib[int(line[2]) + int(line[4])] = 1
        else:
            distrib[int(line[2]) + int(line[4])] += 1

print(nbr)
print(nbr/50)

print(distrib)