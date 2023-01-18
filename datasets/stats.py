
# statistique on human certitude



"""

with open("hats.txt", "r", encoding="utf8") as file:
    certitude = []
    next(file)
    for ligne in file:
        line = ligne.split("\t")
        nbrA = int(line[2]) 
        nbrB = int(line[4])
        if nbrA + nbrB >= 5:
            maximum = max(nbrA, nbrB)
            cert = (maximum) / (nbrA+nbrB)
            certitude.append(cert)

print(sum(certitude)/len(certitude))
print()
for i in sorted(set(certitude)):
    print(i, certitude.count(i))

# my data
hyp1: 1, hyp2: 6
hyp1: 7, hyp2: 0

#ici sujet en colonne et rater en 
[
    [1, 6], 
    [7, 0],
    [2, 5]
]
"""