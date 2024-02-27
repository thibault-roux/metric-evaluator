import jiwer

def needleman_wunsch(A, B, d):
    len_A = len(A)
    len_B = len(B)

    # Initialize the matrix F with zeros
    F = [[0] * (len_B + 1) for _ in range(len_A + 1)]

    # Initialize the first row and column of F with gap penalties
    for i in range(len_A + 1):
        F[i][0] = d * i
    for j in range(len_B + 1):
        F[0][j] = d * j

    # Fill in the matrix F using the recurrence relation
    for i in range(1, len_A + 1):
        for j in range(1, len_B + 1):
            choice1 = F[i-1][j-1] + similarity_score(A[i-1], B[j-1])  # Assuming similarity_score is a function to calculate match/mismatch scores
            choice2 = F[i-1][j] + d
            choice3 = F[i][j-1] + d
            F[i][j] = max(choice1, choice2, choice3)

    return F

def similarity_score(a, b):
    # Assuming a simple match/mismatch scoring function
    # return 1 if a == b else -1

    # based on character error rate
    return 1 - jiwer.cer(a, b)




def needleman_wunsch_traceback(A, B, F, d):
    AlignmentA = ""
    AlignmentB = ""
    i = len(A)
    j = len(B)

    while i > 0 and j > 0:
        Score = F[i][j]
        ScoreDiag = F[i - 1][j - 1]
        ScoreUp = F[i][j - 1]
        ScoreLeft = F[i - 1][j]

        if Score == ScoreDiag + similarity_score(A[i - 1], B[j - 1]):
            AlignmentA = A[i - 1] + AlignmentA
            AlignmentB = B[j - 1] + AlignmentB
            i -= 1
            j -= 1
        elif Score == ScoreLeft + d:
            AlignmentA = A[i - 1] + AlignmentA
            AlignmentB = "<eps> " + AlignmentB
            i -= 1
        else:  # Score == ScoreUp + d
            AlignmentA = "<eps> " + AlignmentA
            AlignmentB = B[j - 1] + AlignmentB
            j -= 1


    while i > 0:
        AlignmentA = A[i - 1] + AlignmentA
        AlignmentB = "<eps> " + AlignmentB
        i -= 1
        
    while j > 0:
        AlignmentA = "<eps> " + AlignmentA
        AlignmentB = B[j - 1] + AlignmentB
        j -= 1
        
    return AlignmentA, AlignmentB



def transform(txt):
    # if txt[0] == " ":
    #     txt = txt[1:]
    # if txt[-1] == " ":
    #     txt = txt[:-1]
    temp = txt.split(" ")
    transformed = []
    for t in temp:
        transformed.append(t + " ")
    return transformed


def print_alignment(alignmentA, alignmentB):
    saveA = alignmentA
    saveB = alignmentB
    alignmentA = alignmentA.split()
    alignmentB = alignmentB.split()
    if len(alignmentA) != len(alignmentB):
        print("Error: Alignment length mismatch")
        print(saveA)
        print(saveB)
        raise ValueError
    line1 = "" # ref
    line2 = "" # errors
    line3 = "" # hyp
    for i in range(len(alignmentA)):
        if alignmentA[i] == alignmentB[i]:
            line1 += alignmentA[i] + " "*(len(alignmentB[i])-len(alignmentA[i])) + " "
            line2 += " "*max(len(alignmentA[i]), len(alignmentB[i])) + " "
            line3 += alignmentB[i] + " "*(len(alignmentA[i])-len(alignmentB[i])) + " "
        elif alignmentA[i] == "<eps>":
            line1 += "*"*len(alignmentB[i]) + " "
            line2 += "I" + " "*(max(len(alignmentA[i]), len(alignmentB[i]))-1) + " "
            line3 += alignmentB[i] + " "
        elif alignmentB[i] == "<eps>":
            line1 += alignmentA[i] + " "
            line2 += "D" + " "*(max(len(alignmentA[i]), len(alignmentB[i]))-1) + " "
            line3 += "*"*len(alignmentA[i]) + " "
        elif alignmentA[i] != alignmentB[i]:
            line1 += alignmentA[i] + " "*(len(alignmentB[i])-len(alignmentA[i])) + " "
            line2 += "S" + " "*(max(len(alignmentA[i]), len(alignmentB[i]))-1) + " "
            line3 += alignmentB[i] + " "*(len(alignmentA[i])-len(alignmentB[i])) + " "
        else:
            print("Error: Unknown case")
            raise ValueError
    txt = line1 + "\n" + line2 + "\n" + line3
    return txt



# A = "sur ce plateaux"
# B = "source plateau"

# A = transform(A)
# B = transform(B)
# d = -2

# matrix_F = needleman_wunsch(A, B, d)
# alignmentA, alignmentB = needleman_wunsch_traceback(A, B, matrix_F, d)

# print_alignment(alignmentA, alignmentB)

def align(ref, hyp):
    A = transform(ref)
    B = transform(hyp)
    d = -2

    print(ref)
    print(hyp)

    matrix_F = needleman_wunsch(A, B, d)
    alignmentA, alignmentB = needleman_wunsch_traceback(A, B, matrix_F, d)

    return print_alignment(alignmentA, alignmentB)


if __name__ == "__main__":
    # ref = input("Enter reference: ")
    # hyp = input("Enter hypothesis: ")

    ref = "tout même si y en avait sur les"
    hyp = "tout mois mas sine a e sor es entai"

    print()

    print(align(ref, hyp))