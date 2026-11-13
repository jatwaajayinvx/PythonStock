numbers = [10, 20, 5, 20, 8, 10]

firstLargNumber = float('-inf')
secondLargNumber = float('-inf')

for var in numbers:
    if var > firstLargNumber:
        secondLargNumber = firstLargNumber
        firstLargNumber = var

    elif var > secondLargNumber and var != firstLargNumber:
        secondLargNumber = var

print("First Largest:", firstLargNumber)
print("Second Largest:", secondLargNumber)