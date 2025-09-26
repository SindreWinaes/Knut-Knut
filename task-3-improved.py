

soloutions = []

for c in range (1, 450):
    a = c +11

    # b is even
    if 1 <= a <= 449:
        b = (a * c) % 2377

        if b % 2 == 0 and 1 <= b <= 449:
            c_rule = a * b - ( 7 * a * ( a - 1 ) ) // 2 + 142

            if c == c_rule:
                soloutions.append((a,b,c))

    # b is odd
    a = 2 * c - 129
    if 1 <= a <= 449:
        b = (a * c) % 2377

        if b % 2 == 1 and 1 <= b <= 449:
            c_rule =  a * b - ( 7 * a * ( a - 1 )) // 2 + 142

            if c == c_rule:
                soloutions.append((a,b,c))

print(f"Soloutions found : {soloutions}")