import random


guesses = 0

while True:
    a = random.randint(1,449)
    b = random.randint(1, 449)
    c = random.randint(1, 449)

    guesses += 1
    print(guesses )
    # Rule 1
    if b % 2 == 0:
        if a != c +11:
            continue
    else:
        if a != 2*c - 129:
            continue

    # Rule 2
    if b != (a * c) % 2377:
        continue

    # Rule 3
    c_rule = a * b - (7 * a * (a - 1)) // 2 + 142

    if c != c_rule:
        continue


    print(f"Souloution, a={a}, b={b}, c={c} (guess {guesses})")

    break
