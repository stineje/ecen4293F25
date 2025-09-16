import random
import string


def generate_password(length, use_uppercase, use_numbers, use_specials):
    # Start with lowercase letters always included
    characters = list(string.ascii_lowercase)

    if use_uppercase:
        characters.extend(string.ascii_uppercase)
    if use_numbers:
        characters.extend(string.digits)
    if use_specials:
        characters.extend(string.punctuation)

    if not characters:
        raise ValueError(
            "No character sets selected! Cannot generate password.")

    # Randomly choose from the allowed characters
    password = ''.join(random.choice(characters) for _ in range(length))
    return password


def main():
    try:
        length = int(input("Enter the desired password length: "))
        if length <= 0:
            print("Password length must be greater than 0.")
            return

        use_uppercase = input(
            "Include uppercase letters? (y/n): ").strip().lower() == 'y'
        use_numbers = input("Include numbers? (y/n): ").strip().lower() == 'y'
        use_specials = input(
            "Include special characters? (y/n): ").strip().lower() == 'y'

        password = generate_password(
            length, use_uppercase, use_numbers, use_specials)
        print("\nGenerated Password:", password)

    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
