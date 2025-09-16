import string
import random


def generate_password(length, include_uppercase, include_numbers, include_special):
    if length < (include_uppercase + include_numbers + include_special):
        raise ValueError

    # create password
    password = ''
    if include_uppercase:
        password += random.choice(string.ascii_uppercase)
    if include_numbers:
        password += random.choice(string.digits)
    if include_special:
        password += random.choice(string.punctuation)

    # Fill the remaining length with any allowec characters
    characters = string.ascii_lowercase
    if include_uppercase:
        characters += string.ascii_uppercase
    if include_numbers:
        characters += string.digits
    if include_special:
        characters += string.punctuation

    for _ in range(length - len(password)):
        password += random.choice(characters)

    # randomize character order (dont make easy to guess)
    password_list = list(password)
    random.shuffle(password_list)
    return ''.join(password_list)


def main():
    length = int(input('Enter password length: '))
    include_uppercase = input(
        'Include uppercase letters (y/n)?: ').lower() == 'y'
    include_numbers = input(
        'Include number (y/n)?: ').lower() == 'y'
    include_special = input(
        'Include special characters (y/n)?: ').lower() == 'y'
    try:
        password = generate_password(
            length, include_uppercase, include_numbers, include_special)
        print(password)
    except ValueError as e:
        print('Password length is too short for the specified criteria.')


if __name__ == '__main__':
    main()
