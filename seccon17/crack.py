import itertools
import hashlib


def encrypt(key, plaintext, alphabet):
    ciphertext = ""
    # Encrypt each character of the plaintext with the according character of the key
    for i in range(0, len(plaintext)):
        # Find the current plaintext character
        character = plaintext[i]

        # Find the according key character
        keychar = key[i % len(key)]

        # Find the alphabet indexes of the plaintext and key characters
        shift = alphabet.index(character)
        key_index = alphabet.index(keychar)

        # Apply the shift
        ciphertext += alphabet[(shift + key_index) % len(alphabet)]
    return ciphertext


def decrypt(ciphertext, key, alphabet):
    plaintext = ""
    # Decript each character of the ciphertext with the according character of the key
    for i in range(0, len(ciphertext)):
        # Find the current ciphertext character
        character = ciphertext[i]

        # Find the according key character
        keychar = key[i % len(key)]

        # Find the alphabet indexes of the ciphertext and key characters
        shift = alphabet.index(character)
        key_index = alphabet.index(keychar)

        # Apply the shift
        plaintext += alphabet[(shift - key_index) % len(alphabet)]
    return plaintext


def brute_force(ciphertext, alphabet):
    # Start with the known part of the key
    key = ""
    plaintext = ""

    # Generate a list of all possible 4-character combinations from our (custom) alphabet
    possible_keys = [''.join(i) for i in itertools.product(alphabet, repeat=14)]
    print ("Trying {} keys".format(len(possible_keys)))

    possible_solutions = []

    # Try each possible 4-character combination
    for possible_key in possible_keys:
        # Append the combination to the known part of the key (e.g. 'VIGENERE' + 'AAAA')
        temp_key = key + possible_key

        # Obtain the plaintext using the current combination as the key
        plaintext = decrypt(ciphertext, temp_key, alphabet)

        # If the plaintext does not end in '}', we are not interested
        if plaintext[-1] == "}":
            # Store the result (the (key,plaintext) combination)
            possible_solutions.append((temp_key, plaintext))

    print ("Found {} possible solutions...".format(len(possible_solutions)))

    # For each obtained plaintext:
    for solution in possible_solutions:
        # ...calculate the MD5 hash and compare it to the known correct hash
        if hashlib.md5(solution[1]).hexdigest() == "f3ad0aad4b772e4428468b792e12f0ce":
            # If they match, we found the correct key, resulting in the correct plaintext
            print ("FOUND!!!! --> {}".format(solution))


# plaintext = "SECCON{???????????????????????????????????}"
# key = "VIGENERE????"

ciphertext = "POR4dnyTLHBfwbxAAZhe}}ocZR3Cxcftw9"
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz_{}"

brute_force(ciphertext, alphabet)