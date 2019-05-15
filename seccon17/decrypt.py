import sys


def _l(idx, s):
    return s[idx:] + s[:idx]


def main(c):
    s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz_{}"
    t = [[_l((i + j) % len(s), s) for j in range(len(s))] for i in range(len(s))]
    p = "SECCON{}"
    key = ""
    pi = 0

    for w in c:
        for i1 in range(len(s)):
            for i2 in range(len(s)):
                for i3 in range(len(s)):
                    if w == t[s.find(i1)][s.find(i2)][s.find(i3)]:


        key += t[0][s.find(w)][]

    print("Plain: " + p + ", Key: " + key)

    # for a in t:
    #     # if pi % 14 == 0:
    #     if pi > 1:
    #         print("\n")
    #         break
    #
    #     print(a)
    #     pi += 1
    #
    # print(pi)
    print(t[0][1][2])


ciphertext = "POR4dnyTLHBfwbxAAZhe}}ocZR3Cxcftw9"
print(main(ciphertext))
