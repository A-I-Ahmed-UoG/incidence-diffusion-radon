"""Reference implementation of the p = 23 instance of Section 8.3, plus the
statistical checks quoted in the paper. Run: python3 instance23.py"""
import os
import numpy as np
from lib import radon

P, ALPHA, R = 23, 2, 12
N = P * P
RATE, CAP = 415, 114
OUT_CELLS = 85

SBOX = np.array([0] + [pow(x, -1, P) for x in range(1, P)], dtype=np.int64)
A = (radon(P) + ALPHA * np.eye(N, dtype=np.int64)) % P
KAPPA = np.array([[[(y + (r + 1) * x + r + 1) ** 3 % P for x in range(P)]
                   for y in range(P)] for r in range(R)], dtype=np.int64).reshape(R, N)

def perm(state):
    s = state.copy()
    for r in range(R):
        s = (SBOX[s] + KAPPA[r]) % P
        s = (A @ s) % P
    return s

def encode(msg: bytes):
    sym = []
    for b in msg:
        sym.append(b >> 4); sym.append(b & 0xF)
    sym.append(16)
    while len(sym) % RATE:
        sym.append(0)
    return np.array(sym, dtype=np.int64)

def digest(msg: bytes) -> bytes:
    s = np.zeros(N, dtype=np.int64)
    sym = encode(msg)
    for i in range(0, len(sym), RATE):
        s[:RATE] = (s[:RATE] + sym[i:i + RATE]) % P
        s = perm(s)
    acc = 0
    for z in reversed(s[:OUT_CELLS]):
        acc = acc * P + int(z)
    return (acc % (1 << 256)).to_bytes(32, "big")

if __name__ == "__main__":
    # sanity: the permutation is a bijection on a random sample (inverse check)
    inv = lambda z: pow(z, -1, P)
    Ainv = (inv(ALPHA) * np.eye(N, dtype=np.int64) - inv(ALPHA)**2 * radon(P)
            + inv(ALPHA)**3 * (radon(P) @ radon(P) % P)
            - inv(ALPHA)**3 * inv(ALPHA - 1) * (radon(P) @ radon(P) @ radon(P) % P)) % P
    rng = np.random.default_rng(7)
    v = rng.integers(0, P, N)
    w = perm(v)
    for r in reversed(range(R)):
        w = SBOX[(Ainv @ w % P - KAPPA[r]) % P]
    print("round-inverse check:", bool((w == v).all()))

    # encoding is injective: 0x00 and 0x17 differ (a mod-p reduction would not)
    print("H(00) != H(17):", digest(b"\x00") != digest(b"\x17"))

    # avalanche over single input-bit flips
    flips = []
    for t in range(200):
        m = bytearray(rng.integers(0, 256, 64, dtype=np.uint8).tolist())
        h1 = digest(bytes(m))
        j = int(rng.integers(0, 8 * len(m)))
        m[j // 8] ^= 1 << (j % 8)
        h2 = digest(bytes(m))
        flips.append(sum(bin(a ^ b).count("1") for a, b in zip(h1, h2)) / 256)
    flips = np.array(flips)
    print(f"avalanche over {len(flips)} flips: mean {flips.mean():.4f}, "
          f"sd {flips.std(ddof=1):.4f} (ideal 0.5, 0.03125)")
    np.savetxt(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figdata"), "avalanche.dat"), np.c_[np.arange(len(flips)), flips],
               header="i frac", comments="")
    # byte-frequency of the digest bytes, as a uniformity check of the output map
    cnt = np.zeros(256)
    for t in range(200):
        for b in digest(bytes(rng.integers(0, 256, 32, dtype=np.uint8).tolist())):
            cnt[b] += 1
    chi = ((cnt - cnt.mean()) ** 2 / cnt.mean()).sum()
    print(f"output byte chi-square: {chi:.1f} on 255 df (expect ~255)")
