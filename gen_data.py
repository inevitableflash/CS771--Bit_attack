"""
Synthetic data generator for CS771 Assignment 2 (Bit-in-the-Middle Attack).

Mimics the exact generation process described in the assignment:
  1. Sample random 16-bit challenges c.
  2. A hidden 16-bit arbiter PUF (u*, a*) generates the middle bit
         z = (sign(u*^T phi16(c) + a*) + 1) / 2.
  3. The middle bit is inserted: I(c, z) = [c[:8], z, c[8:]]  (17 bits).
  4. A 17-bit arbiter PUF (w*, b*) generates the response
         r = (sign(w*^T phi17(I(c,z)) + b*) + 1) / 2.
  5. The middle bit is erased; each row of the output file is
     16 challenge bits followed by 1 response bit (space separated 0/1),
     matching the format described in the assignment.

Usage:  python3 gen_data.py [seed]
Creates train.dat (8000 rows) and test.dat (2000 rows).
"""

import sys
import numpy as np


def phi(C):
    """Standard arbiter-PUF embedding: phi_i(c) = prod_{j >= i} (1 - 2 c_j)."""
    D = 1 - 2 * C.astype(np.int64)              # {0,1} -> {+1,-1}
    return np.cumprod(D[:, ::-1], axis=1)[:, ::-1].astype(np.float64)


def main(seed=42):
    rng = np.random.default_rng(seed)
    n_total = 10000

    # Ground-truth PUFs (standard linear-delay approximation of arbiter PUFs)
    u_star = rng.normal(0, 1, size=16)
    a_star = rng.normal(0, 0.1)
    w_star = rng.normal(0, 1, size=17)
    b_star = rng.normal(0, 0.1)

    C = rng.integers(0, 2, size=(n_total, 16))

    # Hidden 16-bit PUF produces the middle bit
    z = (phi(C) @ u_star + a_star > 0).astype(np.int64)

    # Insert middle bit at position 8 (0-indexed) => the 9th bit
    C17 = np.concatenate([C[:, :8], z[:, None], C[:, 8:]], axis=1)

    # 17-bit PUF produces the response
    r = (phi(C17) @ w_star + b_star > 0).astype(np.int64)

    data = np.concatenate([C, r[:, None]], axis=1)
    np.savetxt("train.dat", data[:8000], fmt="%d")
    np.savetxt("test.dat", data[8000:], fmt="%d")

    # Keep ground truth around so we can sanity-check recovery
    np.savez("ground_truth.npz", u=u_star, a=a_star, w=w_star, b=b_star,
             z_train=z[:8000], z_test=z[8000:])
    print("wrote train.dat (8000), test.dat (2000)")
    print(f"middle-bit balance (train): {z[:8000].mean():.3f}, "
          f"response balance (train): {r[:8000].mean():.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
