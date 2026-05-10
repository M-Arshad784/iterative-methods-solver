import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from sympy import sympify, linear_eq_to_matrix, Eq

# =========================
# 1. EQUATION PARSER
# =========================
def get_matrix(equations):
    """Convert equations into A matrix and b vector"""

    eq_list = []

    for eq in equations:
        eq = eq.replace(" ", "")

        # Convert 2x → 2*x
        eq = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq)

        if "=" in eq:
            left, right = eq.split("=")
            eq_list.append(Eq(sympify(left), sympify(right)))

    variables = sorted(eq_list[0].free_symbols, key=lambda x: x.name)

    A, b = linear_eq_to_matrix(eq_list, variables)

    return np.array(A, dtype=float), np.array(b, dtype=float).flatten(), variables


# =========================
# 2. JACOBI METHOD
# =========================
def jacobi(A, b, max_iter=50, tol=1e-5):
    n = len(A)
    x = np.zeros(n)

    errors = []
    history = []

    for i in range(max_iter):
        x_new = np.zeros(n)

        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - s) / A[j][j]

        error = np.max(np.abs(x_new - x))
        x = x_new

        errors.append(error)
        history.append((i + 1, x.copy(), error))

        if error < tol:
            break

    return errors, x, history


# =========================
# 3. GAUSS-SEIDEL METHOD
# =========================
def gauss_seidel(A, b, max_iter=50, tol=1e-5):
    n = len(A)
    x = np.zeros(n)

    errors = []
    history = []

    for i in range(max_iter):
        old_x = x.copy()

        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            x[j] = (b[j] - s) / A[j][j]

        error = np.max(np.abs(x - old_x))

        errors.append(error)
        history.append((i + 1, x.copy(), error))

        if error < tol:
            break

    return errors, x, history


# =========================
# 4. POWER METHOD
# =========================
def power_method(A, max_iter=100, tol=1e-6):
    n = len(A)
    x = np.ones(n)

    eigenvalues = []
    history = []

    for i in range(max_iter):
        x_new = A @ x

        lam = (x @ x_new) / (x @ x)
        x_new = x_new / np.linalg.norm(x_new)

        eigenvalues.append(lam)
        history.append((i + 1, lam))

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return eigenvalues, x, history


# =========================
# 5. QR METHOD
# =========================
def qr_method(A, iters=30):
    A = A.copy()
    errors = []

    for _ in range(iters):
        Q, R = np.linalg.qr(A)
        A = R @ Q

        err = np.linalg.norm(A - np.diag(np.diag(A)))
        errors.append(err)

    return np.real(np.diag(A)), errors


# =========================
# 6. DIAGONALIZATION
# =========================
def diagonalize(A):
    eigvals, eigvecs = np.linalg.eig(A)

    D = np.diag(np.real(eigvals))
    P = eigvecs
    P_inv = np.linalg.inv(P)

    return eigvals, D, P, P_inv


# =========================
# 7. STREAMLIT UI
# =========================
st.title("🔢 Linear Algebra Solver (Educational Version)")

n = st.number_input("Number of Equations", 2, 10, 3)

st.subheader("Enter Equations (example: 2x + y = 5)")

equations = []
for i in range(n):
    equations.append(st.text_input(f"Equation {i+1}"))


if st.button("Solve"):

    try:
        # ---------- Convert to Matrix ----------
        A, b, vars = get_matrix(equations)

        st.subheader("Matrix A")
        st.write(A)

        st.subheader("Vector b")
        st.write(b)

        # ---------- Solve Methods ----------
        j_err, j_sol, j_hist = jacobi(A, b)
        g_err, g_sol, g_hist = gauss_seidel(A, b)
        p_vals, p_vec, p_hist = power_method(A)
        qr_vals, qr_err = qr_method(A)
        eigvals, D, P, Pinv = diagonalize(A)

        # ---------- Results ----------
        st.subheader("Jacobi Method")
        for i, x, e in j_hist:
            st.write(f"Iter {i}: {np.round(x, 4)} | Error: {e:.6f}")

        st.write("Final:", j_sol)

        st.subheader("Gauss-Seidel Method")
        for i, x, e in g_hist:
            st.write(f"Iter {i}: {np.round(x, 4)} | Error: {e:.6f}")

        st.write("Final:", g_sol)

        st.subheader("Power Method")
        for i, val in p_hist:
            st.write(f"Iter {i}: λ ≈ {val}")

        st.write("Eigenvector:", p_vec)

        st.subheader("QR Method")
        st.write("Eigenvalues:", qr_vals)

        st.subheader("Diagonalization")
        st.write("Eigenvalues:", eigvals)
        st.write("D Matrix:", D)
        st.write("P Matrix:", P)
        st.write("P Inverse:", Pinv)

        # ---------- Graph 1 ----------
        fig1, ax1 = plt.subplots()

        ax1.plot(j_err, marker='o', linewidth=2, label="Jacobi")
        ax1.plot(g_err, marker='s', linewidth=2, label="Gauss-Seidel")
        
        ax1.set_title("Convergence Comparison")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Error")
        ax1.legend()
        ax1.grid()
        
        st.pyplot(fig1)

        # ---------- Graph 2 ----------
        fig2, ax2 = plt.subplots()

        ax2.plot(p_vals, marker='o', linewidth=2)
        
        ax2.set_title("Power Method")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Eigenvalue/Error")
        ax2.grid()
        
        st.pyplot(fig2)

        # ---------- Graph 3 ----------
        fig3, ax3 = plt.subplots()

        ax3.plot(qr_err, marker='^', linewidth=2)
        
        ax3.set_title("QR Method")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Error")
        ax3.grid()
        
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error: {e}")