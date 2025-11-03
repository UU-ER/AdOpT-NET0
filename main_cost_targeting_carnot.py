import matplotlib.pyplot as plt
import numpy as np

def calculate_carnot_cop(t_low, t_high):
    return 1 / (1 - (t_low + 273) / (t_high + 273))

def main():
    p_th = 1
    a = 1

    t_low = np.linspace(-10, 60, 100)
    t_high = np.linspace(70, 200, 100)
    X, Y = np.meshgrid(t_low, t_high)
    Z = p_th*(1-a/calculate_carnot_cop(X, Y))

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    # Surface plot
    surf = ax.contourf(X, Y, Z, cmap='viridis')

    # Labels and title
    ax.set_xlabel("t_in")
    ax.set_ylabel("t_out")
    # ax.set_zlabel("z")

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    plt.show()

if __name__ == "__main__":
    main()