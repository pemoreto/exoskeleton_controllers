import numpy as np
import matplotlib.pyplot as plt
import os

def compute_jacobian(full_X, full_Y, save_dir, version, title='', show_plot=True):
    """
    Computes the Jacobian matrix from Poincaré map data and plots Floquet multipliers.

    Parameters:
    - full_X (ndarray): State vectors at step n (N-1 x D).
    - full_Y (ndarray): State vectors at step n+1 (N-1 x D).
    - save_dir (str): Directory to save the Floquet plot.
    - version (str): Version suffix for saved file.
    - show_plot (bool): Whether to display the plot.

    Returns:
    - J (ndarray): Estimated Jacobian matrix.
    - eigenvalues (ndarray): Floquet multipliers (eigenvalues of J).
    """

    # Definition: je 2 Werte pro Gelenk
    joint_labels = [
        "Knee (angle)", "Knee (velocity)",
        "Ankle (angle)", "Ankle (velocity)",
        "Hip (angle)", "Hip (velocity)"
    ]
    joint_colors = [
        "darkblue", "lightblue",
        "darkgreen", "lightgreen",
        "darkorange", "navajowhite"
    ]

    # Mittelwert-Punkt berechnen
    fixed_point = full_X.mean(axis=0)
    delta_x = full_X - fixed_point
    delta_y = full_Y - fixed_point

    # Least-squares für Jacobian
    J, _, _, _ = np.linalg.lstsq(delta_x, delta_y, rcond=None)

    # Eigenwerte berechnen
    eigenvalues = np.linalg.eigvals(J)

    # Plot vorbereiten
    fig, ax = plt.subplots()
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Einheitskreis')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    # Alle EW einzeichnen
    for i, (ev, label, color) in enumerate(zip(eigenvalues, joint_labels, joint_colors)):
        ax.plot(np.real(ev), np.imag(ev), 'o', color=color, label=label)

    ax.set_aspect('equal')
    ax.set_xlabel('Realteil')
    ax.set_ylabel('Imaginärteil')
    ax.set_title('Floquet-Multiplikatoren im komplexen Zahlenraum')
    ax.grid(True)
    #ax.legend(loc='best', fontsize='small')

    if save_dir is not None:
        filename = os.path.join(save_dir, f"{title}_{version}.svg")
        plt.savefig(filename, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return J, eigenvalues
