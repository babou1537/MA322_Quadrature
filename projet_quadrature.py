""" MA322 - Quadrature & resolution numerique & d’EDO
Projet : modelisation des systemes chaotiques par le pendule double 

Membre du Groupe :
    - Maëva AGUESSY
    - Chloé TONSO
    - Baptiste MATHIEU
    - Phuong Mai NGUON

Classe 3PSB1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    1) Implémenter une fonction F 
   ----------------------------------------------------------------
"""
# Définition de la fonction F
def F(Y, l1, l2, m1, m2):
    g = 9.81
    """
    theta1, theta2, theta1_dot, theta2_dot = Y
    
    # Calcul des dérivées secondes
    theta1_double_dot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (theta2_dot ** 2 * l2 + theta1_dot ** 2 * l1 * np.cos(theta1 - theta2))) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    theta2_double_dot = (2 * np.sin(theta1 - theta2) * (theta1_dot ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + theta2_dot ** 2 * l2 * m2 * np.cos(theta1 - theta2))) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    
    return np.array([theta1_dot, theta2_dot, theta1_double_dot, theta2_double_dot])
    """

    theta1, theta2, theta1_dot, theta2_dot = Y
    delta_theta = theta1 - theta2

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta_theta)**2
    den2 = (l2 / l1) * den1

    theta1_double_dot = ((m2 * g * np.sin(theta2) * np.cos(delta_theta) - m2 * l2 * theta2_dot**2 * np.sin(delta_theta)
                          - (m1 + m2) * g * np.sin(theta1)) / den1)
    theta2_double_dot = (((m1 + m2) * (l1 * theta1_dot**2 * np.sin(delta_theta) - g * np.sin(theta2) + g * np.sin(theta1) * np.cos(delta_theta))
                          + m2 * l2 * theta2_dot**2 * np.sin(delta_theta) * np.cos(delta_theta)) / den2)
    
    return np.array([theta1_dot, theta2_dot, theta1_double_dot, theta2_double_dot])

"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    2) Implémentation de la méthode Eurler
   ----------------------------------------------------------------
"""
# Implémentation de la méthode d'Euler explicite
def method_Euler(Y, h, l1, l2, m1, m2):
    """
    Calcule une approximation de la solution du système d'équations différentielles
    du pendule double en utilisant la méthode d'Euler explicite.

    Parameters:
        Y (numpy.array): Vecteur des variables d'état [θ1, θ2, θ1_dot, θ2_dot]
        h (float): Pas de temps
        l1 (float): Long de la tige1 du pendule
        l2 (float): Long de la tige2 du pendule
        m1 (float): Masse de la masse 1
        m2 (float): Masse de la masse 2

    Returns:
        numpy.array: variables d'état à l'instant t+h [θ1_new, θ2_new, θ1_dot_new, θ2_dot_new]
    """
    
    return Y + h * F(Y, l1, l2, m1, m2)


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    3 - 4) Trajectoire et conditions initiales
   ----------------------------------------------------------------
"""
## - Conditions initiales
m1 = m2 = 1                                  # Masse en kg
l1 = l2 = 1                                  # Long en m

Y0 = np.array([np.pi / 2, np.pi / 2, 0, 0])  # [θ1, θ2, θ1_dot, θ2_dot]
N = 10**3                                    # Nombre d'itérations

T = 10                                        # Durée du tracer trajectoire en secondes
h = T / N                                     # Pas de temps

## - Calcul de la trajectoire et initialisations des listes
Y = Y0
trajectoire = [Y0]
angles = []
angular_velocities = []
time_points = []

for i in range(N): #Debut de la boucle
    Y = method_Euler(Y, h, l1, l2, m1, m2)

    #stockage
    trajectoire.append(Y)
    angles.append(Y[:2])
    angular_velocities.append(Y[2:])
    time_points.append(i * h)


# Transpose
trajectoire = np.array(trajectoire).T
angles = np.array(angles).T
angular_velocities = np.array(angular_velocities).T
time_points = np.array(time_points)


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    5) Visualiser et sauvegarder
   ----------------------------------------------------------------
"""
def anglePlot(angles, time_points):
    """
    Trace les valeurs de θ1(t) et θ2(t) en fonction du t = tn, n = 0,...,N

    Parameters:
        angles (numpy.array): Tableau contenant θ1 et θ2 pour chaque instant de temps.
        time_points (numpy.array): Tableau contenant temps correspondants.

    Returns:
        None
    """
    plt.plot(time_points, angles[0], label='θ1')
    plt.plot(time_points, angles[1], label='θ2')
    plt.xlabel('Temps')
    plt.ylabel('Angles (radians)')
    plt.title('Évolution des angles θ1 et θ2 en fonction du temps')
    plt.legend()
    plt.savefig('angularEvolution.pdf')
    plt.show()

def angularVelocityPlot(angular_velocities, time_points):
    """
    Trace les valeurs des vitesses angulaires (θ1_dot(t), θ2_dot(t)) en fonction du temps

    Parameters:
        angular_velocities (numpy.array): Tableau contenant θ1_dot et θ2_dot pour chaque instant de temps
        time_points (numpy.array): Tableau contenant temps correspondants.

    Returns:
        None
    """
    plt.plot(time_points, angular_velocities[0], label='θ1_dot')
    plt.plot(time_points, angular_velocities[1], label='θ2_dot')
    plt.xlabel('Temps')
    plt.ylabel('Vitesses angulaires (radians/s)')
    plt.title('Évolution des vitesses angulaires θ1_dot et θ2_dot en fonction du temps')
    plt.legend()
    plt.savefig('angularVelocityEvolution.pdf')
    plt.show()


def plot_trajectory(angle, l1, l2):
    """
    Trace la trajectoire du double pendule en fonction des angles et des longueurs des pendules.

    Args:
        angle (array): Une liste d'angles comprenant theta1 et theta2 aux différents instants.
        l1 (float): La longueur du premier pendule.
        l2 (float): La longueur du deuxième pendule.
    """
    # Séparer les angles theta1 et theta2 de la liste d'angles
    print(angles)
    theta1 = angle[0]
    print(theta1)
    theta2 = angle[1]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label='Masse 1', color='b')
    plt.plot(x2, y2, label='Masse 2', color='r')
    plt.xlabel('Position en x')
    plt.ylabel('Position en y')
    plt.title('Trajectoire du double pendule')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory.pdf')
    plt.show()


#-------------------------------------------------------------
def update(frame, x1, y1, x2, y2, line1, line2):
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    return line1, line2

def animate_trajectory(angle, l1, l2):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2 * (l1 + l2), 2 * (l1 + l2))
    ax.set_ylim(-2 * (l1 + l2), 2 * (l1 + l2))
    ax.set_xlabel('Position en x')
    ax.set_ylabel('Position en y')
    ax.set_title('Animation de la trajectoire du double pendule')

    x1 = l1 * np.sin(angle[0])
    y1 = -l1 * np.cos(angle[0])
    x2 = l1 * np.sin(angle[0]) + l2 * np.sin(angle[1])
    y2 = -l1 * np.cos(angle[0]) - l2 * np.cos(angle[1])

    line1, = ax.plot([], [], 'b-', label='Masse 1')
    line2, = ax.plot([], [], 'r-', label='Masse 2')
    ax.legend()

    anim = FuncAnimation(fig, update, frames=len(angle), fargs=(x1, y1, x2, y2, line1, line2), interval=1, blit=True)
    plt.show()


# -------- Représentations -----
# anglePlot(angles, time_points)
# angularVelocityPlot(angular_velocities, time_points)
# plot_trajectory(angles, l1, l2)
animate_trajectory(angles, l1, l2)