""" MA322 - Quadrature & resolution numerique & d’EDO
Projet : modelisation des syst ´ emes chaotiques par le pendule double 

Membre du Groupe :
    - Maëva AGUESSY
    - Chloé TONSO
    - Baptiste MATHIEU
    - Phuong Mai NGUON

Classe 3PSB1
"""

import numpy as np
import matplotlib.pyplot as plt

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
h = T / N                                    # Pas de temps

## - Calcul de la trajectoire et initialisations des listes
Y = Y0 
trajectoire = [Y]
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

def trajectoryPlot(trajectoire):
    """
    Trace la trajectoire des masses pour tout instant t.

    Parameters:
        trajectoire (numpy.array): Tableau contenant les coordonnées des masses m1 et m2 pour chaque instant de temps.

    Returns:
        None
    """
    plt.plot(trajectoire[0], trajectoire[1], label='Trajectoire de m1')
    plt.plot(trajectoire[2], trajectoire[3], label='Trajectoire de m2')
    plt.xlabel('Position x')
    plt.ylabel('Position y')
    plt.title('Trajectoire des masses m1 et m2')
    plt.legend()
    plt.savefig('trajectory.pdf')
    plt.show()



# -------- Représentations -----
anglePlot(angles, time_points)
angularVelocityPlot(angular_velocities, time_points)
trajectoryPlot(trajectoire)
