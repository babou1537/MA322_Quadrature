""" MA322 - Quadrature & resolution numerique & d’EDO
Projet : modelisation des systemes chaotiques par le pendule double 

Membre du Groupe :
    - Maëva AGUESSY
    - Chloé TONSO
    - Baptiste MATHIEU

Classe 3PSB1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    INITIALISATION
    ----------------------------------------------------------------
"""
## - Conditions initiales
m1 = m2 = 1                                  # Masse en kg
l1 = 1
l2 = 1                                  # Long en m

Y0 = np.array([np.pi / 2, np.pi / 2, 0, 0])  # [θ1, θ2, θ1_dot, θ2_dot]
N = 10**3                                   # Nombre d'itérations
T = 10                                        # Durée du tracer trajectoire en secondes
h = T / N                                     # Pas de temps



"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    1) Implémenter une fonction F 
    ----------------------------------------------------------------
"""
# Définition de la fonction F
def F(Y, l1, l2, m1, m2):
    g = 9.81

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
    3 - 4) Trajectoire
    ----------------------------------------------------------------
"""
#------------------------------------------------------------------------------------
## - Calcul de la trajectoire et initialisations des listes
print("Calcul de la trajectoire avec la méthode d'Euler explicite...")
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

print("Calcul de la trajectoire avec la méthode d'Euler explicite terminé.")


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    5) Visualiser et sauvegarder
    ----------------------------------------------------------------
"""
def anglePlot(angles, time_points, alpha=None):
    """
    Trace les valeurs de θ1(t) et θ2(t) en fonction du t = tn, n = 0,...,N

    Parameters:
        angles (numpy.array): Tableau contenant θ1 et θ2 pour chaque instant de temps.
        time_points (numpy.array): Tableau contenant temps correspondants.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(time_points, angles[0], label='θ1')
    plt.plot(time_points, angles[1], label='θ2')
    plt.xlabel('Temps')
    plt.ylabel('Angles (radians)')
    plt.title('Évolution des angles θ1 et θ2 en fonction du temps')
    plt.legend()

    # if alpha != None:
    #     # plt.suptitle(f"- Pour Alpha = {str(alpha)} -")
    #     # plt.savefig(f'FIG4_{alpha}_angularEvolution.png', dpi=600)

    #     plt.suptitle(f"- {str(alpha)} -")
    #     plt.savefig('FIG1-2_angularEvolution.png', dpi=600)


    # plt.savefig('angularEvolution.pdf')
    plt.show()

def angularVelocityPlot(angular_velocities, time_points, alpha=None):
    """
    Trace les valeurs des vitesses angulaires (θ1_dot(t), θ2_dot(t)) en fonction du temps

    Parameters:
        angular_velocities (numpy.array): Tableau contenant θ1_dot et θ2_dot pour chaque instant de temps
        time_points (numpy.array): Tableau contenant temps correspondants.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(time_points, angular_velocities[0], label='θ1_dot')
    plt.plot(time_points, angular_velocities[1], label='θ2_dot')
    plt.xlabel('Temps')
    plt.ylabel('Vitesses angulaires (radians/s)')
    plt.title('Évolution des vitesses angulaires θ1_dot et θ2_dot en fonction du temps')
    plt.legend()

    # if alpha != None:
        # plt.suptitle(f"- Pour Alpha = {str(alpha)} -")
        # plt.savefig(f'FIG4_{alpha}angularVelocityEvolution.png', dpi=600)

        # plt.suptitle(f"- {str(alpha)} -")
        # plt.savefig('FIG1-2_angularVelocityEvolution.png', dpi=600)

    # plt.savefig('angularVelocityEvolution.pdf')
    plt.show()


def calcul_trajectoire(angle,l1,l2):
    # Séparer les angles theta1 et theta2 de la liste d'angles
    # print(angles)
    theta1 = angle[0]
    # print(theta1)
    theta2 = angle[1]

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    return x1,y1,x2,y2

def plot_trajectory(angle,l1,l2, subtitle=None):
    """
    Trace la trajectoire du double pendule en fonction des angles et des longueurs des pendules.

    Args:
        angle (array): Une liste d'angles comprenant theta1 et theta2 aux différents instants.
        l1 (float): La longueur du premier pendule.
        l2 (float): La longueur du deuxième pendule.
    """
    x1,y1,x2,y2 = calcul_trajectoire(angle,l1,l2)

    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label='Masse 1', color='b')
    plt.plot(x2, y2, label='Masse 2', color='r')
    plt.xlabel('Position en x')
    plt.ylabel('Position en y')
    plt.title('Trajectoire du double pendule')
    plt.legend()
    plt.grid(True)

    # if subtitle != None:
    #     plt.suptitle(subtitle)
        # plt.savefig('FIG2_trajectoire_Verlet.png', dpi=600)
    # plt.savefig('trajectory.pdf')

    plt.show()


#-----------------ANIMATION--------------------------------------------
def update(frame, angle, line1, line2, l1, l2, mass1, mass2, line_origin_mass1, line_mass1_mass2, time_text):
    x1 = l1 * np.sin(angle[0][:frame+1])
    y1 = -l1 * np.cos(angle[0][:frame+1])
    x2 = l1 * np.sin(angle[0][:frame+1]) + l2 * np.sin(angle[1][:frame+1])
    y2 = -l1 * np.cos(angle[0][:frame+1]) - l2 * np.cos(angle[1][:frame+1])

    line1.set_data(x1, y1)
    line2.set_data(x2, y2)

    mass1.set_data(np.array([x1[-1]]), np.array([y1[-1]]))  # Position de la masse 1
    mass2.set_data(np.array([x2[-1]]), np.array([y2[-1]]))  # Position de la masse 2

    line_origin_mass1.set_data([0, x1[-1]], [0, y1[-1]])
    line_mass1_mass2.set_data([x1[-1], x2[-1]], [y1[-1], y2[-1]])

    time_text.set_text(f'Time : {frame * (T / N):.2f} sec')

    return line1, line2, mass1, mass2, line_origin_mass1, line_mass1_mass2, time_text

def animate_trajectory(angle, l1, l2, methode):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2 * (l1 + l2), 2 * (l1 + l2))
    ax.set_ylim(-2 * (l1 + l2), 2 * (l1 + l2))
    ax.set_xlabel('Position en x')
    ax.set_ylabel('Position en y')
    ax.set_title(f'Animation de la trajectoire du double pendule\n- Méthode de {methode} - ')

    line1, = ax.plot([], [], 'b-', label='Masse 1')
    line2, = ax.plot([], [], 'r-', label='Masse 2')
    ax.legend()

    # Initialisation des positions des masses
    mass1, = ax.plot([], [], 'bo', markersize=10)  # Masse 1
    mass2, = ax.plot([], [], 'ro', markersize=10)  # Masse 2

    # Initialisation des lignes reliant les origines aux masses
    line_origin_mass1, = ax.plot([], [], 'b--', lw=1)  # De l'origine à la masse 1
    line_mass1_mass2, = ax.plot([], [], 'b--', lw=1)  # De la masse 1 à la masse 2

    # Texte pour afficher le temps écoulé
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    anim = FuncAnimation(fig, update, frames=len(angle[0]), fargs=(angle, line1, line2, l1, l2, mass1, mass2, line_origin_mass1, line_mass1_mass2, time_text), interval=(T * 1000) / N, blit=True, repeat=False)
    plt.show()


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    6) Energies
    ----------------------------------------------------------------
"""
def energyCompute(Y, l1, l2, m1, m2):
    g = 9.81
    theta1, theta2, theta1_dot, theta2_dot = Y
    ET = np.zeros(len(angles[0]))

    for i in range(len(angles[0])):
        Y = angles[:, i]
        # Calcul de l'énergie cinétique
        Ec = 0.5 * m1 * (l1 * theta1_dot)**2 + 0.5 * m2 * ((l1 * theta1_dot)**2 + (l2 * theta2_dot)**2 + 2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
        
        # Calcul de l'énergie potentielle
        Ep = -m1 * g * l1 * np.cos(theta1) - m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))

        ET[i] = Ec + Ep

    return ET


def energyEvolution(ET, time_points):


    plt.plot(time_points, ET)
    plt.xlabel('Temps (s)')
    plt.ylabel('Énergie totale (J)')
    plt.title('Évolution de l\'énergie totale du pendule double')
    plt.grid(True)
    plt.show()

# Calcul de l'énergie totale
ET = energyCompute(Y, l1, l2, m1, m2)

# Tracer l'évolution de l'énergie totale
time_points = np.linspace(0, T, len(ET))


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    7) Méthode de Runge-Kutta d'ordre 4 (RK4)
    ----------------------------------------------------------------
"""
def runge_kutta_4(Y, h, l1, l2, m1, m2):
    """
    Applique la méthode de Runge-Kutta d'ordre 4 pour résoudre le problème de Cauchy du pendule double.

    Parameters:
        Y (numpy.array): Vecteur des variables d'état [θ1, θ2, θ1_dot, θ2_dot]
        h (float): Pas de temps
        l1 (float): Longueur de la première tige du pendule
        l2 (float): Longueur de la deuxième tige du pendule
        m1 (float): Masse de la première masse
        m2 (float): Masse de la deuxième masse

    Returns:
        numpy.array: variables d'état à l'instant t+h [θ1_new, θ2_new, θ1_dot_new, θ2_dot_new]
    """
    k1 = h * F(Y, l1, l2, m1, m2)
    k2 = h * F(Y + 0.5 * k1, l1, l2, m1, m2)
    k3 = h * F(Y + 0.5 * k2, l1, l2, m1, m2)
    k4 = h * F(Y + k3, l1, l2, m1, m2)
    
    return Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Calcul de la trajectoire avec la méthode de Runge-Kutta d'ordre 4
print("Calcul de la trajectoire avec la méthode de Runge-Kutta d'ordre 4...")
Y = Y0
trajectory_rk4 = [Y0]
angles_rk4 = []
angular_velocities_rk4 = []
time_points_rk4 = []

for i in range(N):
    Y = runge_kutta_4(Y, h, l1, l2, m1, m2)

    trajectory_rk4.append(Y)
    angles_rk4.append(Y[:2])
    angular_velocities_rk4.append(Y[2:])
    time_points_rk4.append(i * h)


# Transpose de la trajectoire
trajectory_rk4 = np.array(trajectory_rk4).T
angles_rk4 = np.array(angles_rk4).T
angular_velocities_rk4 = np.array(angular_velocities_rk4).T
time_points_rk4 = np.array(time_points_rk4)
print("Calcul de la trajectoire avec la méthode de Runge-Kutta d'ordre 4 terminé.")



"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    III) Méthode de Verlet
    ----------------------------------------------------------------
"""
def s1(Y, l1, l2, m1, m2):
    g = 9.81
    y1, y2, y3, y4 = Y
    
    num1 = ((-m2 * l1 * y3**2 * np.sin(y1 - y2) * np.cos(y1 - y2)) + (g * m2 * np.sin(y2) * np.cos(y1 - y2)) 
            - (m2 * l2 * y4**2 * np.sin(y1 - y2)) - (g * (m2 + m1) * np.sin(y1)))
    
    den1 = (l1 * (m1 + m2) - m2 * l1 * np.cos(y1 - y2)**2)
    s1_value = num1 / den1
    
    return s1_value

def s2(Y, l1, l2, m1, m2):
    g = 9.81
    y1, y2, y3, y4 = Y
    
    num2 = ((m2 * l2 * y4**2 * np.sin(y1 - y2) * np.cos(y1 - y2)) + (g * (m1+m2) * np.sin(y1) * np.cos(y1 - y2)) 
            + ((m1+m2) * l1 * y3**2 * np.sin(y1 - y2)) - (g * (m2 + m1) * np.sin(y2)))
    
    den2 = (l2 * (m1 + m2) - m2 * l2 * np.cos(y1 - y2)**2)
    s2_value = num2 / den2
    
    return s2_value


def VerletMethod(Y, h, l1, l2, m1, m2):
    y1, y2, y3, y4 = Y
    
    # Calcul de y1 et y2 à l'instant tn+1
    y1_new = y1 + h * y3 + (h**2 / 2) * s1(Y, l1, l2, m1, m2)
    y2_new = y2 + h * y4 + (h**2 / 2) * s2(Y, l1, l2, m1, m2)
    
    # Vecteur intermédiaire w(tn) = (y1(tn+1), y2(tn+1), y3(tn), y4(tn))
    w = np.array([y1_new, y2_new, y3, y4])
    
    # Calcul de y3 et y4 à l'instant tn+1
    y3_new = y3 + (h / 2) * (s1(w, l1, l2, m1, m2) + s1(Y, l1, l2, m1, m2))
    y4_new = y4 + (h / 2) * (s2(w, l1, l2, m1, m2) + s2(Y, l1, l2, m1, m2))
    
    return np.array([y1_new, y2_new, y3_new, y4_new])


# Calcul de la trajectoire avec la méthode de Verlet
print("Calcul de la trajectoire avec la méthode de Verlet...")
trajectoire_verlet = [Y0]
angles_verlet = []
time_points_verlet = []

Y_verlet = Y0
for i in range(N):
    Y_verlet = VerletMethod(Y_verlet, h, l1, l2, m1, m2)
    trajectoire_verlet.append(Y_verlet)
    angles_verlet.append(Y_verlet[:2])
    time_points_verlet.append(i * h)

# Conversion en tableaux numpy
trajectoire_verlet = np.array(trajectoire_verlet).T
angles_verlet = np.array(angles_verlet).T
time_points_verlet = np.array(time_points_verlet)
print("Calcul de la trajectoire avec la méthode de Verlet terminé.")



"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    III-3) Comparaisons des 3 méthodes
    ----------------------------------------------------------------
"""
def comparaisonPlot(trajectoire_euler, trajectoire_rk4, trajectoire_verlet):
    """
    Trace les trajectoires calculées par les trois méthodes (Euler explicite, Runge-Kutta 4, Verlet)
    sur un même graphique avec les deux masses représentées.

    Args:
        trajectoire_euler (tuple): Tuple contenant les coordonnées (x1, y1, x2, y2) de la trajectoire calculée
                                    par Euler explicite.
        trajectoire_rk4 (tuple): Tuple contenant les coordonnées (x1, y1, x2, y2) de la trajectoire calculée
                                  par Runge-Kutta 4.
        trajectoire_verlet (tuple): Tuple contenant les coordonnées (x1, y1, x2, y2) de la trajectoire calculée
                                      par la méthode de Verlet.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))

    # Trajectoire avec Euler explicite
    plt.plot(trajectoire_euler[0], trajectoire_euler[1], label='Euler Explicite - Masse 1', color='b')
    plt.plot(trajectoire_euler[2], trajectoire_euler[3], label='Euler Explicite - Masse 2', color='b', linestyle='--')

    # Trajectoire avec Runge-Kutta 4
    plt.plot(trajectoire_rk4[0], trajectoire_rk4[1], label='Runge-Kutta 4 - Masse 1', color='r')
    plt.plot(trajectoire_rk4[2], trajectoire_rk4[3], label='Runge-Kutta 4 - Masse 2', color='r', linestyle='--')

    # Trajectoire avec Verlet
    plt.plot(trajectoire_verlet[0], trajectoire_verlet[1], label='Verlet - Masse 1', color='g')
    plt.plot(trajectoire_verlet[2], trajectoire_verlet[3], label='Verlet - Masse 2', color='g', linestyle='--')

    plt.xlabel('Position en x')
    plt.ylabel('Position en y')
    plt.title('Comparaison des trajectoires avec différentes méthodes')
    plt.legend()
    plt.grid(True)

    # plt.savefig('FIG3_compraisonMethodes.png', dpi=600)
    plt.show()



def animate_trajectory_comparison(angle_euler, angle_rk4, angle_verlet, l1, l2, T, N):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
    ax.set_ylim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
    ax.set_xlabel('Position en x')
    ax.set_ylabel('Position en y')
    ax.set_title('Animation de la trajectoire du double pendule')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    line1_euler, = ax.plot([], [], 'b-', label='Méthode Euler')
    line2_euler, = ax.plot([], [], 'b-', label='_nolegend_', alpha=0.5)
    mass1_euler, = ax.plot([], [], 'bo', markersize=10)
    mass2_euler, = ax.plot([], [], 'bo', markersize=10)
    line_origin_mass1_euler, = ax.plot([], [], 'b--', lw=1)
    line_mass1_mass2_euler, = ax.plot([], [], 'b--', lw=1)

    line1_rk4, = ax.plot([], [], 'r-', label='Méthode RK4')
    line2_rk4, = ax.plot([], [], 'r-', label='_nolegend_', alpha=0.5)
    mass1_rk4, = ax.plot([], [], 'ro', markersize=10)
    mass2_rk4, = ax.plot([], [], 'ro', markersize=10)
    line_origin_mass1_rk4, = ax.plot([], [], 'r--', lw=1)
    line_mass1_mass2_rk4, = ax.plot([], [], 'r--', lw=1)

    line1_verlet, = ax.plot([], [], 'g-', label='Méthode Verlet')
    line2_verlet, = ax.plot([], [], 'g-', label='_nolegend_', alpha=0.5)
    mass1_verlet, = ax.plot([], [], 'go', markersize=10)
    mass2_verlet, = ax.plot([], [], 'go', markersize=10)
    line_origin_mass1_verlet, = ax.plot([], [], 'g--', lw=1)
    line_mass1_mass2_verlet, = ax.plot([], [], 'g--', lw=1)

    

    # Initialisation des positions des masses
    def init_comparison():
        line1_euler.set_data([], [])
        line2_euler.set_data([], [])
        mass1_euler.set_data([], [])
        mass2_euler.set_data([], [])
        line_origin_mass1_euler.set_data([], [])
        line_mass1_mass2_euler.set_data([], [])

        line1_rk4.set_data([], [])
        line2_rk4.set_data([], [])
        mass1_rk4.set_data([], [])
        mass2_rk4.set_data([], [])
        line_origin_mass1_rk4.set_data([], [])
        line_mass1_mass2_rk4.set_data([], [])

        line1_verlet.set_data([], [])
        line2_verlet.set_data([], [])
        mass1_verlet.set_data([], [])
        mass2_verlet.set_data([], [])
        line_origin_mass1_verlet.set_data([], [])
        line_mass1_mass2_verlet.set_data([], [])

        return (line1_euler, line2_euler, mass1_euler, mass2_euler, line_origin_mass1_euler, line_mass1_mass2_euler,
                line1_rk4, line2_rk4, mass1_rk4, mass2_rk4, line_origin_mass1_rk4, line_mass1_mass2_rk4,
                line1_verlet, line2_verlet, mass1_verlet, mass2_verlet, line_origin_mass1_verlet, line_mass1_mass2_verlet)

    # Mise à jour des positions des masses et des fils
    def update_comparison(frame):
        # Méthode Euler Explicite
        x1_euler = l1 * np.sin(angle_euler[0][:frame+1])
        y1_euler = -l1 * np.cos(angle_euler[0][:frame+1])
        x2_euler = l1 * np.sin(angle_euler[0][:frame+1]) + l2 * np.sin(angle_euler[1][:frame+1])
        y2_euler = -l1 * np.cos(angle_euler[0][:frame+1]) - l2 * np.cos(angle_euler[1][:frame+1])

        line1_euler.set_data(np.array([x1_euler, y1_euler]))
        line2_euler.set_data(np.array([x2_euler, y2_euler]))
        mass1_euler.set_data(np.array([x1_euler[-1]]), np.array([y1_euler[-1]]))
        mass2_euler.set_data(np.array([x2_euler[-1]]), np.array([y2_euler[-1]]))
        line_origin_mass1_euler.set_data([0, x1_euler[-1]], [0, y1_euler[-1]])
        line_mass1_mass2_euler.set_data([x1_euler[-1], x2_euler[-1]], [y1_euler[-1], y2_euler[-1]])

        # Méthode RK4
        x1_rk4 = l1 * np.sin(angle_rk4[0][:frame+1])
        y1_rk4 = -l1 * np.cos(angle_rk4[0][:frame+1])
        x2_rk4 = l1 * np.sin(angle_rk4[0][:frame+1]) + l2 * np.sin(angle_rk4[1][:frame+1])
        y2_rk4 = -l1 * np.cos(angle_rk4[0][:frame+1]) - l2 * np.cos(angle_rk4[1][:frame+1])

        line1_rk4.set_data(np.array([x1_rk4, y1_rk4]))
        line2_rk4.set_data(np.array([x2_rk4, y2_rk4]))
        mass1_rk4.set_data(np.array([x1_rk4[-1]]), np.array([y1_rk4[-1]]))
        mass2_rk4.set_data(np.array([x2_rk4[-1]]), np.array([y2_rk4[-1]]))
        line_origin_mass1_rk4.set_data([0, x1_rk4[-1]], [0, y1_rk4[-1]])
        line_mass1_mass2_rk4.set_data([x1_rk4[-1], x2_rk4[-1]], [y1_rk4[-1], y2_rk4[-1]])

        # Méthode Verlet
        x1_verlet = l1 * np.sin(angle_verlet[0][:frame+1])
        y1_verlet = -l1 * np.cos(angle_verlet[0][:frame+1])
        x2_verlet = l1 * np.sin(angle_verlet[0][:frame+1]) + l2 * np.sin(angle_verlet[1][:frame+1])
        y2_verlet = -l1 * np.cos(angle_verlet[0][:frame+1]) - l2 * np.cos(angle_verlet[1][:frame+1])

        line1_verlet.set_data(np.array([x1_verlet, y1_verlet]))
        line2_verlet.set_data(np.array([x2_verlet, y2_verlet]))
        mass1_verlet.set_data(np.array([x1_verlet[-1]]), np.array([y1_verlet[-1]]))
        mass2_verlet.set_data(np.array([x2_verlet[-1]]), np.array([y2_verlet[-1]])) 
        line_origin_mass1_verlet.set_data([0, x1_verlet[-1]], [0, y1_verlet[-1]])
        line_mass1_mass2_verlet.set_data([x1_verlet[-1], x2_verlet[-1]], [y1_verlet[-1], y2_verlet[-1]])

        # Temps
        time_text.set_text(f'Time : {frame * (T / N):.2f} sec')

        return (line1_euler, line2_euler, mass1_euler, mass2_euler, line_origin_mass1_euler, line_mass1_mass2_euler,
                line1_rk4, line2_rk4, mass1_rk4, mass2_rk4, line_origin_mass1_rk4, line_mass1_mass2_rk4,
                line1_verlet, line2_verlet, mass1_verlet, mass2_verlet, line_origin_mass1_verlet, line_mass1_mass2_verlet, time_text)

    ani = FuncAnimation(fig, update_comparison, frames=len(angle_euler[0]), init_func=init_comparison, interval=(T * 1000) / N, blit=True, repeat=False)
    plt.legend(loc='upper right')
    plt.show()


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    IV) Mouvement chaotique et sensibilite par rapport aux conditions
    initiales
    ----------------------------------------------------------------
"""
#1/
def PhaseSpacePlot(angles, angular_velocities, alpha=None):
    """
    Trace les courbes Ci dans le plan R2 définies par (θi(t), θ'i(t)) : t ∈ [0, T].

    Parameters:
        angles (numpy.array): Tableau contenant θ1 et θ2 pour chaque instant de temps.
        angular_velocities (numpy.array): Tableau contenant θ1_dot et θ2_dot pour chaque instant de temps.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(angles[0], angular_velocities[0], label=r'$C_1$', color='b')
    ax.plot(angles[1], angular_velocities[1], label=r'$C_2$', color='r')
    
    ax.set_xlabel(r'Angle $\theta$')
    ax.set_ylabel(r'Vitesse angulaire $\dot{\theta}$')
    ax.set_title('Espace des phases')
    ax.legend()

    if alpha != None:
        plt.suptitle(f"- Pour Alpha = {str(alpha)} -")
        # plt.savefig(f'FIG4_{alpha}_phase.png', dpi=600)
    
    plt.show()

#2/
def calculer_trajectoire_pour_conditions_initiales(Y0, T, N, l1, l2, m1, m2, method):
    """
    Calcule la trajectoire du double pendule pour des conditions initiales données en utilisant une méthode numérique spécifiée.

    Parameters:
        Y0 (numpy.array): Vecteur des conditions initiales [θ1, θ2, θ1_dot, θ2_dot]
        T (float): Durée de la simulation en secondes
        N (int): Nombre d'itérations
        l1 (float): Longueur de la première tige du pendule
        l2 (float): Longueur de la deuxième tige du pendule
        m1 (float): Masse de la première masse
        m2 (float): Masse de la deuxième masse
        method (function): Méthode numérique à utiliser pour calculer la trajectoire (e.g., method_Euler, runge_kutta_4, VerletMethod)

    Returns:
        tuple: Contient les angles, les vitesses angulaires et les points temporels calculés
    """
    h = T / N
    Y = Y0
    trajectoire = [Y0]
    angles = []
    angular_velocities = []
    time_points = []

    for i in range(N):
        Y = method(Y, h, l1, l2, m1, m2)
        trajectoire.append(Y)
        angles.append(Y[:2])
        angular_velocities.append(Y[2:])
        time_points.append(i * h)

    # Conversion en tableaux numpy
    trajectoire = np.array(trajectoire).T
    angles = np.array(angles).T
    angular_velocities = np.array(angular_velocities).T
    time_points = np.array(time_points)

    return angles, angular_velocities, time_points


"""xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    Représentations
    ----------------------------------------------------------------
"""
while True:
    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("Sélectionnez une partie du code à exécuter :")
    print("1. Méthode Euler Explicite")
    print("2. Méthode RK4")
    print("3. Méthode de Verlet")
    print("4. Comparaison des 3 méthodes")
    print("5. Partie 4 - Chaos")
    print("0. Quitter")
    
    choice = input("\n>> Votre choix : ")
    
    if choice == "1":
        """METHODE Euler Explicite"""
        print("Méthode Euler Explicite sélectionnée")
        anglePlot(angles, time_points, "Méthode Euler Explicite")
        angularVelocityPlot(angular_velocities, time_points, "Méthode Euler Explicite")
        plot_trajectory(angles, l1, l2, "Méthode Euler Explicite")
        print("Animation de la trajectoire avec la méthode d'Euler explicite...")
        animate_trajectory(angles, l1, l2,"Euler Explicite")
        print("Animation de la trajectoire avec la méthode d'Euler explicite terminée.")
        print()
        energyEvolution(ET, time_points)
        
    elif choice == "2":
        """METHODE RK4"""
        print("Méthode RK4 sélectionnée")
        plot_trajectory(angles_rk4, l1, l2, "Méthode Runge-Kutta d'ordre 4")
        print("Animation de la trajectoire avec la méthode de Runge-Kutta d'ordre 4...")
        animate_trajectory(angles_rk4, l1, l2, "RK4")
        print("Animation de la trajectoire avec la méthode de Runge-Kutta d'ordre 4 terminée.\n")
        
    elif choice == "3":
        """METHODE de Verlet"""
        print("Méthode de Verlet sélectionnée")
        plot_trajectory(angles_verlet, l1, l2, "Méthode de Verlet")
        print("Animation de la trajectoire avec la méthode de Verlet...")
        animate_trajectory(angles_verlet, l1, l2, "Verlet")
        print("Animation de la trajectoire avec la méthode de Verlet terminée.\n")

    elif choice == "4":
        """COMPARAISON des 3 méthodes"""
        print("Comparaison des 3 méthodes sélectionnée")
        comparaisonPlot(calcul_trajectoire(angles,l1,l2), calcul_trajectoire(angles_rk4,l1,l2), calcul_trajectoire(angles_verlet,l1,l2))
        print("Animation de la trajectoire - COMPARAISON...")
        animate_trajectory_comparison(angles, angles_rk4, angles_verlet, l1, l2, T, N)
        print("Animation de la trajectoire - COMPARAISON terminée.")
        
    elif choice == "5":
        print("Partie 4 - Chaos sélectionnée")
        # Conditions initiales et paramètres
        Y0_1 = np.array([10**-8, np.sqrt(2) * 10**-8, 0, 0])
        T = 20
        N = 1000

        # Calcul des trajectoires avec la méthode d'Euler explicite
        angles_1, angular_velocities_1, time_points_1 = calculer_trajectoire_pour_conditions_initiales(Y0_1, T, N, l1, l2, m1, m2, method_Euler)

        # Graphiques des angles et des vitesses angulaires en fonction du temps pour la première condition initiale
        anglePlot(angles_1, time_points_1)
        angularVelocityPlot(angular_velocities_1, time_points_1)

        # Nouvelle condition initiale -------------------------------------------------------
        Y0_2 = np.array([10**-8, 10**-8, 0, 0])

        # Calcul des trajectoires avec la méthode d'Euler explicite
        angles_2, angular_velocities_2, time_points_2 = calculer_trajectoire_pour_conditions_initiales(Y0_2, T, N, l1, l2, m1, m2, method_Euler)

        # Graphiques des angles et des vitesses angulaires en fonction du temps pour la deuxième condition initiale
        anglePlot(angles_2, time_points_2)
        angularVelocityPlot(angular_velocities_2, time_points_2)


        # Étude de la sensibilité pour différentes valeurs de α
        alphas = [10**-9, 10**-7, 0.005, 0.05, 1, 1.2, 1.5, 1.7, 1.9, 2]
        T = 20
        N = 1000

        for alpha in alphas:
            Y0 = np.array([alpha, alpha, 0, 0])
            angles, angular_velocities, time_points = calculer_trajectoire_pour_conditions_initiales(Y0, T, N, l1, l2, m1, m2, method_Euler)
            
            # Tracer les graphes des angles et des vitesses angulaires
            anglePlot(angles, time_points,alpha)
            angularVelocityPlot(angular_velocities, time_points,alpha)
            
            # Représentation dans l'espace des phases
            PhaseSpacePlot(angles, angular_velocities,alpha)
        
    elif choice == "0":
        print("Fin du programme")
        break
        
    else:
        print("Choix invalide. Veuillez sélectionner une option valide.")

