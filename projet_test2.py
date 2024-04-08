import numpy as np
import matplotlib.pyplot as plt

# Définition de la fonction F
def F(Y, l1, l2, m1, m2):
    g = 9.81
    theta1, theta2, theta1_dot, theta2_dot = Y
    
    # Calcul des dérivées secondes
    theta1_double_dot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (theta2_dot ** 2 * l2 + theta1_dot ** 2 * l1 * np.cos(theta1 - theta2))) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    theta2_double_dot = (2 * np.sin(theta1 - theta2) * (theta1_dot ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + theta2_dot ** 2 * l2 * m2 * np.cos(theta1 - theta2))) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    
    return np.array([theta1_dot, theta2_dot, theta1_double_dot, theta2_double_dot])


def method_Euler_explicite(F, y0, a, b, N):
    """
    Méthode d'Euler explicite pour résoudre une équation différentielle ordinaire (EDO).

    Args:
        F (function): La fonction F(Y, t) qui définit l'EDO sous la forme Y'(t) = F(Y, t).
        y0 (array): La condition initiale sous forme d'un vecteur.
        a (float): Le temps initial.
        b (float): Le temps final.
        N (int): Le nombre d'itérations.

    Returns:
        tuple: Un tuple contenant deux arrays, t et Y, où t contient les temps et Y contient les approximations de la solution.
    """
    h = (b - a) / N
    t = np.linspace(a, b, N+1)  # Utiliser linspace pour générer les points de temps uniformément espacés
    Y = np.zeros((len(y0), N+1))
    Y[:, 0] = y0  # Définir la condition initiale
    for n in range(N):
        Y[:, n+1] = Y[:, n] + h * F(Y[:, n], l1, l2, m1, m2)
    return t, Y


# Conditions initiales
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
theta1_dot_0 = 0
theta2_dot_0 = 0
Y0 = np.array([theta1_0, theta2_0, theta1_dot_0, theta2_dot_0])

# Paramètres du pendule double
m1 = m2 = 1  # kg
l1 = l2 = 1  # m

# Temps initial et final
a = 0
b = 10  # secondes

# Nombre d'itérations
N = 1000

# Résolution du système
t, Y = method_Euler_explicite(F, Y0, a, b, N)

# Affichage des résultats
print("Temps t : ", t)
print("Solution Y(t) : ", Y)




# Calcul des coordonnées (x, y) des extrémités des pendules
x1 = l1 * np.sin(Y[0])
y1 = -l1 * np.cos(Y[0])
x2 = l1 * np.sin(Y[0]) + l2 * np.sin(Y[1])
y2 = -l1 * np.cos(Y[0]) - l2 * np.cos(Y[1])

# Traçage de la trajectoire
plt.plot(x1, y1, label='Masse 1')
plt.plot(x2, y2, label='Masse 2')
plt.title('Trajectoire du pendule double')
plt.xlabel('Position horizontale')
plt.ylabel('Position verticale')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Pour assurer que les axes x et y ont la même échelle
plt.show()