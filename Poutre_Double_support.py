import numpy as np
import matplotlib.pyplot as plt

# Proprietes geometriques
longueur_L: float = 600
b: float = 15
h: float = 30
I: float = (b * h * h * h) / 12

# Proprietes physiques
E: float = 210000  # MPa
nb_elements: int = 4
# si nb_elements est impair, on force un nombre pair pour avoir un noeud au milieu
if nb_elements % 2 != 0:
    nb_elements += 1
F = 10000

# Discretisation
longueur_elementaire = longueur_L / nb_elements
nodes = np.linspace(0, longueur_L, nb_elements + 1)

# Assemblage Kglobale
Hf = E * I / (longueur_elementaire ** 3)
L = longueur_elementaire
matKe = Hf * np.array(
    [
        [12, 6 * L, -12, 6 * L],
        [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2],
    ],
    dtype=float,
)

taille_globale = 2 * (nb_elements + 1)  # 2 ddl (w,theta) par noeud
matKg = np.zeros((taille_globale, taille_globale), float)

for i in range(nb_elements):
    offset = 2 * i
    matKg[offset : offset + 4, offset : offset + 4] += matKe

# Force : charge ponctuelle au noeud du milieu (ddl deplacement)
F_global = np.zeros(taille_globale)
noeud_milieu = nb_elements // 2
F_global[2 * noeud_milieu] = -F

# Conditions aux limites : appuis simples (w=0 aux extremites, rotations libres)
fixed_dofs = [0, taille_globale - 2]  # w du premier et du dernier noeud
free_dofs = [i for i in range(taille_globale) if i not in fixed_dofs]

matK_reduced = matKg[np.ix_(free_dofs, free_dofs)]
F_reduced = F_global[free_dofs]

# Resolution
vecteurU_reduced = np.linalg.solve(matK_reduced, F_reduced)
vecteurU_complet = np.zeros(taille_globale)
vecteurU_complet[free_dofs] = vecteurU_reduced

print(vecteurU_reduced)
print("\n")
print(vecteurU_complet)

# =========================== Contraintes ======================== #
def matB_beam(x, L):
    """Matrice B Euler-Bernoulli : kappa = B @ [w1, th1, w2, th2]."""
    xi = x / L
    return (1 / L**2) * np.array(
        [[-6 + 12 * xi, L * (-4 + 6 * xi), 6 - 12 * xi, L * (-2 + 6 * xi)]]
    )


courbures = np.zeros(nb_elements)
moments = np.zeros(nb_elements)
x_local = longueur_elementaire / 2  # milieu d'element
B_mid = matB_beam(x_local, longueur_elementaire)

for e in range(nb_elements):
    u_elem = vecteurU_complet[2 * e : 2 * e + 4]
    courbures[e] = (B_mid @ u_elem)[0]
    moments[e] = E * I * courbures[e]

print("Courbure kappa (milieu de chaque element) :", courbures)
print("Moment flechissant (milieu de chaque element) :", moments)

# =========================== Affichage ========================== #
# Deplacements verticaux w par noeud le long de la poutre
deplacements_w = vecteurU_complet[::2]
plt.plot(nodes, deplacements_w, "-o")
plt.xlabel("x (mm)")
plt.ylabel("Deplacement vertical w (mm)")
plt.title("Deplacement le long de la poutre (double appui)")
plt.grid(True)
plt.show()

# Moment flechissant au milieu de chaque element
positions_milieu = nodes[:-1] + longueur_elementaire / 2
plt.plot(positions_milieu, moments, "-o", color="tab:red")
plt.xlabel("x (mm)")
plt.ylabel("Moment flechissant M (MPa*mm)")
plt.title("Moment flechissant le long de la poutre (points milieux)")
plt.grid(True)
plt.show()
