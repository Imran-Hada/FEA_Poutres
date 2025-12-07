import numpy as np
import matplotlib.pyplot as plt

# Propriétés gémométriques
longueur_L : float = 600
b: float = 15
h : float = 30
I : float = (b*h*h*h)/12

# Propriétés physiques
E : float = 210000  # MPa
dim_prob :int = 2
nb_elements : int = 4
F = 10000
# Matrices
matK = np.zeros(((dim_prob*2), (dim_prob*2)))
vecU = np.zeros((dim_prob*2))
vecF = np.zeros((dim_prob*2))

# Discretisation 
longueur_elementaire = longueur_L / nb_elements
nodes = np.linspace(0, longueur_L,nb_elements+1) # crée un vecteur de position de noeuds le premier argument est la position du premier noeud, le deuxième la position du dernier noeud le troisième argument est le nombre total de noeud
# ajout des valeurs sur la matrice K

# Assemblage Kglobale
Hf = E * I / ((longueur_elementaire)**3)
# K_elementaire (Euler-Bernoulli, symetrique)
L = longueur_elementaire
matKe = Hf * np.array([
    [12,    6*L,   -12,    6*L],
    [6*L, 4*L**2, -6*L, 2*L**2],
    [-12,  -6*L,    12,   -6*L],
    [6*L, 2*L**2, -6*L, 4*L**2]
], dtype=float)

matSize = len(matKe[0])

# K_globale
taille_globale = (matSize * nb_elements) - ((nb_elements-1)*2)
matKg = np.zeros((taille_globale,taille_globale), float)


for i in range(nb_elements): #On boucle sur chaque éléments de 0 à N-1 et on place les matrices élémentaires sur la matrice globale des positions e à e+2
    offset = 2 * i # chaque élément partage 2 ddl avec le précédent
    matKg[offset : offset + 4, offset:offset+4 ] += matKe
# Force
F_global = np.zeros((2*len(nodes)))
F_global[-2] = -F  # charge appliquee sur le ddl deplacement du dernier noeud

# Matrice avec les conditions 
matK_reduced = matKg[2:,2:] # On retire les deux premieres colonnes et lignes (v1,01)
F_reduced = F_global[2:] # On retire les deux premieres lignes

# Résolution
vecteurU_reduced = np.linalg.solve(matK_reduced, F_reduced)
vecteurU_complet = np.r_[0, 0, vecteurU_reduced] # On concatene le vecteur en ajoutant deux 0 aux premières valeurs du tableau

print(vecteurU_reduced)
print("\n")
print(vecteurU_complet)
#print(matKg)

# =========================== Contraintes ======================== #

def matB_beam(x, L):
    """Matrice B Euler-Bernoulli : kappa = B @ [w1, th1, w2, th2]."""
    xi = x / L
    return (1 / L**2) * np.array([
        [-6 + 12*xi,   L*(-4 + 6*xi),   6 - 12*xi,   L*(-2 + 6*xi)]
    ])


# Courbure et moment au milieu de chaque element (B * u_elem)
courbures = np.zeros(nb_elements)
moments = np.zeros(nb_elements)
x_local = longueur_elementaire / 2  # milieu d'element
B_mid = matB_beam(x_local, longueur_elementaire)

for e in range(nb_elements):
    u_elem = vecteurU_complet[2*e:2*e+4]  # on récupère le ddl de chaque élément sur le vecteur globale
    courbures[e] = (B_mid @ u_elem)[0] # on récupère la valeure de la courbure on ajoute [0] pour recuperer le résultat sans reprendre le tableau
    moments[e] = E * I * courbures[e] 

print("Courbure kappa (milieu de chaque element) :", courbures)
print("Moment flechissant (milieu de chaque element) :", moments)

# =========================== Affichage ========================== #
# Deplacements verticaux w par noeud le long de la poutre
deplacements_w = vecteurU_complet[::2]  # w0, w1, ...
plt.plot(nodes, deplacements_w, "-o")
plt.xlabel("x (mm)")
plt.ylabel("Deplacement vertical w (mm)")
plt.title("Deplacement le long de la poutre")
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
