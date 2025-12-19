import numpy as np
import matplotlib.pyplot as plt

# Propriétés physiques en mm #

Longueur = 1000
Section = 10
ModuleYoung = 210000

# Propriétés du problème #
nb_elements = 4 # nombre d'éléments finis de taille égales
dim = 1 # dimension probleme

F = 1000

# discrétisation
longueur_Elementaire = Longueur/nb_elements
nodes = np.linspace(0, Longueur,nb_elements+1) # crée un vecteur de position de noeuds le premier argument est la position du premier noeud, le deuxième la position du dernier noeud le troisième argument est le nombre total de noeud

# Assemblage de Kglobale
matKg = np.zeros((nb_elements+1, nb_elements+1), float)
matKe = (ModuleYoung * Section / longueur_Elementaire) * np.array([[1, -1],[-1, 1]])

for i in range(nb_elements): #On boucle sur chaque éléments de 0 à N-1 et on place les matrices élémentaires sur la matrice globale des positions e à e+2
    matKg[i : i+2, i:i+2] += matKe

# Vecteur de force 
F_global = np.zeros((len(nodes)))
F_global[-1] = F

# Conditions aux limites
matK_reduced = matKg[1:,1:] # On retire la premiere colonne et la première ligne
F_reduced = F_global[1:] # On retire la premiere ligne

print(F_reduced)
print(matKg)
# Résolution
vecteurU_reduced = np.linalg.solve(matK_reduced, F_reduced)
vecteurU_complet = np.insert(vecteurU_reduced, 0, 0) # création du vecteur complet avec u0 = 0


# Contraintes 

Sigma = ModuleYoung * np.diff(vecteurU_complet) / longueur_Elementaire # On calcul ici la différence entre chaque noeud de déplacement (n - n-1) qu'on divise par la longueur élémentaire (la meme pour chaque déplacement) et ensuite on multiplie par le module d'Young

# Affichage
print("Déplacement nodaux (mm) : ", vecteurU_complet)
print("Contrainte (MPa) : ", Sigma)


# Graphique
plt.plot(nodes, vecteurU_complet, '-o') # choisit les variables à afficher
plt.xlabel('x (mm)')
plt.ylabel('Deplacement (mm)')
plt.title('Deplacement axial de la barre')
plt.grid(True)
plt.show()