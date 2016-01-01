# Ceci est un bout de code qu'il faut coller à la fin du script
# qui entraine le modele.
# Cela va creer un fichier texte 'sauvegarde_MYMODEL' (a renommer en fonction du modele)
# qui contiendra le modele deja entraine

# remplacer MYMODELINSTANCE par le modèle fitté sur les données

import pickle

output=file('./sauvegarde_MYMODEL','w')
pickle.dump(MYMODELINSTANCE,output)
output.close()
