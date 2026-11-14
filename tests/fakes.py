"""Doublures de test partagées entre plusieurs fichiers (importées, contrairement aux fixtures de conftest)."""


class FauxBackend:
    """Renvoie des réponses écrites à l'avance et garde les messages reçus, pour vérifier ce que le modèle a vu."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.recus = []

    def generate(self, messages):
        self.recus.append(messages)
        # IndexError si le code appelle le modèle plus souvent que prévu : le test échoue bruyamment.
        return self.reponses.pop(0)
