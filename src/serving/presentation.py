"""Ce que l'interface montre de l'agent : tentatives au fil de l'eau, Markdown, phrase d'état.

Aucun import de Gradio ni de FastAPI : l'écran (interface.py) et l'API (étape suivante) s'appuient sur
ces mêmes fonctions, qu'on teste ici sans navigateur ni serveur.
"""

from collections.abc import Iterator, Sequence

from src.serving.agent import Tentative


def suivre_agent(graphe, question: str, schema: str) -> Iterator[list[Tentative]]:
    """Lance l'agent et renvoie l'historique des tentatives après chacune d'elles, sans attendre la fin.

    Un générateur : l'interface peut afficher la tentative 1 (et son erreur) pendant que le modèle
    prépare la correction.
    """
    # En mode "values", le premier état émis est celui de départ (aucune tentative) : l'écran
    # afficherait « 0 tentative » avant même l'appel au modèle.
    etat_initial = {"question": question, "schema": schema, "tentatives": []}
    for etat in graphe.stream(etat_initial, stream_mode="values"):
        if etat["tentatives"]:
            yield etat["tentatives"]


def tentatives_en_markdown(tentatives: Sequence[Tentative]) -> str:
    """Une section par tentative : son SQL, puis l'erreur SQLite si elle a échoué."""
    sections = []
    for i, tentative in enumerate(tentatives, start=1):
        section = f"**Tentative {i}**\n\n```sql\n{tentative.sql}\n```"
        if tentative.resultat.erreur is not None:
            # En code : un message SQLite contient des guillemets et des * que Markdown interpréterait.
            section += f"\n\n❌ Erreur : `{tentative.resultat.erreur}`"
        else:
            # Sans marqueur, seules les tentatives ratées attirent l'œil : une correction réussie passait pour un échec.
            section += "\n\n✅ Exécutée sans erreur"
        sections.append(section)
    return "\n\n".join(sections)


def resumer(tentatives: Sequence[Tentative]) -> str:
    """Phrase d'état sous le tableau : comment l'agent a fini, puis combien de lignes il montre."""
    final = tentatives[-1].resultat
    n = len(tentatives)
    if final.erreur is not None:
        return f"Échec après {n} tentative{'s' if n > 1 else ''}"

    # « Exécutée », pas « Réussie » : on sait que SQLite a accepté la requête, pas qu'elle répond à la question.
    issue = "Exécutée du premier coup" if n == 1 else f"Exécutée après correction (tentative {n})"
    nb_lignes = len(final.lignes)
    if final.tronque:
        # Le vrai total est inconnu : execute_sql s'arrête dès max_lignes + 1 lignes lues.
        lignes = f"{nb_lignes} premières lignes (résultat tronqué)"
    elif nb_lignes == 0:
        # Le modèle n'a jamais appris à refuser : une question hors sujet donne du SQL valide qui ne trouve rien.
        lignes = "aucune ligne : la question sort peut-être du périmètre de la base"
    else:
        lignes = f"{nb_lignes} ligne{'s' if nb_lignes > 1 else ''}"
    return f"{issue} · {lignes}"
