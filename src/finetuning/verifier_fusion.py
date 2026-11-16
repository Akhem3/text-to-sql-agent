"""Vérifie qu'une fusion LoRA n'a modifié que les poids ciblés, et du bon montant.

    uv run python -m src.finetuning.verifier_fusion --base <snapshot> --fusionne <dossier> --adaptateur <dossier>

Une fusion ratée produirait quand même un modèle qui se charge et qui répond : sans ce contrôle, on ne
la découvrirait qu'au score de l'évaluation, sans savoir si la cause est la fusion, la conversion GGUF
ou la quantification.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open


@dataclass
class Bilan:
    erreurs: list[str] = field(default_factory=list)
    n_cibles: int = 0
    n_inchanges: int = 0
    n_valeurs_cibles: int = 0
    n_valeurs_modifiees: int = 0
    n_valeurs_conformes: int = 0
    somme_carres_delta: float = 0.0
    somme_carres_base: float = 0.0


def indexer_tenseurs(dossier: Path) -> dict[str, Path]:
    """Associe chaque nom de tenseur à son fichier : le découpage en fichiers diffère entre deux modèles."""
    index = {}
    # glob non récursif : le snapshot contient un sous-dossier original/ qu'il ne faut pas lire.
    for chemin in sorted(dossier.glob("*.safetensors")):
        with safe_open(chemin, framework="pt") as fichier:
            index.update(dict.fromkeys(fichier.keys(), chemin))
    return index


def lire_tenseur(index: dict[str, Path], nom: str) -> torch.Tensor:
    # Lecture tenseur par tenseur : charger les deux modèles d'un coup demanderait 32 Go de mémoire.
    with safe_open(index[nom], framework="pt") as fichier:
        return fichier.get_tensor(nom)


def poids_cibles(dossier_adaptateur: Path) -> dict[str, tuple[str, str]]:
    """Renvoie {nom du poids du modèle : (nom de lora_a, nom de lora_b)} d'après l'adaptateur."""
    with safe_open(dossier_adaptateur / "adapters.safetensors", framework="pt") as fichier:
        noms = list(fichier.keys())
    prefixes = [nom.removesuffix(".lora_a") for nom in noms if nom.endswith(".lora_a")]
    return {f"{prefixe}.weight": (f"{prefixe}.lora_a", f"{prefixe}.lora_b") for prefixe in prefixes}


def calculer_fusion(base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> torch.Tensor:
    """Recalcule un poids fusionné comme mlx-lm : même formule, même moteur, même appareil.

    Le GPU est imposé car son produit matriciel diffère de celui du CPU, torch compris : sur CPU, environ
    1 % des valeurs finales changent sans que la fusion soit fausse, ce qui interdirait d'exiger l'égalité.
    """
    with mx.stream(mx.gpu):
        poids = mx.array(base.float().numpy()).astype(mx.bfloat16)
        a = mx.array(lora_a.numpy())
        b = mx.array(lora_b.numpy())
        # Le delta passe en 16 bits avant l'addition : c'est l'ordre des opérations de mlx-lm.
        delta = ((scale * b.T) @ a.T).astype(mx.bfloat16)
        resultat = (poids + delta).astype(mx.float32)
        mx.eval(resultat)
    # Passage par 32 bits car numpy ne connaît pas le bf16 ; sans perte, 32 bits contenant tout le bf16.
    return torch.from_numpy(np.array(resultat)).to(torch.bfloat16)


def controler_cible(
    nom: str,
    base: torch.Tensor,
    fusionne: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    bilan: Bilan,
) -> None:
    attendu = calculer_fusion(base, lora_a, lora_b, scale)

    n_modifiees = (fusionne != base).sum().item()
    n_conformes = (fusionne == attendu).sum().item()

    bilan.n_cibles += 1
    bilan.n_valeurs_cibles += fusionne.numel()
    bilan.n_valeurs_modifiees += n_modifiees
    bilan.n_valeurs_conformes += n_conformes
    bilan.somme_carres_delta += (fusionne.float() - base.float()).square().sum().item()
    bilan.somme_carres_base += base.float().square().sum().item()

    if n_modifiees == 0:
        bilan.erreurs.append(f"{nom} : poids ciblé, mais identique à la base")
    if n_conformes != fusionne.numel():
        accord = n_conformes / fusionne.numel()
        bilan.erreurs.append(f"{nom} : seulement {accord:.4%} des valeurs correspondent au calcul attendu")


def controler_fusion(dossier_base: Path, dossier_fusionne: Path, dossier_adaptateur: Path) -> Bilan:
    bilan = Bilan()
    index_base = indexer_tenseurs(dossier_base)
    index_fusionne = indexer_tenseurs(dossier_fusionne)
    index_lora = indexer_tenseurs(dossier_adaptateur)
    cibles = poids_cibles(dossier_adaptateur)
    config = json.loads((dossier_adaptateur / "adapter_config.json").read_text(encoding="utf-8"))
    scale = config["lora_parameters"]["scale"]

    bilan.erreurs += [f"{nom} : absent du modèle fusionné" for nom in sorted(index_base.keys() - index_fusionne.keys())]
    bilan.erreurs += [f"{nom} : absent de la base" for nom in sorted(index_fusionne.keys() - index_base.keys())]
    bilan.erreurs += [f"{nom} : poids ciblé absent de la base" for nom in sorted(cibles.keys() - index_base.keys())]

    for nom in sorted(index_base.keys() & index_fusionne.keys()):
        base = lire_tenseur(index_base, nom)
        fusionne = lire_tenseur(index_fusionne, nom)

        # Vérifiés en premier : les comparaisons suivantes n'ont pas de sens si la forme diffère, et un
        # changement de précision fausserait la quantification sans aucune erreur visible.
        if base.shape != fusionne.shape or base.dtype != fusionne.dtype:
            bilan.erreurs.append(
                f"{nom} : {base.dtype} {list(base.shape)} devenu {fusionne.dtype} {list(fusionne.shape)}"
            )
            continue

        if nom in cibles:
            nom_a, nom_b = cibles[nom]
            lora_a = lire_tenseur(index_lora, nom_a)
            lora_b = lire_tenseur(index_lora, nom_b)
            controler_cible(nom, base, fusionne, lora_a, lora_b, scale, bilan)
        elif torch.equal(base, fusionne):
            bilan.n_inchanges += 1
        else:
            bilan.erreurs.append(f"{nom} : modifié alors que l'adaptateur ne le cible pas")

    return bilan


def afficher(bilan: Bilan) -> None:
    print(f"Tenseurs ciblés contrôlés     : {bilan.n_cibles}")
    print(f"Tenseurs non ciblés identiques : {bilan.n_inchanges}")
    if bilan.n_valeurs_cibles:
        print(f"Valeurs modifiées (cibles)    : {bilan.n_valeurs_modifiees / bilan.n_valeurs_cibles:.2%}")
        print(f"Valeurs conformes au calcul   : {bilan.n_valeurs_conformes / bilan.n_valeurs_cibles:.4%}")
    if bilan.somme_carres_base:
        print(f"Norme du delta / norme de base : {(bilan.somme_carres_delta / bilan.somme_carres_base) ** 0.5:.4%}")
    print(f"Erreurs                       : {len(bilan.erreurs)}")
    for erreur in bilan.erreurs[:20]:
        print(f"  - {erreur}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fusionne", type=Path, required=True)
    parser.add_argument("--adaptateur", type=Path, required=True)
    args = parser.parse_args()

    bilan = controler_fusion(args.base, args.fusionne, args.adaptateur)
    afficher(bilan)
    if bilan.erreurs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
