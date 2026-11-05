"""Analyse du fine-tuning QLoRA : recalcule les chiffres cités dans le README.

Lancer depuis la racine du repo :
    uv run python -m src.text2sql.analyse_finetuning

La prédiction et la règle de décision du plan 2×2 ont été figées avant de voir les scores, et le
split de test n'a servi qu'une fois, après ce choix. Les sections exploratoires ont été choisies
après coup : à lire comme des pistes, pas comme des preuves.
"""

from statistics import mean, stdev

from .analyse_ablation import afficher_largeurs, charger, holm, mcnemar

GRAINES = [42, 1, 2]
SEUIL = 0.05
METRIQUES = [("valide", "SQL valide"), ("correct", "EX"), ("correct_relache", "EX relâchée")]


def nb_justes(run: list[dict]) -> int:
    return sum(e["correct"] for e in run)


def prediction_confirmee(
    gains_base: list[int], gains_lora: list[int], holm_base: list[float], holm_lora: list[float]
) -> bool:
    """Le few-shot aide le modèle de base mais apporte moins au LoRA (les 3 conditions figées)."""
    plus_grand_partout = all(b > l for b, l in zip(gains_base, gains_lora))
    # Un gain doit être positif : une baisse significative ne confirmerait pas que le few-shot aide.
    base_significative = sum(g > 0 and p < SEUIL for g, p in zip(gains_base, holm_base)) >= 2
    # Sans condition sur le signe : la prédiction exclut tout effet net sur le LoRA, gain ou perte.
    lora_stable = all(p >= SEUIL for p in holm_lora)
    return plus_grand_partout and base_significative and lora_stable


def lora_5shot_retenu(ex_0shot: int, ex_5shot: list[int], gains: list[int], holm_lora: list[float]) -> bool:
    """Règle de décision figée : sinon, le 0-shot l'emporte, car il est plus simple et plus rapide."""
    significatifs = sum(g > 0 and p < SEUIL for g, p in zip(gains, holm_lora))
    return mean(ex_5shot) > ex_0shot and significatifs >= 2


def verifier_alignement(data: dict[str, list[dict]]) -> None:
    # McNemar est apparié : un décalage entre les fichiers fausserait tout sans lever d'erreur.
    questions = [[e["question"] for e in run] for run in data.values()]
    assert all(q == questions[0] for q in questions), "questions non alignées"


def afficher_test() -> None:
    test = {
        "base": charger("mlx4bit_0shot_test_prompt_v2"),
        "lora": charger("mlx4bit_lora_v1_iter440_0shot_test_prompt_v2"),
    }
    verifier_alignement(test)

    print("=== Test final, 0-shot (sur 100) ===")
    for cle, titre in METRIQUES:
        gains, pertes, p = mcnemar(test["base"], test["lora"], cle)
        base, lora = (sum(e[cle] for e in test[nom]) for nom in ("base", "lora"))
        print(f"{titre:12} base={base:3} lora={lora:3} +{gains}/-{pertes} p={p:.2g}")

    print("\n=== Test : nombre de colonnes prédit vs gold (exploratoire) ===")
    afficher_largeurs(test)


def afficher_plan() -> None:
    base_5 = [charger(f"mlx4bit_5shot_s{g}_validation_prompt_v2") for g in GRAINES]
    lora_5 = [charger(f"mlx4bit_lora_v1_iter440_5shot_s{g}_validation_prompt_v2") for g in GRAINES]
    base_0 = charger("mlx4bit_0shot_validation_prompt_v2")
    lora_0 = charger("mlx4bit_lora_v1_iter440_0shot_validation_prompt_v2")
    base_16 = charger("hf_0shot_validation_prompt_v2")

    runs = {"16bits_0": base_16, "base_0": base_0, "lora_0": lora_0}
    runs |= {f"base_5_s{g}": run for g, run in zip(GRAINES, base_5)}
    runs |= {f"lora_5_s{g}": run for g, run in zip(GRAINES, lora_5)}
    verifier_alignement(runs)

    print("\n=== Validation : scores (sur 100) ===")
    for nom, run in runs.items():
        print(
            f"{nom:11} valide={sum(e['valide'] for e in run):3} EX={nb_justes(run):3} "
            f"relâchée={sum(e['correct_relache'] for e in run):3}"
        )

    gains, pertes, p = mcnemar(base_16, base_0, "correct")
    print(f"\nPassage de 16 à 4 bits (EX) : +{gains}/-{pertes} p={p:.2f}")

    # Holm porte sur les 6 tests du plan, pas sur chaque modèle : c'est ce qui a été figé.
    tests = [mcnemar(base_0, run, "correct") for run in base_5] + [mcnemar(lora_0, run, "correct") for run in lora_5]
    ajustees = holm([p for _, _, p in tests])
    holm_base, holm_lora = ajustees[:3], ajustees[3:]

    ex_base, ex_lora = [nb_justes(run) for run in base_5], [nb_justes(run) for run in lora_5]
    gains_base = [ex - nb_justes(base_0) for ex in ex_base]
    gains_lora = [ex - nb_justes(lora_0) for ex in ex_lora]

    print("\n=== Plan 2×2 : 5-shot vs 0-shot du même modèle (EX, Holm sur 6 tests) ===")
    lignes = [("base", g, t, h) for g, t, h in zip(GRAINES, tests[:3], holm_base)]
    lignes += [("lora", g, t, h) for g, t, h in zip(GRAINES, tests[3:], holm_lora)]
    for modele, graine, (plus, moins, p), p_holm in lignes:
        print(f"{modele} graine {graine:2} : +{plus}/-{moins} bascules={plus + moins:2} p={p:.4f} Holm={p_holm:.4f}")

    print(f"\nBase 5-shot : {mean(ex_base):.1f} ± {stdev(ex_base):.1f} (0-shot : {nb_justes(base_0)})")
    print(f"LoRA 5-shot : {mean(ex_lora):.1f} ± {stdev(ex_lora):.1f} (0-shot : {nb_justes(lora_0)})")
    print(f"Gain base − gain LoRA par graine : {[b - l for b, l in zip(gains_base, gains_lora)]}")

    print("\n=== Règles figées ===")
    print(f"Prédiction confirmée : {prediction_confirmee(gains_base, gains_lora, holm_base, holm_lora)}")
    retenu = lora_5shot_retenu(nb_justes(lora_0), ex_lora, gains_lora, holm_lora)
    print(f"Configuration envoyée au test : {'LoRA 5-shot' if retenu else 'LoRA 0-shot'}")

    print("\n=== Validation : nombre de colonnes prédit vs gold (exploratoire) ===")
    afficher_largeurs({"base_0": base_0, "lora_0": lora_0})

    print("\n=== Validation : réponses avec saut de ligne (exploratoire) ===")
    for nom, run in (("base_0", base_0), ("lora_0", lora_0)):
        print(f"{nom:11} {sum(chr(10) in e['raw_output'] for e in run)}/{len(run)}")


def main() -> None:
    afficher_test()
    afficher_plan()


if __name__ == "__main__":
    main()
