"""Empirische Abdeckungsprobe: Welche realistischen Prompts liefert FreqAI leer?

Kein Test, sondern eine Diagnose. Laedt ein kleines deklaratives Testkorpus und
haelt fest, bei welchen alltaeglichen Eingaben der Decoder leer bleibt. Die
Ausgabe zeigt die echten Lücken vor und nach einer Aenderung.
"""
import sys

sys.path.insert(0, ".")

from freqai.unpaired import UnpairedWaveModel

DECLARATIONS = [
    {"text": "Der Assistent heißt FreqAI. Der Assistent ist ein Rechenprogramm. "
             "Der Assistent hat keine eigenen Gefühle. Der Assistent kann Texte beschreiben."},
    {"text": "Das Wort Hallo ist ein Begrüßungswort. Das Wort Danke ist ein Dankeswort. "
             "Das Wort Bitte ist ein Höflichkeitswort. Das Wort Tschüss ist ein Abschiedswort."},
    {"text": "Das Befinden ist eine persönliche Wahrnehmung."},
    {"text": "Die Frequenz ist die Anzahl der Schwingungen pro Sekunde. "
             "Die Einheit der Frequenz ist das Hertz. Die Formel der Frequenz ist 1 durch die Periodendauer."},
    {"text": "Der Kehrwert der Frequenz ist die Periodendauer."},
    {"text": "Das Signal Lumor hat eine Frequenz von 19 Hertz. Das Signal Tavor hat eine Frequenz von 73 Hertz."},
]

PROMPTS = [
    "Hallo",
    "Wie heißt du?",
    "Was kannst du?",
    "Hast du Gefühle?",
    "Wie geht es dir?",
    "Danke",
    "Tschüss",
    "Was ist eine Frequenz?",
    "Welche Einheit hat die Frequenz?",
    "Wie berechnet man die Frequenz?",
    "Was ist der Kehrwert der Frequenz?",
    "Welche Frequenz hat das Signal Lumor?",
    "Welches Signal hat eine Frequenz von 73 Hertz?",
    "Erkläre mir den Zusammenhang zwischen Frequenz und Periodendauer.",
    "Was ist ein Bortelium?",
]


def main() -> None:
    model = UnpairedWaveModel(DECLARATIONS)
    empty = 0
    for index, prompt in enumerate(PROMPTS):
        output = model.generate(prompt)
        is_empty = not output["tokens"]
        empty += int(is_empty)
        print(f"{index:02d}. {prompt}")
        print(f"     reason={output.get('reason')!r} ended={output.get('ended')} "
              f"-> {output['text']!r}{'' if not is_empty else '   [LEER]'}")
    print(f"\n{empty} von {len(PROMPTS)} Antworten bleiben leer.")


if __name__ == "__main__":
    main()
