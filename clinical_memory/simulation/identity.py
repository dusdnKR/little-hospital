"""Fictional patient names. Separate randomness never changes clinical outcomes."""

import hashlib
import random

DOCTORS = ("Dr. RED", "Dr. BLUE", "Dr. YELLOW", "Dr. GREEN", "Dr. PINK")
FULL_MODELS = ("Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B")
FIRST = "Alex Avery Blair Cameron Casey Charlie Dakota Dana Devon Drew Eden Ellis Emery Finley Frankie Harper Hayden Jamie Jordan Jules Kai Kendall Lane Lee Logan Marley Morgan Noel Parker Quinn Reese Remy Riley Robin Rowan Sage Sam Sasha Sidney Skyler Taylor Terry Toby Val Winter Wren".split()
LAST = "Adams Allen Bailey Baker Bell Bennett Brooks Brown Campbell Carter Clark Collins Cooper Davis Ellis Evans Fisher Flores Foster Garcia Gray Green Hall Harris Hayes Hill Howard James Kelly Kim King Lane Lee Lewis Martin Miller Moore Morgan Morris Nelson Nguyen Park Parker Perez Perry Reed Rivera Roberts Ross Scott Smith Stone Taylor Thomas Turner Walker Ward White Wood Wright Young".split()


class Pseudonyms:
    def __init__(self, seed):
        self.seed = seed
        self.names = {}
        self.used = set()

    def get(self, case_id):
        if case_id not in self.names:
            digest = hashlib.sha256(f"{self.seed}:{case_id}".encode()).digest()
            rng = random.Random(int.from_bytes(digest, "big"))
            start = rng.randrange(len(FIRST) * len(LAST))
            for offset in range(len(FIRST) * len(LAST)):
                index = (start + offset) % (len(FIRST) * len(LAST))
                name = f"{FIRST[index // len(LAST)]} {LAST[index % len(LAST)]}"
                if name not in self.used:
                    break
            else:
                name += f" {len(self.names) + 1}"
            self.names[case_id] = name
            self.used.add(name)
        return self.names[case_id]


def doctor_label(room):
    return DOCTORS[room] + (f" ({FULL_MODELS[room]})" if room < 3 else "")
