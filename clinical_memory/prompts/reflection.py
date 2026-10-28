from .feedback import INTRO, SHEET, UNVERIFIED, RULES, COLLECT

COLLECT_WORDS, DIAGNOSE_WORDS = (45, 45)
KEY = (
    INTRO
    + SHEET
    + f'Write one lesson with two parts. collect, for a later patient with the same main complaint: what you should have asked and which kind of examination you should have requested first, so that fewer requests return nothing, aiming for about {COLLECT_WORDS} words. diagnose, for a later patient for whom the confirmed diagnosis of this patient is one of the candidate diagnoses: which findings point to it and what separates it from the diagnosis you submitted or from its closest alternative, aiming for about {DIAGNOSE_WORDS} words. Longer lessons are allowed when useful. Both are different patients, whose case files hold other records: name kinds of questions, examinations and findings, not the particulars of this patient. "Not provided" means the record is absent from this case file: it is not a normal result and not a refusal by the patient. Do not copy patient-specific ages, durations, measurements or identifiers, and do not write answer letters or an option list. Diagnostic concepts are allowed. Prefer {{"collect":"...","diagnose":"..."}}. A part that this encounter gives no lesson for is an empty string.'
)
PROMPTS = {
    "self": INTRO
    + UNVERIFIED
    + "Write one lesson for future patients with the same main complaint. "
    + COLLECT["self"]
    + f"aiming for about {COLLECT_WORDS} words. Longer lessons are allowed when useful. "
    + RULES
    + 'Prefer {"collect":"..."}. If this encounter gives no lesson return {"collect":""}.',
    "key": KEY,
}
