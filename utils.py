import random

GEC = [
    "Fix grammar",
    "Fix grammar in this sentence",
    "Fix grammar in the sentence",
    "Fix grammar errors",
    "Fix grammatical errors",
    "Fix grammaticality",
    "Fix all grammatical errors",
    "Fix grammatical errors in this sentence",
    "Fix grammar errors in this sentence",
    "Fix grammatical mistakes in this sentence",
    "Fix grammaticality in this sentence",
    "Fix grammaticality of the sentence",
    "Fix disfluencies in the sentence",
    "Make the sentence grammatical",
    "Make the sentence fluent",
    "Fix errors in this text",
    "Update to remove grammar errors",
    "Remove all grammatical errors from this text",
    "Improve the grammar of this text",
    "Improve the grammaticality",
    "Improve the grammaticality of this text",
    "Improve the grammaticality of this sentence",
    "Grammar improvements",
    "Remove grammar mistakes",
    "Remove grammatical mistakes",
    "Fix the grammar mistakes",
    "Fix grammatical mistakes Clarity Clarify the sentence",
]
Clarify = [
    "Clarify this sentence",
    "Clarify this text",
    "Write a clearer version for the sentence",
    "Write a clarified version of the sentence",
    "Write a readable version of the sentence",
    "Write a better readable version of the sentence",
    "Rewrite the sentence more clearly",
    "Rewrite this sentence clearly",
    "Rewrite this sentence for clarity",
    "Rewrite this sentence for readability",
    "Improve this sentence for readability",
    "Make this sentence better readable",
    "Make this sentence more readable",
    "Make this sentence readable",
    "Make the sentence clear",
    "Make the sentence clearer",
    "Clarify",
    "Make the text more understandable",
    "Make this easier to read",
    "Clarification",
    "Change to clearer wording",
    "Clarify this paragraph",
    "Use clearer wording Simplification Simplify the sentence",
    "Simplify this sentence",
    "Simplify this text",
    "Write a simpler version for the sentence",
    "Rewrite the sentence to be simpler",
    "Rewrite this sentence in a simpler manner",
    "Rewrite this sentence for simplicity",
    "Rewrite this with simpler wording",
    "Make the sentence simple",
    "Make the sentence simpler",
    "Make this text less complex",
    "Make this simpler",
    "Simplify",
    "Simplification",
    "Change to simpler wording",
    "Simplify this paragraph",
    "Simplify this text",
    "Use simpler wording",
    "Make this easier to understand",
]
Coherence = [
    "Fix coherence",
    "Fix coherence in this sentence",
    "Fix coherence in the sentence",
    "Fix coherence in this text",
    "Fix coherence in the text",
    "Fix coherence errors",
    "Fix sentence flow",
    "Fix sentence transition",
    "Fix coherence errors in this sentence",
    "Fix coherence mistakes in this sentence",
    "Fix coherence in this sentence",
    "Fix coherence of the sentence",
    "Fix lack of coherence in the sentence",
    "Make the text more coherent",
    "Make the text coherent",
    "Make the text more cohesive",
    "logically linked and consistent as a whole",
    "Make the text more cohesive",
    "Improve the cohesiveness of the text",
    "Make the text more logical",
    "Make the text more consistent",
    "Improve the consistency of the text",
    "Make the text clearer",
    "Improve the coherence of the text",
]
Formality_Style_Transfer = [
    "Formalize",
    "Improve formality",
    "Formalize the sentence",
    "Formalize this sentence",
    "Formalize the text",
    "Formalize this text",
    "Make this formal",
    "Make this more formal",
    "Make this sound more formal",
    "Make the sentence formal",
    "Make the sentence more formal",
    "Make the sentence sound more formal",
    "Write more formally",
    "Write less informally",
    "Rewrite more formally",
    "Write this more formally",
    "Rewrite this more formally",
    "Write in a formal manner",
    "Write in a more formal manner",
    "Rewrite in a more formal manner",
]
Neutralization = [
    "Remove POV",
    "Remove POVs",
    "Remove POV in this text",
    "Remove POVs in this text",
    "Neutralize this text",
    "Neutralize the text",
    "Neutralize this sentence",
    "Neutralize the sentence",
    "Make this more neutral",
    "Make this text more neutral",
    "Make this sentence more neutral",
    "Make this paragraph more neutral",
    "Remove unsourced opinions",
    "Remove unsourced opinions from this text",
    "Remove non-neutral POVs",
    "Remove non-neutral POV",
    "Remove non-neutral points of view",
    "Remove points of view",
    "Make this text less biased Paraphrasing Paraphrase the sentence",
    "Paraphrase this sentence",
    "Paraphrase this text",
]
Paraphrase = [
    "Write a paraphrase for the sentence",
    "Write a paraphrased version of the sentence",
    "Rewrite the sentence with different wording",
    "Use different wording",
    "Rewrite this sentence",
    "Reword this sentence",
    "Rephrase this sentence",
    "Rewrite this text",
    "Reword this text",
    "Rephrase this text",
]

instruction_prompts = {
    "Grammar Error Correction": GEC,
    "Clarify": Clarify,
    "Coherence": Coherence,
    "Formality Style Transfer": Formality_Style_Transfer,
    "Neutralization": Neutralization,
    "Paraphrase": Paraphrase,
}


def get_prompt_list(instruction_type: str) -> list:
    """
    Returns a list of prompts for the given instruction type.

    Args:
        instruction_type: The type of instruction, e.g., "Grammar Error Correction".

    Returns:
        A list of prompts corresponding to the instruction type.
    """
    return instruction_prompts[instruction_type]


def get_random_prompt(instruction_type: str) -> str:
    """
    Returns a random prompt from the list of prompts for the given instruction type.

    Args:
        instruction_type: The type of instruction, e.g., "Grammar Error Correction".

    Returns:
        A random prompt from the list of prompts for the instruction type.
    """
    return random.choice(instruction_prompts[instruction_type])
