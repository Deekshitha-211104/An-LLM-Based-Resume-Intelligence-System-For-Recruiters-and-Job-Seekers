import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

print("Loading AI Career Advisor Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

print("AI Career Advisor Model Loaded")


# ---------------------------------
# Generate Career Suggestions
# ---------------------------------

def generate_career_suggestions(role, matched, missing, score):

    suggestions = []

    if missing:
        suggestions.append(
            f"Add projects demonstrating {missing[0]} related to the {role} role."
        )

    if len(missing) > 1:
        suggestions.append(
            f"Gain hands-on experience with {missing[1]} and include it in your resume."
        )

    if len(missing) > 2:
        suggestions.append(
            f"Build a practical project using {missing[2]} and showcase it on GitHub."
        )

    if score >= 75:
        suggestions.append(
            "Your resume already matches most requirements. Add measurable achievements and deployment experience."
        )

    if len(suggestions) == 0:
        suggestions.append(
            "Strengthen your resume by adding real-world projects and quantifiable results."
        )

    formatted = "\n".join([f"{i+1}. {s}" for i, s in enumerate(suggestions[:3])])

    return formatted

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response