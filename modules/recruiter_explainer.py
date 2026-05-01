def explain_ranking(llm, candidate_name, matched_skills, ats_score):

    prompt = f"""
You are an AI recruitment assistant.

Candidate: {candidate_name}

Matched Skills: {matched_skills}

ATS Score: {ats_score}

Explain in one sentence why this candidate is ranked at this position.
Keep it professional.
"""

    output = llm(prompt, max_length=80)

    return output[0]["generated_text"]