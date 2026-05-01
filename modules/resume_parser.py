import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
print("Model Loaded Successfully")


# ─────────────────────────────────────────────────────────────
# Skill Database
# ─────────────────────────────────────────────────────────────

skills_db = [
    # Programming
    "python", "java", "c", "c++", "c#", "go", "rust", "kotlin", "swift",
    "javascript", "typescript", "dart", "php", "ruby", "bash", "shell scripting",
    # Core CS
    "data structures", "algorithms", "object oriented programming",
    "system design", "design patterns", "software architecture",
    # Web
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
    "rest api", "graphql", "websockets", "microservices",
    # Mobile
    "android", "ios", "swiftui", "flutter", "react native",
    # Backend
    "spring", "spring boot", "django", "flask", "fastapi", ".net", ".net core",
    "api development", "authentication", "authorization", "jwt", "oauth",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite",
    "redis", "cassandra", "dynamodb", "firebase",
    # Data & Analytics
    "pandas", "numpy", "excel", "power bi", "tableau",
    "statistics", "data analysis", "data visualization",
    # AI / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "keras",
    "feature engineering", "model evaluation", "bert",
    "spacy", "transformers", "f1-score", "precision", "recall", "accuracy",
    "named entity recognition", "sentiment analysis", "semantic segmentation",
    "llm", "generative ai", "prompt engineering",
    "speech recognition", "image processing",
    # Data Engineering
    "etl", "data pipelines", "data warehousing", "data modeling",
    "spark", "hadoop", "airflow", "kafka",
    # Cloud
    "aws", "azure", "gcp", "ec2", "s3", "lambda", "cloud functions", "api gateway",
    # DevOps
    "docker", "kubernetes", "jenkins", "ci/cd",
    "terraform", "ansible", "helm", "prometheus", "grafana", "monitoring", "logging",
    # Networking
    "tcp/ip", "dns", "http", "https", "load balancing",
    # Security
    "network security", "application security", "cryptography",
    "penetration testing", "ethical hacking", "vulnerability assessment",
    # Blockchain
    "blockchain", "ethereum", "solidity", "smart contracts", "web3",
    # Embedded / IoT
    "embedded c", "rtos", "arduino", "raspberry pi", "iot", "mqtt", "sensor integration",
    # AR/VR
    "unity", "unreal engine", "ar development", "vr development", "3d graphics",
    # Testing
    "unit testing", "integration testing", "test automation",
    "selenium", "pytest", "junit", "cypress",
    "manual testing", "automation testing", "performance testing", "jmeter",
    # UI/UX
    "ui design", "ux design", "figma", "adobe xd",
    "wireframing", "prototyping", "user research",
    # Product / Architecture
    "product management", "roadmap planning", "agile", "scrum",
    "user stories", "stakeholder management",
    "distributed systems", "scalable systems",
    "enterprise architecture", "solution architecture",
    "api integration", "middleware", "system integration",
    # OS / VCS
    "linux", "unix", "windows",
    "git", "github", "gitlab", "bitbucket",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def extract_skills_from_text(text):
    """Always-reliable regex-based skill extraction from raw resume text."""
    lower = text.lower()
    found = []
    for skill in skills_db:
        # Use word-boundary matching to avoid false positives like "c" inside "react"
        pattern = r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])'
        if re.search(pattern, lower):
            found.append(skill)
    return found


def extract_name_from_text(text):
    """Heuristic: first non-empty line in the top 5 lines that looks like a name."""
    lines = text.split("\n")
    for line in lines[:8]:
        line = line.strip()
        if not line:
            continue
        # Skip lines with email / phone / URLs / long lines
        if "@" in line or any(c.isdigit() for c in line):
            continue
        if len(line) > 40:
            continue
        words = line.split()
        if 1 <= len(words) <= 5:
            return line
    return "Unknown"


def extract_email(text):
    emails = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    return emails[0] if emails else ""


def extract_phone(text):
    phones = re.findall(r"\+?[\d][\d\s\-\(\)]{7,15}", text)
    return phones[0].strip() if phones else ""


def extract_json_block(text):
    """Extract the first complete JSON object from model output."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def fallback_extraction(resume_text):
    """Pure regex/heuristic extraction — never fails."""
    return {
        "name":  extract_name_from_text(resume_text),
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "skills": extract_skills_from_text(resume_text),
        "education": "",
        "experience": "",
    }


# ─────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────

def parse_resume_llm(resume_text):
    """
    Try LLM extraction first; fall back to regex if LLM fails or returns
    incomplete data.  Skill extraction from raw text is ALWAYS run as a
    safety net so skills are never empty.
    """

    if not resume_text or not resume_text.strip():
        return fallback_extraction("")

    trimmed = resume_text[:3000]

    prompt = (
        "Extract structured information from the resume below.\n"
        "Return ONLY a valid JSON object — no explanation, no markdown.\n\n"
        "Format:\n"
        '{"name": "", "email": "", "phone": "", "skills": []}\n\n'
        f"Resume:\n{trimmed}"
    )

    parsed = {}

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_str = extract_json_block(decoded)

        if json_str:
            parsed = json.loads(json_str)

    except Exception as e:
        print(f"[resume_parser] LLM error: {e}")

    finally:
        torch.cuda.empty_cache()

    # ── Always enrich with regex-based extraction ──────────────────────────────

    # Name
    if not parsed.get("name") or parsed["name"].strip() in ("", "Unknown"):
        parsed["name"] = extract_name_from_text(resume_text)

    # Email
    if not parsed.get("email"):
        parsed["email"] = extract_email(resume_text)

    # Phone
    if not parsed.get("phone"):
        parsed["phone"] = extract_phone(resume_text)

    # Skills — ALWAYS supplement with regex extraction so list is never empty
    llm_skills = [s.lower().strip() for s in parsed.get("skills", []) if s]
    regex_skills = extract_skills_from_text(resume_text)

    # Merge: keep LLM skills + add any regex skills not already present
    merged_skills = list(llm_skills)
    for sk in regex_skills:
        if sk not in merged_skills:
            merged_skills.append(sk)

    parsed["skills"] = merged_skills if merged_skills else regex_skills

    print(f"[resume_parser] Parsed: name={parsed.get('name')}, "
          f"skills_count={len(parsed.get('skills', []))}, "
          f"email={parsed.get('email')}")

    return parsed