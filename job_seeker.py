import matplotlib.pyplot as plt
import re
import plotly.graph_objects as go
from modules.ai_career_advisor import generate_career_suggestions
from modules.text_extraction import extract_text_from_file


# =====================================================
# ROLE DATABASE
# =====================================================

roles_db = [
"Software Engineer","Software Developer","Full Stack Developer",
"Backend Developer","Frontend Developer","Web Developer",
"Mobile App Developer","Android Developer","iOS Developer",
"AI Engineer","Machine Learning Engineer","Deep Learning Engineer",
"Data Scientist","Data Analyst","Business Analyst","Junior AI Engineer","Junior AI Engineer","Junior ML Engineer","AI Researcher","Data Engineer","Data Analytics Engineer","Generative AI Engineer","LLM Engineer","MLOps Engineer",
"NLP Engineer","Computer Vision Engineer",
"Cloud Engineer","Cloud Architect","DevOps Engineer",
"Site Reliability Engineer","Cyber Security Engineer",
"Blockchain Developer","IoT Engineer","Embedded Systems Engineer",
"Database Engineer","Data Engineer","Big Data Engineer",
"QA Engineer","Automation Engineer","Test Engineer",
"Product Manager","Technical Product Manager",
"UI Designer","UX Designer","UI/UX Engineer",
"AR Developer","VR Developer",
"Network Engineer","System Engineer",
"Solutions Architect","Enterprise Architect",
"Platform Engineer","Application Engineer"
]


# =====================================================
# SKILL DATABASE
# =====================================================

skills_db = [

# ================= Programming =================
"python","java","c","c++","c#","go","rust","kotlin","swift",
"javascript","typescript","dart","php","ruby","bash","shell scripting",

# ================= Core CS =================
"data structures","algorithms","object oriented programming",
"system design","design patterns","software architecture",

# ================= Web Development =================
"html","css","javascript","typescript",
"react","angular","vue","next.js","node.js","express",
"rest api","graphql","websockets","microservices",

# ================= Mobile Development =================
"android","kotlin android","java android",
"ios","swiftui","flutter","react native",

# ================= Backend =================
"spring","spring boot","django","flask","fastapi",".net",".net core",
"api development","authentication","authorization","jwt","oauth",

# ================= Databases =================
"sql","mysql","postgresql","mongodb","oracle","sqlite",
"redis","cassandra","dynamodb","firebase",

# ================= Data & Analytics =================
"pandas","numpy","excel","power bi","tableau",
"statistics","data analysis","data visualization",

# ================= AI / ML =================
"machine learning","deep learning","nlp","computer vision",
"pytorch","tensorflow","scikit-learn","keras",
"feature engineering","model evaluation","bert",
"spacy","transformers","f1-score","precision","recall","accuracy",
"named entity recognition","sentiment analysis","semantic segmentation",

# ================= Specialized AI Roles =================
"transformers","llm","generative ai","prompt engineering",
"speech recognition","image processing",

# ================= Data Engineering =================
"etl","data pipelines","data warehousing","data modeling",
"spark","hadoop","airflow","kafka",

# ================= Cloud =================
"aws","azure","gcp",
"ec2","s3","lambda","cloud functions","api gateway",

# ================= DevOps / SRE =================
"docker","kubernetes","jenkins","ci/cd",
"terraform","ansible","helm",
"prometheus","grafana","monitoring","logging",

# ================= Networking =================
"tcp/ip","dns","http","https","load balancing",

# ================= Cyber Security =================
"network security","application security","cryptography",
"penetration testing","ethical hacking",
"vulnerability assessment","oauth","jwt",

# ================= Blockchain =================
"blockchain","ethereum","solidity","smart contracts","web3",

# ================= Embedded / IoT =================
"embedded c","rtos","arduino","raspberry pi",
"iot","mqtt","sensor integration",

# ================= AR / VR =================
"unity","unreal engine","ar development","vr development","3d graphics",

# ================= Testing =================
"unit testing","integration testing","test automation",
"selenium","pytest","junit","cypress",

# ================= QA / Automation =================
"manual testing","automation testing","performance testing","jmeter",

# ================= UI / UX =================
"ui design","ux design","figma","adobe xd",
"wireframing","prototyping","user research",

# ================= Product Roles =================
"product management","roadmap planning","agile","scrum",
"user stories","stakeholder management",

# ================= Architecture Roles =================
"distributed systems","scalable systems",
"enterprise architecture","solution architecture",

# ================= Platform / Application =================
"api integration","middleware","system integration",

# ================= Operating Systems =================
"linux","unix","windows",

# ================= Version Control =================
"git","github","gitlab","bitbucket"

]


# =====================================================
# ROLE DETECTION
# =====================================================

def extract_job_role(jd):

    jd = jd.lower()

    role_keywords = {

        "AI Engineer": ["ai engineer", "artificial intelligence", "ai/ml", "ai developer"],
        "Machine Learning Engineer": ["machine learning", "ml engineer", "ml model"],
        "Deep Learning Engineer": ["deep learning", "neural networks"],
        "Data Scientist": ["data scientist", "data science"],
        "Data Analyst": ["data analyst", "data analysis"],
        "Business Analyst": ["business analyst"],
        "NLP Engineer": ["nlp", "natural language processing"],
        "Computer Vision Engineer": ["computer vision", "image processing"],

        "Software Engineer": ["software engineer", "software developer"],
        "Full Stack Developer": ["full stack"],
        "Backend Developer": ["backend", "back-end"],
        "Frontend Developer": ["frontend", "front-end"],
        "Web Developer": ["web developer"],

        "Mobile App Developer": ["mobile developer", "app developer"],
        "Android Developer": ["android"],
        "iOS Developer": ["ios"],

        "Cloud Engineer": ["cloud engineer"],
        "Cloud Architect": ["cloud architect"],
        "DevOps Engineer": ["devops"],
        "Site Reliability Engineer": ["sre", "site reliability"],

        "Cyber Security Engineer": ["cyber security", "security engineer"],
        "Blockchain Developer": ["blockchain", "web3"],
        "IoT Engineer": ["iot"],
        "Embedded Systems Engineer": ["embedded", "firmware"],

        "Database Engineer": ["database engineer", "dba"],
        "Data Engineer": ["data engineer"],
        "Big Data Engineer": ["big data", "hadoop", "spark"],

        "QA Engineer": ["qa engineer", "quality assurance"],
        "Automation Engineer": ["automation engineer", "test automation"],
        "Test Engineer": ["testing", "test engineer"],

        "Product Manager": ["product manager"],
        "Technical Product Manager": ["technical product"],

        "UI Designer": ["ui designer"],
        "UX Designer": ["ux designer"],
        "UI/UX Engineer": ["ui ux", "ui/ux"],

        "AR Developer": ["augmented reality", "ar developer"],
        "VR Developer": ["virtual reality", "vr developer"],

        "Network Engineer": ["network engineer"],
        "System Engineer": ["system engineer"],

        "Solutions Architect": ["solutions architect"],
        "Enterprise Architect": ["enterprise architect"],

        "Platform Engineer": ["platform engineer"],
        "Application Engineer": ["application engineer"]
    }

    # Check keyword matches
    for role, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword in jd:
                return role

    # Fallback: direct role match
    for role in roles_db:
        if role.lower() in jd:
            return role

    return "Software Engineer"


# =====================================================
# SKILL EXTRACTION
# =====================================================

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills_db:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.append(skill)

    return list(set(found))


# =====================================================
# ANALYZE RESUME
# =====================================================

def analyze_resume(resume_file, jd):

    # Extract resume text
    resume_text = extract_text_from_file(resume_file)

    # Extract skills
    resume_skills = extract_skills(resume_text)
    jd_skills = list(set(extract_skills(jd) + extract_skills(jd.lower())))

    # Match & missing
    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    # Score calculation
    if len(jd_skills) == 0:
        score = 0
    else:
        score = int((len(matched) / len(jd_skills)) * 100)

    # Role detection
    role = extract_job_role(jd)

    # =====================================================
    # ATS SCORE GRAPH
    # =====================================================

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "ATS Compatibility Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ]
        }
    ))

    # =====================================================
    # AI SUMMARY
    # =====================================================

    ai_advice = generate_career_suggestions(
        role,
        matched,
        missing,
        score
    )

    summary = f"""
Detected Job Role: {role}

Resume Match Score: {score}%

AI Career Suggestions:

{ai_advice}
"""

    # =====================================================
    # RETURN OUTPUTS (IMPORTANT FOR GRADIO)
    # =====================================================

    return fig, "\n".join(sorted(matched)), "\n".join(sorted(missing)), summary



   