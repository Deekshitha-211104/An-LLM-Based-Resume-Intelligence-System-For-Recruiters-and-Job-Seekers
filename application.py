import gradio as gr
import pandas as pd
import os
import base64

from job_seeker import analyze_resume
from modules.text_extraction import extract_text_from_file
from modules.resume_parser import parse_resume_llm
from modules.embedding_engine import generate_embedding
from modules.jd_processor import embed_jd
from modules.ranking_engine import add_resume_embedding, search_candidates, index, resume_meta


# ─────────────────────────────────────────────────────────────
# JD Keyword list
# ─────────────────────────────────────────────────────────────

JD_KEYWORDS = [
    "python", "java", "c", "c++", "c#", "go", "rust", "kotlin", "swift",
    "javascript", "typescript", "dart", "php", "ruby", "bash", "shell scripting",
    "data structures", "algorithms", "object oriented programming",
    "system design", "design patterns", "software architecture",
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
    "rest api", "graphql", "websockets", "microservices",
    "android", "ios", "swiftui", "flutter", "react native",
    "spring", "spring boot", "django", "flask", "fastapi", ".net", ".net core",
    "api development", "authentication", "authorization", "jwt", "oauth",
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite",
    "redis", "cassandra", "dynamodb", "firebase",
    "pandas", "numpy", "excel", "power bi", "tableau",
    "statistics", "data analysis", "data visualization",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "keras",
    "feature engineering", "model evaluation", "bert",
    "spacy", "transformers", "f1-score", "precision", "recall", "accuracy",
    "named entity recognition", "sentiment analysis", "semantic segmentation",
    "llm", "generative ai", "prompt engineering",
    "speech recognition", "image processing",
    "etl", "data pipelines", "data warehousing", "data modeling",
    "spark", "hadoop", "airflow", "kafka",
    "aws", "azure", "gcp", "ec2", "s3", "lambda", "cloud functions", "api gateway",
    "docker", "kubernetes", "jenkins", "ci/cd",
    "terraform", "ansible", "helm", "prometheus", "grafana", "monitoring", "logging",
    "tcp/ip", "dns", "http", "https", "load balancing",
    "network security", "application security", "cryptography",
    "penetration testing", "ethical hacking", "vulnerability assessment",
    "blockchain", "ethereum", "solidity", "smart contracts", "web3",
    "embedded c", "rtos", "arduino", "raspberry pi", "iot", "mqtt", "sensor integration",
    "unity", "unreal engine", "ar development", "vr development", "3d graphics",
    "unit testing", "integration testing", "test automation", "selenium", "pytest", "junit", "cypress",
    "manual testing", "automation testing", "performance testing", "jmeter",
    "ui design", "ux design", "figma", "adobe xd", "wireframing", "prototyping", "user research",
    "product management", "roadmap planning", "agile", "scrum",
    "user stories", "stakeholder management",
    "distributed systems", "scalable systems", "enterprise architecture", "solution architecture",
    "api integration", "middleware", "system integration",
    "linux", "unix", "windows", "git", "github", "gitlab", "bitbucket",
]


# ─────────────────────────────────────────────────────────────
# RECRUITER PIPELINE
# ─────────────────────────────────────────────────────────────

def recruiter_pipeline(resumes, jd):

    # ── Validate inputs ──────────────────────────────────────
    if not resumes:
        return pd.DataFrame([{"Error": "Please upload at least one resume."}])
    if not jd or not jd.strip():
        return pd.DataFrame([{"Error": "Please enter a Job Description."}])

    # ── Reset index (also clears resume_meta list in-place) ──
    index.reset()

    # ── Parse and embed every resume ─────────────────────────
    failed = 0
    for file in resumes:
        try:
            text = extract_text_from_file(file)

            if not text or not text.strip():
                print(f"[pipeline] Empty text for: {getattr(file, 'name', str(file))}")
                failed += 1
                continue

            parsed    = parse_resume_llm(text)
            embedding = generate_embedding(parsed)

            meta = {
                "name":   parsed.get("name",   "Unknown"),
                "skills": parsed.get("skills", []),
                "email":  parsed.get("email",  "N/A"),
                "phone":  parsed.get("phone",  "N/A"),
            }

            add_resume_embedding(embedding, meta)
            print(f"[pipeline] Added: {meta['name']} | skills={len(meta['skills'])}")

        except Exception as e:
            print(f"[pipeline] Error processing resume: {e}")
            failed += 1
            continue

    # ── Check at least one resume was embedded ───────────────
    total_embedded = len(resume_meta)
    print(f"[pipeline] Embedded {total_embedded} resumes ({failed} failed)")

    if total_embedded == 0:
        return pd.DataFrame([{
            "Error": f"Could not extract text from any uploaded resume. "
                     f"({failed} file(s) failed — check they are PDF/DOCX)"
        }])

    # ── Embed JD and search ───────────────────────────────────
    jd_embedding = embed_jd(jd)
    results      = search_candidates(jd_embedding, top_k=total_embedded)

    if not results:
        return pd.DataFrame([{"Error": "Search returned no results."}])

    # ── Extract JD required skills ────────────────────────────
    jd_lower       = jd.lower()
    required_skills = [kw for kw in JD_KEYWORDS if kw in jd_lower]
    print(f"[pipeline] JD required skills: {len(required_skills)} → {required_skills[:10]}")

    # ── Score every candidate ─────────────────────────────────
    ranked = []
    for r in results:
        name   = r.get("name",   "Unknown")
        skills = r.get("skills", [])
        email  = r.get("email",  "N/A")
        phone  = r.get("phone",  "N/A")
        score  = r.get("score",  0.0)        # semantic similarity (0-1)

        skills_lower = [s.lower().strip() for s in skills if s]

        # Fuzzy skill match
        matched = set()
        for rs in skills_lower:
            for req in required_skills:
                if req in rs or rs in req:
                    matched.add(req)

        n_matched  = len(matched)
        n_required = max(len(required_skills), 1)

        skill_coverage = n_matched / n_required
        semantic_sim   = max(0.0, min(1.0, float(score)))
        ats_score      = int((skill_coverage * 0.6 + semantic_sim * 0.4) * 100)

        print(f"[pipeline] {name}: skill_cov={skill_coverage:.2f}, "
              f"sem={semantic_sim:.3f}, ats={ats_score}")

        ranked.append({
            "Name":      name,
            "Skills":    ", ".join(skills_lower[:12]) if skills_lower else "—",
            "Email":     email,
            "Phone":     phone,
            "ATS Score": ats_score,
            "Reason":    f"{n_matched} of {n_required} required skills matched",
        })
        
    # ── Filter candidates with ATS >= 60 ──────────────────────
    ranked = [c for c in ranked if c["ATS Score"] >= 60]

    # If no candidate passes the threshold
    if len(ranked) == 0:
        return pd.DataFrame([{
            "Rank": "-",
            "Name": "No Qualified Candidates",
            "Skills": "-",
            "Email": "-",
            "Phone": "-",
            "ATS Score": "-",
            "Reason": "No candidate scored above ATS threshold (60)"
    }])

    # ── Sort and assign ranks ─────────────────────────────────
    ranked = sorted(ranked, key=lambda x: x["ATS Score"], reverse=True)
    for i, c in enumerate(ranked, start=1):
        c["Rank"] = i

    df = pd.DataFrame(ranked)[["Rank", "Name", "Skills", "Email", "Phone", "ATS Score", "Reason"]]
    
    return df


# ─────────────────────────────────────────────────────────────
# CAROUSEL IMAGES → BASE64
# ─────────────────────────────────────────────────────────────

image_paths = [
    "/content/drive/MyDrive/ResumeLLMSystem/Carousel/Image-1.png",
    "/content/drive/MyDrive/ResumeLLMSystem/Carousel/Image-2.png",
    "/content/drive/MyDrive/ResumeLLMSystem/Carousel/Image-3.png",
    "/content/drive/MyDrive/ResumeLLMSystem/Carousel/Image-4.png",
    "/content/drive/MyDrive/ResumeLLMSystem/Carousel/Image-5.png",
]
image_paths = [p for p in image_paths if os.path.exists(p)]


def to_base64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


images = [to_base64(p) for p in image_paths]

fallback_gradients = [
    "linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%)",
    "linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%)",
    "linear-gradient(135deg,#0f172a 0%,#164e63 100%)",
    "linear-gradient(135deg,#0f172a 0%,#312e81 100%)",
    "linear-gradient(135deg,#0f172a 0%,#0c4a6e 100%)",
]

has_images   = len(images) > 0
total_slides = len(images) if has_images else 5
js_imgs      = "[" + ",".join(['"' + img + '"' for img in images]) + "]" if has_images else "[]"
js_grads     = "[" + ",".join(['"' + g  + '"' for g  in fallback_gradients]) + "]"


# ─────────────────────────────────────────────────────────────
# HERO CAROUSEL
# ─────────────────────────────────────────────────────────────

def build_hero(js_imgs_str, js_grads_str, n_slides):
    inner = (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><style>"
        "*{margin:0;padding:0;box-sizing:border-box;}"
        "body{background:#020617;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}"
        ".hero{position:relative;width:100%;height:440px;overflow:hidden;}"
        ".track{display:flex;height:100%;transition:transform 0.85s cubic-bezier(0.77,0,0.175,1);}"
        ".slide{min-width:100%;height:100%;background-size:cover;background-position:center;flex-shrink:0;}"
        ".ov1{position:absolute;inset:0;background:linear-gradient(105deg,rgba(2,6,23,0.93) 0%,rgba(2,6,23,0.72) 45%,rgba(2,6,23,0.15) 100%);pointer-events:none;}"
        ".ov2{position:absolute;inset:0;background-image:linear-gradient(rgba(99,102,241,0.07) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,0.07) 1px,transparent 1px);background-size:50px 50px;pointer-events:none;}"
        ".content{position:absolute;top:50%;left:60px;transform:translateY(-50%);z-index:10;max-width:620px;}"
        ".badge{display:inline-flex;align-items:center;gap:8px;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);border-radius:999px;padding:5px 14px;margin-bottom:18px;font-size:11px;font-weight:600;color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;}"
        ".pulse{width:6px;height:6px;border-radius:50%;background:#38bdf8;animation:dp 2s ease-in-out infinite;}"
        "@keyframes dp{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.3;transform:scale(0.6);}}"
        ".title{font-size:48px;font-weight:900;line-height:1.1;margin-bottom:14px;background:linear-gradient(90deg,#f0f9ff 20%,#38bdf8 55%,#818cf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-1.5px;}"
        ".caption{font-size:15px;color:#94a3b8;line-height:1.7;margin-bottom:26px;}"
        ".caption b{color:#e2e8f0;font-weight:600;}"
        ".stats{display:flex;gap:28px;}"
        ".sv{font-size:22px;font-weight:800;color:#38bdf8;letter-spacing:-0.5px;}"
        ".sl{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px;}"
        ".arrow{position:absolute;top:50%;transform:translateY(-50%);z-index:20;width:42px;height:42px;border-radius:50%;border:1px solid rgba(255,255,255,0.15);background:rgba(2,6,23,0.65);display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:22px;color:white;user-select:none;transition:background 0.2s,border-color 0.2s;}"
        ".arrow:hover{background:rgba(56,189,248,0.2);border-color:rgba(56,189,248,0.5);}"
        "#ap{left:16px;}#an{right:16px;}"
        ".dots{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);display:flex;gap:8px;z-index:20;}"
        ".dot{height:7px;border-radius:999px;background:rgba(255,255,255,0.22);cursor:pointer;transition:all 0.35s ease;width:7px;}"
        ".dot.on{background:#38bdf8;width:26px;}"
        ".pbar{position:absolute;bottom:0;left:0;height:3px;background:linear-gradient(90deg,#38bdf8,#6366f1);z-index:20;width:0;}"
        ".ctr{position:absolute;top:18px;right:18px;z-index:20;font-size:12px;color:rgba(255,255,255,0.35);letter-spacing:0.08em;}"
        ".ctr b{color:#38bdf8;}"
        "</style></head><body>"
        "<div class='hero'><div class='track' id='tr'></div>"
        "<div class='ov1'></div><div class='ov2'></div>"
        "<div class='content'>"
        "<div class='badge'><div class='pulse'></div>AI-Powered Platform</div>"
        "<div class='title'>Resume Intelligence<br>System</div>"
        "<div class='caption'><b>Unlock the Power of AI</b> &mdash; Smart Resume Analysis for Faster Hiring and Job Matching. Instantly rank candidates, decode resumes, and match talent with precision.</div>"
        "<div class='stats'>"
        "<div><div class='sv'>10&times;</div><div class='sl'>Faster Screening</div></div>"
        "<div><div class='sv'>User Friendly</div><div class='sl'>Easy to Use</div></div>"
        "<div><div class='sv'>LLM</div><div class='sl'>AI Powered</div></div>"
        "</div></div>"
        "<div class='arrow' id='ap'>&#8249;</div>"
        "<div class='arrow' id='an'>&#8250;</div>"
        "<div class='dots' id='dt'></div>"
        "<div class='pbar' id='pb'></div>"
        "<div class='ctr'><b id='cc'>1</b> / <span id='ct'>" + str(n_slides) + "</span></div>"
        "</div>"
        "<script>"
        "var imgs=" + js_imgs_str + ";"
        "var grads=" + js_grads_str + ";"
        "var total=" + str(n_slides) + ";"
        "var cur=0,timer=null;"
        "var tr=document.getElementById('tr'),dt=document.getElementById('dt'),"
        "    pb=document.getElementById('pb'),cc=document.getElementById('cc');"
        "for(var i=0;i<total;i++){"
        "  var s=document.createElement('div');s.className='slide';"
        "  if(imgs.length>0){s.style.backgroundImage='url('+imgs[i]+')';}else{s.style.background=grads[i%grads.length];}"
        "  tr.appendChild(s);}"
        "for(var i=0;i<total;i++){(function(idx){"
        "  var d=document.createElement('div');d.className='dot'+(idx===0?' on':'');"
        "  d.addEventListener('click',function(){stopA();goTo(idx);startA();});dt.appendChild(d);"
        "})(i);}"
        "function upd(){tr.style.transform='translateX(-'+cur*100+'%)';cc.textContent=cur+1;"
        "  dt.querySelectorAll('.dot').forEach(function(d,i){d.classList.toggle('on',i===cur);});}"
        "function animP(){pb.style.transition='none';pb.style.width='0%';"
        "  setTimeout(function(){pb.style.transition='width 3.2s linear';pb.style.width='100%';},30);}"
        "function goTo(n){cur=(n+total)%total;upd();animP();}"
        "function next(){goTo(cur+1);}function prev(){goTo(cur-1);}"
        "function startA(){timer=setInterval(next,3500);}function stopA(){clearInterval(timer);}"
        "document.getElementById('an').addEventListener('click',function(){stopA();next();startA();});"
        "document.getElementById('ap').addEventListener('click',function(){stopA();prev();startA();});"
        "animP();startA();"
        "</script></body></html>"
    )
    escaped = inner.replace('"', "&quot;")
    return '<iframe srcdoc="' + escaped + '" style="width:100%;height:440px;border:none;display:block;" scrolling="no"></iframe>'


HERO_HTML = build_hero(js_imgs, js_grads, total_slides)


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

css = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: #ffffff !important;
}

/* Hide footer */
.gradio-container footer, footer.svelte-1ax1toq,
.footer, div.svelte-1b1p3p2, .built-with {
    display: none !important;
}

/* Action buttons — scoped via elem_classes="mode-btn" */
.mode-btn {
    height: 60px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 30px !important;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.mode-btn:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.55) !important;
}

/* Panel background */
.gradio-container .gr-column, .gradio-container .gr-box {
    background: #ffffff !important;
}
.panel-section { padding: 28px 44px !important; }

/* Labels */
.gradio-container label span, .gradio-container .gr-form label {
    color:# 374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Inputs */
.gradio-container textarea, .gradio-container input[type=text] {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus {
    border-color: #38bdf8 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* File upload */
.gradio-container .upload-container, .gradio-container .gr-file,
.gradio-container [data-testid="file-upload"], .gradio-container .file-preview-holder {
    background: #0f172a !important;
    border: 1px dashed #334155 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}
.gradio-container .upload-container svg, .gradio-container [data-testid="file-upload"] svg {
    color: #38bdf8 !important;
    fill: #38bdf8 !important;
}

/* Markdown headings */
.gradio-container .gr-markdown h2 {
    color: #f0f9ff !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(56,189,248,0.18) !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
}

/* Dataframe */
.gradio-container .gr-dataframe,
.gradio-container [data-testid="dataframe"],
.gradio-container .wrap.svelte-1occ011 {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    color: #e2e8f0 !important;
}
.gradio-container table thead tr th,
.gradio-container .gr-dataframe thead th,
.gradio-container [data-testid="dataframe"] thead th {
    background: #1e293b !important;
    color: #38bdf8 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    padding: 12px 14px !important;
    border-bottom: 1px solid #334155 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    height: auto !important;
    transform: none !important;
}
.gradio-container table tbody tr td,
.gradio-container .gr-dataframe tbody td,
.gradio-container [data-testid="dataframe"] tbody td {
    background: #0f172a !important;
    color: #cbd5e1 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #1e293b !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    height: auto !important;
}
.gradio-container table tbody tr:nth-child(even) td { background: #111827 !important; }
.gradio-container table tbody tr:hover td            { background: #1e293b !important; }
.gradio-container table th button, .gradio-container table td button,
.gradio-container table th span,   .gradio-container table td span {
    background: none !important; border: none !important;
    border-radius: 0 !important; box-shadow: none !important;
    color: inherit !important;   height: auto !important;
    padding: 0 !important;       font-size: inherit !important;
    font-weight: inherit !important;
}
"""


# ─────────────────────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────────────────────

with gr.Blocks(css=css) as demo:

    gr.HTML(HERO_HTML)

    with gr.Row():
        recruiter_btn = gr.Button("🚀  Recruiter Mode", elem_classes=["mode-btn"])
        jobseeker_btn = gr.Button("👤  Job Seeker Mode", elem_classes=["mode-btn"])

    # ── Job Seeker ────────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["panel-section"]) as jobseeker_panel:
        gr.Markdown("## 👤 Resume Analyzer")
        resume_file = gr.File(label="Upload Your Resume")
        jd_input    = gr.Textbox(lines=10, label="Job Description",
                                  placeholder="Paste the job description here...")
        analyze_btn = gr.Button("✨  Analyze Resume", elem_classes=["mode-btn"])
        ats_plot    = gr.Plot()
        matched_box = gr.Textbox(lines=5, label="✅ Matched Skills")
        missing_box = gr.Textbox(lines=5, label="⚠️ Missing Skills")
        summary_box = gr.Textbox(lines=7, label="📋 AI Summary")
        analyze_btn.click(
            analyze_resume,
            inputs=[resume_file, jd_input],
            outputs=[ats_plot, matched_box, missing_box, summary_box],
        )

    # ── Recruiter ─────────────────────────────────────────────
    with gr.Column(visible=False, elem_classes=["panel-section"]) as recruiter_panel:
        gr.Markdown("## 🚀 Candidate Ranking")
        resume_files    = gr.File(file_count="multiple", label="Upload Resumes")
        jd_input_rec    = gr.Textbox(lines=10, label="Job Description",
                                      placeholder="Paste the job description here...")
        analyze_btn_rec = gr.Button("🔍  Rank Candidates", elem_classes=["mode-btn"])
        output_table    = gr.Dataframe(
            headers=["Rank", "Name", "Skills", "Email", "Phone", "ATS Score", "Reason"],
            datatype=["number", "str", "str", "str", "str", "number", "str"],
            wrap=True,
            interactive=False,
            label="Shortlisted Candidates",
        )
        analyze_btn_rec.click(
            recruiter_pipeline,
            inputs=[resume_files, jd_input_rec],
            outputs=output_table,
        )

    # ── Toggle panels ─────────────────────────────────────────
    recruiter_btn.click(
        lambda: (gr.update(visible=True),  gr.update(visible=False)),
        outputs=[recruiter_panel, jobseeker_panel],
    )
    jobseeker_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[recruiter_panel, jobseeker_panel],
    )

demo.launch(share=True)