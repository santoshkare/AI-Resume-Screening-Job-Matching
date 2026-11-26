"""
SkillScout v2.0 — AI Resume Ranking & Talent Intelligence Platform
==================================================================
Enterprise-grade Streamlit UI with Plotly Visualizations, Blind Screening,
Stacked Ensemble Explainability, and AI Recruiter Copilot.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import io
import sys
from datetime import datetime
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Page Configuration ──
st.set_page_config(
    page_title="SkillScout AI — Intelligent Talent Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Preset Job Descriptions ──
PRESET_JDS = {
    "🤖 Senior AI & ML Engineer": (
        "Role: Senior AI & Machine Learning Engineer\n"
        "Location: Remote / Hybrid\n"
        "Experience Required: 3-6 years\n\n"
        "About the Role:\n"
        "We are looking for a Senior Machine Learning Engineer to architect and deploy production NLP models and LLM microservices.\n\n"
        "Key Responsibilities:\n"
        "- Design, train, and optimize deep learning and NLP architectures using PyTorch and Hugging Face Transformers.\n"
        "- Build production inference pipelines, REST APIs, and microservices using Python, FastAPI, and Docker.\n"
        "- Implement vector search, embeddings, and RAG pipelines for semantic document retrieval.\n"
        "- Collaborate with MLOps on CI/CD pipelines, model monitoring, and distributed cloud deployments (AWS/GCP).\n\n"
        "Requirements & Skills:\n"
        "- Strong proficiency in Python, NumPy, Pandas, and Scikit-Learn.\n"
        "- Proven hands-on experience with PyTorch, Transformers, Sentence-BERT, and vector embeddings.\n"
        "- Experience with Docker, Git, CI/CD, and deploying microservices.\n"
        "- Education: Bachelor's or Master's in Computer Science, Data Science, or related field."
    ),
    "💻 Full-Stack Cloud & Python Engineer": (
        "Role: Full-Stack Cloud & Python Engineer\n"
        "Location: Remote\n"
        "Experience Required: 2-5 years\n\n"
        "About the Role:\n"
        "We are seeking a versatile Full-Stack Engineer to develop scalable cloud applications and user-facing dashboards.\n\n"
        "Key Responsibilities:\n"
        "- Develop robust backend services and RESTful APIs using Python, FastAPI, and Django.\n"
        "- Build responsive, high-performance web interfaces using React, TypeScript, and Tailwind CSS.\n"
        "- Design database schemas and optimize queries with PostgreSQL and Redis caching.\n"
        "- Containerize microservices using Docker and set up automated CI/CD pipelines.\n\n"
        "Requirements & Skills:\n"
        "- Strong proficiency in Python, TypeScript/JavaScript, React, and HTML/CSS.\n"
        "- Experience with PostgreSQL, Redis, RESTful APIs, Git, and Docker.\n"
        "- Education: B.Tech/B.E./B.Sc in Computer Science or equivalent practical experience."
    ),
    "📊 Data Science & Analytics Specialist": (
        "Role: Data Science & Analytics Specialist\n"
        "Location: Hybrid\n"
        "Experience Required: 2-4 years\n\n"
        "About the Role:\n"
        "We are hiring a Data Scientist to transform complex business data into predictive insights and executive dashboards.\n\n"
        "Key Responsibilities:\n"
        "- Extract, clean, and analyze structured and unstructured datasets using Python and SQL.\n"
        "- Build predictive ML models, regression analyses, and clustering models using Scikit-Learn.\n"
        "- Conduct statistical hypothesis testing, A/B testing, and exploratory data analysis.\n"
        "- Build interactive business intelligence dashboards using Plotly, Tableau, and Streamlit.\n\n"
        "Requirements & Skills:\n"
        "- Advanced proficiency in Python, Pandas, NumPy, SQL, and Scikit-Learn.\n"
        "- Strong knowledge of statistical inference, probability, and exploratory data analysis.\n"
        "- Experience with data visualization (Plotly, Seaborn, Tableau) and stakeholder reporting."
    ),
}

SAMPLE_RESUMES_DIR = PROJECT_ROOT / "sample_resumes"

# ── Custom CSS for High-End Glassmorphism Dark UI ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0F19;
        --bg-surface: #111827;
        --bg-glass: rgba(17, 24, 39, 0.75);
        --bg-card-hover: rgba(30, 41, 59, 0.85);
        --accent-blue: #3B82F6;
        --accent-purple: #8B5CF6;
        --accent-cyan: #06B6D4;
        --accent-green: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(59, 130, 246, 0.3);
    }

    .stApp {
        background-color: var(--bg-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(15, 17, 23, 0.95) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2.2rem 2.2rem 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.025em;
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(59, 130, 246, 0.12);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.02rem;
        font-weight: 400;
        line-height: 1.6;
        max-width: 860px;
    }

    /* Metric Cards */
    .kpi-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }
    .kpi-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -5px rgba(59, 130, 246, 0.2);
    }
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-lbl {
        font-size: 0.76rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Candidate Result Card */
    .candidate-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
    }
    .candidate-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    .candidate-name-box {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .candidate-rank-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid var(--border-glass);
    }
    .candidate-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .score-circle-box {
        text-align: right;
    }
    .score-big-num {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .score-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    .badge-excellent { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-good { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-moderate { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-weak { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.3); }

    /* Pill Badges for Stats */
    .stat-pill {
        display: inline-flex;
        flex-direction: column;
        padding: 0.5rem 0.9rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stat-pill-lbl {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-pill-val {
        font-size: 0.98rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-top: 0.15rem;
    }

    /* Skills Badges */
    .skill-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.15rem;
    }
    .skill-matched { background: rgba(16, 185, 129, 0.12); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.25); }
    .skill-missing { background: rgba(244, 63, 94, 0.12); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.25); }
    .skill-extra { background: rgba(59, 130, 246, 0.12); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.25); }

    /* Copilot Question Card */
    .copilot-q-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .copilot-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #C084FC;
        margin-bottom: 0.4rem;
    }
    .copilot-q-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    .copilot-rubric {
        font-size: 0.82rem;
        color: #94A3B8;
        background: rgba(0, 0, 0, 0.25);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #8B5CF6;
    }

    /* Stacking Consensus Meter */
    .consensus-box {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        text-align: center;
        margin-top: 0.5rem;
    }
    .consensus-item-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #60A5FA;
    }
    .consensus-item-lbl {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Blind Review Badge */
    .blind-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Resources ──
@st.cache_resource
def get_inference_pipeline():
    from skillscout.inference_pipeline import InferencePipeline
    return InferencePipeline()

@st.cache_resource
def get_bert_model():
    from skillscout.ranking_feedback_nlp import load_bert
    return load_bert()


# ── Helper Functions ──
def get_score_color(score: float) -> str:
    if score >= 8.0: return "#34D399"
    if score >= 6.5: return "#60A5FA"
    if score >= 5.0: return "#FBBF24"
    return "#FB7185"

def get_score_badge_class(score: float) -> str:
    if score >= 8.0: return "badge-excellent"
    if score >= 6.5: return "badge-good"
    if score >= 5.0: return "badge-moderate"
    return "badge-weak"

def get_score_label(score: float) -> str:
    if score >= 8.0: return "High Match"
    if score >= 6.5: return "Good Fit"
    if score >= 5.0: return "Moderate"
    return "Weak Fit"

def render_skill_tags(skills: list, tag_class: str, max_show: int = 20) -> str:
    html = ""
    for skill in skills[:max_show]:
        html += f'<span class="skill-tag {tag_class}">{skill}</span>'
    if len(skills) > max_show:
        html += f'<span class="skill-tag {tag_class}">+{len(skills)-max_show} more</span>'
    return html

def get_anonymized_alias(index: int) -> str:
    code_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"]
    name = code_names[index % len(code_names)]
    return f"Candidate {name} (ID: #{index+1:02d})"

def generate_interview_questions(cand: dict) -> list:
    fb = cand.get("feedback", {})
    feats = cand.get("features", {})
    missing_mandatory = fb.get("missing_mandatory_skills", [])
    missing = fb.get("missing_skills", [])
    matched = fb.get("matched_skills", [])
    exp_years = feats.get("experience_years", 0)
    req_exp = feats.get("required_experience_years", 0)

    questions = []
    # 1. Technical Gap Probe
    if missing_mandatory:
        target_skill = missing_mandatory[0]
        questions.append({
            "tag": "🎯 Missing Mandatory Skill Probe",
            "skill": target_skill,
            "q": f"The job description specifies hands-on experience with '{target_skill}'. How have you applied {target_skill} in past work or personal projects, or how would you quickly bridge this requirement?",
            "rubric": f"Listen for hands-on architectural familiarity, transferable concepts from related libraries, and rapid self-directed learning capability in {target_skill}."
        })
    elif missing:
        target_skill = missing[0]
        questions.append({
            "tag": "🔍 Technical Gap Evaluation",
            "skill": target_skill,
            "q": f"Our team relies on '{target_skill}' for key workflows. Can you walk us through your familiarity with {target_skill} and what learning curve you anticipate?",
            "rubric": "A strong answer references foundational principles, official documentation habits, and prior examples of learning new frameworks under deadline."
        })

    # 2. Seniority & Experience Alignment
    if req_exp > 0 and exp_years < req_exp:
        questions.append({
            "tag": "⚡ Seniority & Ownership Challenge",
            "skill": "Experience Gap",
            "q": f"This role typically targets ~{req_exp:.0f} years of production experience (profile highlights ~{exp_years:.1f} yrs). Could you share the most technically challenging project you owned end-to-end?",
            "rubric": "Look for strong personal ownership, systematic debugging, quantifiable business metrics, and depth that offsets the tenure delta."
        })
    else:
        questions.append({
            "tag": "🚀 Production Reliability & Incident Response",
            "skill": "System Architecture",
            "q": "Can you share a time when a system, microservice, or model you built broke or degraded under live production load? How did you diagnose and remediate the issue?",
            "rubric": "Evaluates telemetry/monitoring awareness, systematic root-cause analysis, and implementation of preventative safeguards."
        })

    # 3. Verified Strength Deep-Dive
    if matched:
        top_skill = matched[0]
        questions.append({
            "tag": "💪 Core Strength Verification",
            "skill": top_skill,
            "q": f"Your profile demonstrates strong proficiency in '{top_skill}'. What are the most common performance bottlenecks or anti-patterns developers encounter with {top_skill} at scale?",
            "rubric": "High-signal answers show nuanced real-world implementation details rather than textbook syntax definitions."
        })

    return questions

def create_candidate_radar(cand_name: str, feats: dict, fb: dict) -> go.Figure:
    categories = ['Core CS Skills', 'Data/ML Skills', 'Experience Fit', 'Semantic Match', 'Requirement Fit']

    cs_val = min(10.0, (feats.get('cs_skill_count', 0) / 7.0) * 10.0)
    data_val = min(10.0, (feats.get('data_skill_count', 0) / 6.0) * 10.0)
    exp_years = feats.get('experience_years', 0)
    req_exp = max(1.0, feats.get('required_experience_years', 3.0))
    exp_val = min(10.0, (exp_years / req_exp) * 10.0)
    semantic_val = min(10.0, feats.get('bert_sim_score', 5.0))
    req_val = min(10.0, (fb.get('requirement_fit_pct', 50.0) / 10.0))

    values = [cs_val, data_val, exp_val, semantic_val, req_val]
    values.append(values[0])
    cats = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=cats,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.25)',
        line=dict(color='#3B82F6', width=2.5),
        name=cand_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=False,
                linecolor='rgba(255, 255, 255, 0.08)',
                gridcolor='rgba(255, 255, 255, 0.08)'
            ),
            angularaxis=dict(
                linecolor='rgba(255, 255, 255, 0.15)',
                gridcolor='rgba(255, 255, 255, 0.08)',
                tickfont=dict(size=10.5, color='#94A3B8')
            ),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=25, b=25),
        height=280,
        showlegend=False
    )
    return fig

def create_comparison_radar(name_a: str, feats_a: dict, fb_a: dict, name_b: str, feats_b: dict, fb_b: dict) -> go.Figure:
    categories = ['Core CS', 'Data/ML', 'Experience Depth', 'Semantic Match', 'Requirement Fit']

    def calc_vals(f, b):
        cs = min(10.0, (f.get('cs_skill_count', 0) / 7.0) * 10.0)
        data = min(10.0, (f.get('data_skill_count', 0) / 6.0) * 10.0)
        exp_y = f.get('experience_years', 0)
        req_y = max(1.0, f.get('required_experience_years', 3.0))
        exp = min(10.0, (exp_y / req_y) * 10.0)
        sem = min(10.0, f.get('bert_sim_score', 5.0))
        req = min(10.0, (b.get('requirement_fit_pct', 50.0) / 10.0))
        res = [cs, data, exp, sem, req]
        res.append(res[0])
        return res

    v_a = calc_vals(feats_a, fb_a)
    v_b = calc_vals(feats_b, fb_b)
    cats = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v_a, theta=cats, fill='toself', fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#3B82F6', width=2), name=name_a
    ))
    fig.add_trace(go.Scatterpolar(
        r=v_b, theta=cats, fill='toself', fillcolor='rgba(168, 85, 247, 0.2)',
        line=dict(color='#A855F7', width=2), name=name_b
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(size=11, color='#94A3B8')),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#F8FAFC'), orientation="h", y=-0.15, x=0.2),
        margin=dict(l=40, r=40, t=30, b=40),
        height=360
    )
    return fig


# ── Initialize Session State ──
if "revealed_candidates" not in st.session_state:
    st.session_state["revealed_candidates"] = set()
if "candidate_decisions" not in st.session_state:
    st.session_state["candidate_decisions"] = {}
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "last_jd" not in st.session_state:
    st.session_state["last_jd"] = ""


# ──────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Screening Configuration")

    # 1. Blind Screening Toggle (Core Unique Feature)
    st.markdown("---")
    st.markdown("##### 🕶️ Blind Screening Mode")
    blind_screening_on = st.toggle(
        "Enable Anti-Bias Anonymization",
        value=True,
        help="Masks candidate names, gender indicators, and college brand pedigree to enforce merit-first evaluation."
    )
    if blind_screening_on:
        st.caption("🛡️ **Active**: Names replaced with neutral IDs (`Candidate Alpha`, `Beta`, etc.). Pedigree signals neutralized.")
    else:
        st.caption("ℹ️ Standard mode: Original resume filenames visible.")

    # 2. Decision Pipeline Counters
    st.markdown("---")
    st.markdown("##### 📋 Recruiter Pipeline")
    decisions = st.session_state["candidate_decisions"]
    shortlisted_n = sum(1 for v in decisions.values() if v == "⭐ Shortlist")
    screen_n = sum(1 for v in decisions.values() if v == "📞 Phone Screen")
    hold_n = sum(1 for v in decisions.values() if v == "⏸️ On Hold")
    rejected_n = sum(1 for v in decisions.values() if v == "❌ Reject")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.metric("⭐ Shortlisted", shortlisted_n)
        st.metric("⏸️ On Hold", hold_n)
    with c_s2:
        st.metric("📞 Phone Screen", screen_n)
        st.metric("❌ Rejected", rejected_n)

    # 3. Model Architecture Metadata
    st.markdown("---")
    st.markdown("##### 🤖 ML Architecture")
    st.caption("• **Base Level**: CatBoost + XGBoost + LightGBM\n• **Meta-Model**: Ridge Blending Ensemble\n• **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)\n• **Calibration**: Confidence & Penalty Re-ranking")


# ──────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title-row">
        <span class="hero-title">🎯 SkillScout AI</span>
        <span class="hero-badge-pill">v2.0 Enterprise</span>
        <span class="hero-badge-pill" style="background: rgba(16, 185, 129, 0.12); color: #34D399; border-color: rgba(16, 185, 129, 0.3);">
            Stacked Ensemble
        </span>
    </div>
    <div class="hero-subtitle">
        AI-Powered Resume-to-Job Matching with Bias-Mitigated Screening, Multi-Model Stacking Explainability,
        and Recruiter Interview Question Copilot.
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# QUICK DEMO PRESETS
# ──────────────────────────────────────────────
st.markdown("##### ⚡ 1-Click Demo Presets")
preset_cols = st.columns(len(PRESET_JDS) + 1)

selected_preset_jd = None
for idx, (title, jd_body) in enumerate(PRESET_JDS.items()):
    with preset_cols[idx]:
        if st.button(title, use_container_width=True, help="Load this realistic Job Description instantly"):
            st.session_state["jd_input_value"] = jd_body
            st.toast(f"Loaded preset: {title}")

# Button to load sample PDF resumes
with preset_cols[-1]:
    load_bundled_resumes = st.button("📂 Load Bundled Resumes", use_container_width=True, help="Load 3 pre-built sample resumes for instant testing")
    if load_bundled_resumes:
        st.session_state["use_bundled_samples"] = True
        st.toast("Bundled sample resumes activated for testing!")


# ──────────────────────────────────────────────
# INPUT SECTION
# ──────────────────────────────────────────────
col_jd, col_upload = st.columns([3, 2], gap="large")

with col_jd:
    st.markdown("##### 📝 Job Description")
    initial_jd = st.session_state.get("jd_input_value", st.session_state.get("last_jd", ""))
    jd_text = st.text_area(
        "Job Description Text",
        value=initial_jd,
        height=260,
        placeholder="Paste full job description including requirements, qualifications, and core tech stack...",
        label_visibility="collapsed",
    )

with col_upload:
    st.markdown("##### 📄 Candidate Resumes (PDF)")
    use_bundled = st.session_state.get("use_bundled_samples", False)
    
    if use_bundled and SAMPLE_RESUMES_DIR.exists():
        st.info("💡 **Demo Mode Active**: Using 3 bundled sample resumes (`Alex Rivera - AI Eng`, `Priya Sharma - FullStack`, `David Chen - Analyst`). Uncheck to upload custom files.")
        if st.button("Cancel Demo Mode", type="secondary"):
            st.session_state["use_bundled_samples"] = False
            st.rerun()
        uploaded_files = []
    else:
        uploaded_files = st.file_uploader(
            "Upload Resume PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.caption(f"✅ {len(uploaded_files)} PDF resume{'s' if len(uploaded_files) > 1 else ''} ready for screening")
        else:
            st.caption("Drag and drop PDF files here, or click 'Load Bundled Resumes' above for an instant demo.")


# ──────────────────────────────────────────────
# LAUNCH ACTION
# ──────────────────────────────────────────────
st.markdown("")
col_launch, _, _ = st.columns([1.2, 2, 1])
with col_launch:
    run_ranking = st.button("🚀 Launch AI Screening & Ranking", type="primary", use_container_width=True)


if run_ranking:
    if not jd_text or not jd_text.strip():
        st.error("❌ Please provide a Job Description (or pick a 1-Click Preset above) before ranking.")
        st.stop()

    # Determine files to process
    pdf_paths = []
    if use_bundled and SAMPLE_RESUMES_DIR.exists():
        pdf_paths = [str(SAMPLE_RESUMES_DIR / f) for f in os.listdir(SAMPLE_RESUMES_DIR) if f.endswith(".pdf")]
    elif uploaded_files:
        for f in uploaded_files:
            tmp_p = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp_p, "wb") as out:
                out.write(f.read())
            pdf_paths.append(tmp_p)
    else:
        st.error("❌ Please upload at least one resume PDF or click 'Load Bundled Resumes'.")
        st.stop()

    progress_bar = st.progress(0, text="🔄 Initializing ML Ensemble & Sentence-BERT...")

    from skillscout.ranking_feedback_nlp import rank_resumes

    def update_progress(cur, total, msg):
        pct = int((cur / total) * 100) if total > 0 else 0
        progress_bar.progress(pct, text=f"🔄 [{cur}/{total}] {msg}")

    results = rank_resumes(
        jd_text=jd_text,
        resume_pdf_paths=pdf_paths,
        micro_dict=None,
        college_tier="Unknown",
        progress_callback=update_progress,
    )
    progress_bar.progress(100, text="✅ Screening and Ranking Completed!")

    st.session_state["last_results"] = results
    st.session_state["last_jd"] = jd_text


# ──────────────────────────────────────────────
# DASHBOARD RESULTS DISPLAY
# ──────────────────────────────────────────────
results = st.session_state.get("last_results")

if results:
    st.markdown("---")

    # 1. EXECUTIVE KPI SUMMARY ROW
    scores = [r["predicted_score"] for r in results if r["predicted_score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    top_score = max(scores) if scores else 0.0
    high_match_count = sum(1 for s in scores if s >= 8.0)
    avg_conf = sum(r.get("feedback", {}).get("ranking_confidence", 0) for r in results) / len(results) if results else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #60A5FA;">{len(results)}</div>
            <div class="kpi-lbl">Total Evaluated</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: {get_score_color(top_score)};">{top_score:.1f}</div>
            <div class="kpi-lbl">Top Match Score</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #F8FAFC;">{avg_score:.1f}</div>
            <div class="kpi-lbl">Talent Pool Avg</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #34D399;">{high_match_count}</div>
            <div class="kpi-lbl">High Match (≥8.0)</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color: #A78BFA;">{avg_conf:.0f}%</div>
            <div class="kpi-lbl">Ensemble Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # 2. MAIN APPLICATION TABS
    tab_rankings, tab_compare, tab_analytics, tab_bias, tab_export = st.tabs([
        "🏆 Ranked Leaderboard & Dossiers",
        "⚖️ Side-by-Side Dual Comparator",
        "📊 Talent Pool Analytics",
        "🛡️ Bias Mitigation & Fairness Audit",
        "📥 Export Center",
    ])

    # ── TAB 1: RANKED LEADERBOARD ──
    with tab_rankings:
        st.markdown("#### 🏆 Candidate Rankings & Deep Dossiers")

        if blind_screening_on:
            st.markdown("""
            <div class="blind-indicator">
                🕶️ <b>Blind Screening Active:</b> Candidate identities and filenames masked. You can reveal individual identities when ready.
            </div>
            """, unsafe_allow_html=True)

        for idx, r in enumerate(results):
            score = r["predicted_score"]
            raw_filename = os.path.basename(r["file_path"])
            rank = r["rank"]
            fb = r.get("feedback", {})
            feats = r.get("features", {})

            # Blind Anonymization check
            is_revealed = (raw_filename in st.session_state["revealed_candidates"]) or (not blind_screening_on)
            display_name = raw_filename if is_revealed else get_anonymized_alias(idx)

            medal = "🥇 " if rank == 1 else "🥈 " if rank == 2 else "🥉 " if rank == 3 else ""
            score_color = get_score_color(score)
            score_badge = get_score_badge_class(score)
            score_lbl = get_score_label(score)

            with st.container():
                st.markdown(f"""
                <div class="candidate-card">
                    <div class="candidate-header">
                        <div class="candidate-name-box">
                            <span class="candidate-rank-badge">{medal}#{rank}</span>
                            <div>
                                <div class="candidate-title">{display_name}</div>
                                <span style="font-size: 0.8rem; color: #94A3B8;">
                                    {fb.get('verdict', '')} • {fb.get('eligibility_label', '')}
                                </span>
                            </div>
                        </div>
                        <div class="score-circle-box">
                            <div class="score-big-num" style="color: {score_color};">{score:.1f}</div>
                            <span class="score-pill {score_badge}">{score_lbl}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1rem;">
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Experience</span>
                            <span class="stat-pill-val">{feats.get('experience_years', 0):.1f} yrs</span>
                        </div>
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Skills Found</span>
                            <span class="stat-pill-val">{feats.get('num_skills', 0)}</span>
                        </div>
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Skill Match %</span>
                            <span class="stat-pill-val" style="color: #34D399;">{fb.get('skill_match_pct', 0):.0f}%</span>
                        </div>
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Requirement Fit</span>
                            <span class="stat-pill-val" style="color: #60A5FA;">{fb.get('requirement_fit_pct', 0):.0f}%</span>
                        </div>
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Semantic BERT</span>
                            <span class="stat-pill-val">{feats.get('bert_sim_score', 0):.1f} / 10</span>
                        </div>
                        <div class="stat-pill">
                            <span class="stat-pill-lbl">Confidence</span>
                            <span class="stat-pill-val">{fb.get('ranking_confidence', 0):.0f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Action row: Decision Picker & Reveal Button
                c_act1, c_act2, c_act3 = st.columns([1.5, 1.2, 3])
                with c_act1:
                    current_dec = st.session_state["candidate_decisions"].get(raw_filename, "Review")
                    new_dec = st.selectbox(
                        "Decision Status",
                        options=["Review", "⭐ Shortlist", "📞 Phone Screen", "⏸️ On Hold", "❌ Reject"],
                        index=["Review", "⭐ Shortlist", "📞 Phone Screen", "⏸️ On Hold", "❌ Reject"].index(current_dec) if current_dec in ["Review", "⭐ Shortlist", "📞 Phone Screen", "⏸️ On Hold", "❌ Reject"] else 0,
                        key=f"dec_{idx}_{raw_filename}",
                        label_visibility="collapsed"
                    )
                    st.session_state["candidate_decisions"][raw_filename] = new_dec
                with c_act2:
                    if blind_screening_on:
                        if not is_revealed:
                            if st.button("👁️ Reveal Identity", key=f"rev_{idx}"):
                                st.session_state["revealed_candidates"].add(raw_filename)
                                st.rerun()
                        else:
                            st.caption(f"Real File: `{raw_filename}`")

                # Detailed Dossier Expander
                with st.expander(f"📋 Candidate Deep Dossier — {display_name}", expanded=(rank == 1)):
                    dos_tab1, dos_tab2, dos_tab3, dos_tab4 = st.tabs([
                        "🧠 Fit & Gap Analysis",
                        "🎯 Skill DNA Radar",
                        "❓ AI Recruiter Copilot",
                        "🤖 Model Consensus",
                    ])

                    # Sub-Tab 1: Fit & Gap
                    with dos_tab1:
                        st.markdown(f"**Verdict:** {fb.get('verdict_msg', '')}")
                        g_col1, g_col2 = st.columns(2)
                        with g_col1:
                            st.markdown("##### 💪 Key Strengths")
                            for s in fb.get("strengths", []):
                                st.markdown(f"• {s}")
                            if not fb.get("strengths"):
                                st.caption("No notable strengths recorded.")
                        with g_col2:
                            st.markdown("##### ⚠️ Candidate Gaps")
                            for g in fb.get("gaps", []):
                                st.markdown(f"• {g}")
                            if not fb.get("gaps"):
                                st.caption("No critical gaps detected.")

                        recs = fb.get("recommendations", [])
                        if recs:
                            st.markdown("##### 💡 Strategic Recommendations")
                            for rec in recs:
                                st.markdown(f"• {rec}")

                        roles = fb.get("suggested_roles", [])
                        if roles:
                            st.markdown("##### 🎯 Closest Alternate Roles")
                            r_html = " ".join([f"<span class='skill-tag skill-extra'>{role} ({pct}%)</span>" for role, pct in roles])
                            st.markdown(r_html, unsafe_allow_html=True)

                    # Sub-Tab 2: Skill DNA Radar & Tags
                    with dos_tab2:
                        r_col1, r_col2 = st.columns([1.1, 1])
                        with r_col1:
                            st.markdown("##### 🕸️ Multi-Dimensional Skill Radar")
                            radar_fig = create_candidate_radar(display_name, feats, fb)
                            st.plotly_chart(radar_fig, use_container_width=True, key=f"radar_chart_{idx}_{raw_filename}")
                        with r_col2:
                            st.markdown("##### 🔧 Detailed Skill Inventory")
                            missing_mandatory = fb.get("missing_mandatory_skills", [])
                            matched = fb.get("matched_skills", [])
                            missing = fb.get("missing_skills", [])
                            extra = fb.get("extra_skills", [])

                            if missing_mandatory:
                                st.markdown(f"**Missing Mandatory Skills ({len(missing_mandatory)}):**")
                                st.markdown(render_skill_tags(missing_mandatory, "skill-missing"), unsafe_allow_html=True)
                            if matched:
                                st.markdown(f"**Matched with JD ({len(matched)}):**")
                                st.markdown(render_skill_tags(matched, "skill-matched"), unsafe_allow_html=True)
                            if missing:
                                st.markdown(f"**Other Gaps ({len(missing)}):**")
                                st.markdown(render_skill_tags(missing, "skill-missing"), unsafe_allow_html=True)
                            if extra:
                                st.markdown(f"**Additional Resume Skills ({len(extra)}):**")
                                st.markdown(render_skill_tags(extra, "skill-extra"), unsafe_allow_html=True)

                    # Sub-Tab 3: AI Interview Copilot (Unique Feature)
                    with dos_tab3:
                        st.markdown("##### 🎙️ AI Recruiter Screening Guide")
                        st.caption("Tailored behavioral and technical interview questions dynamically generated to probe this candidate's specific skill and experience gaps.")

                        interview_questions = generate_interview_questions(r)
                        for q_idx, item in enumerate(interview_questions):
                            st.markdown(f"""
                            <div class="copilot-q-card">
                                <div class="copilot-tag">{item['tag']} • Focus: {item['skill']}</div>
                                <div class="copilot-q-text">"{item['q']}"</div>
                                <div class="copilot-rubric">
                                    <b>👂 What to listen for:</b> {item['rubric']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Sub-Tab 4: Model Consensus & Explainability
                    with dos_tab4:
                        st.markdown("##### 🤖 Stacked ML Ensemble Consensus")
                        st.caption("SkillScout executes a multi-stage stacked architecture combining gradient-boosted trees and sentence embeddings.")

                        raw_score = r.get("raw_model_score", score)
                        conf = fb.get("ranking_confidence", 85)

                        st.markdown(f"""
                        <div class="consensus-box">
                            <div>
                                <div class="consensus-item-val">{raw_score:.2f}</div>
                                <div class="consensus-item-lbl">Meta-Learner Score</div>
                            </div>
                            <div>
                                <div class="consensus-item-val" style="color: #34D399;">{score:.2f}</div>
                                <div class="consensus-item-lbl">Calibrated Score</div>
                            </div>
                            <div>
                                <div class="consensus-item-val" style="color: #A78BFA;">{conf:.0f}%</div>
                                <div class="consensus-item-lbl">Model Agreement</div>
                            </div>
                            <div>
                                <div class="consensus-item-val" style="color: #FBBF24;">{feats.get('pca_cos_sim', 0):.2f}</div>
                                <div class="consensus-item-lbl">PCA Cosine Sim</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("""
                        - **CatBoost**: Optimized for categorical college tier and structured experience attributes.
                        - **XGBoost**: Granular tree splitting across semantic cosine similarities and penalty factors.
                        - **LightGBM**: Fast leaf-wise tree regression on high-dimensional BERT projection vectors.
                        - **Stacking Meta-Model**: Ensembles individual base predictors to minimize false positives.
                        """)

    # ── TAB 2: SIDE-BY-SIDE CANDIDATE COMPARATOR ──
    with tab_compare:
        st.markdown("#### ⚖️ Head-to-Head Candidate Comparison")
        st.caption("Select any two candidates to directly compare skill coverage, experience, and suitability.")

        if len(results) < 2:
            st.info("Upload at least 2 resumes to activate side-by-side comparison.")
        else:
            c_sel1, c_sel2 = st.columns(2)
            options = [f"#{r['rank']} - {os.path.basename(r['file_path']) if not blind_screening_on else get_anonymized_alias(idx)}" for idx, r in enumerate(results)]

            with c_sel1:
                cand_a_idx = st.selectbox("Candidate A", range(len(options)), index=0, format_func=lambda x: options[x])
            with c_sel2:
                cand_b_idx = st.selectbox("Candidate B", range(len(options)), index=1 if len(options) > 1 else 0, format_func=lambda x: options[x])

            cand_a = results[cand_a_idx]
            cand_b = results[cand_b_idx]

            name_a = options[cand_a_idx]
            name_b = options[cand_b_idx]

            comp_col1, comp_col2 = st.columns([1.1, 1])
            with comp_col1:
                comp_fig = create_comparison_radar(
                    name_a, cand_a["features"], cand_a["feedback"],
                    name_b, cand_b["features"], cand_b["feedback"]
                )
                st.plotly_chart(comp_fig, use_container_width=True, key=f"comp_radar_{cand_a_idx}_{cand_b_idx}")

            with comp_col2:
                st.markdown("##### 📊 Metric Head-to-Head")
                comp_df = pd.DataFrame([
                    {"Metric": "Final Predicted Score", name_a: f"{cand_a['predicted_score']:.2f} / 10", name_b: f"{cand_b['predicted_score']:.2f} / 10"},
                    {"Metric": "Experience (Years)", name_a: f"{cand_a['features'].get('experience_years', 0):.1f} yrs", name_b: f"{cand_b['features'].get('experience_years', 0):.1f} yrs"},
                    {"Metric": "Total Skills Count", name_a: str(cand_a['features'].get('num_skills', 0)), name_b: str(cand_b['features'].get('num_skills', 0))},
                    {"Metric": "Skill Match %", name_a: f"{cand_a['feedback'].get('skill_match_pct', 0):.0f}%", name_b: f"{cand_b['feedback'].get('skill_match_pct', 0):.0f}%"},
                    {"Metric": "Requirement Fit %", name_a: f"{cand_a['feedback'].get('requirement_fit_pct', 0):.0f}%", name_b: f"{cand_b['feedback'].get('requirement_fit_pct', 0):.0f}%"},
                    {"Metric": "Semantic BERT Similarity", name_a: f"{cand_a['features'].get('bert_sim_score', 0):.2f}", name_b: f"{cand_b['features'].get('bert_sim_score', 0):.2f}"},
                    {"Metric": "Model Confidence", name_a: f"{cand_a['feedback'].get('ranking_confidence', 0):.0f}%", name_b: f"{cand_b['feedback'].get('ranking_confidence', 0):.0f}%"},
                ])
                st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Unique skill differentiators
            skills_a = set(cand_a["feedback"].get("matched_skills", []))
            skills_b = set(cand_b["feedback"].get("matched_skills", []))

            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.markdown(f"**Unique Skills in {name_a}:**")
                only_a = list(skills_a - skills_b)
                if only_a:
                    st.markdown(render_skill_tags(only_a, "skill-matched"), unsafe_allow_html=True)
                else:
                    st.caption("No unique matched skills.")

            with diff_col2:
                st.markdown(f"**Unique Skills in {name_b}:**")
                only_b = list(skills_b - skills_a)
                if only_b:
                    st.markdown(render_skill_tags(only_b, "skill-extra"), unsafe_allow_html=True)
                else:
                    st.caption("No unique matched skills.")

    # ── TAB 3: TALENT POOL ANALYTICS ──
    with tab_analytics:
        st.markdown("#### 📊 Interactive Talent Pool Intelligence")

        scatter_data = []
        for idx, r in enumerate(results):
            raw_fn = os.path.basename(r["file_path"])
            name = raw_fn if not blind_screening_on else get_anonymized_alias(idx)
            scatter_data.append({
                "Candidate": name,
                "Score": r["predicted_score"],
                "Experience": r["features"].get("experience_years", 0),
                "Skills_Count": r["features"].get("num_skills", 0),
                "Fit_Pct": r.get("feedback", {}).get("requirement_fit_pct", 0),
                "Verdict": r.get("feedback", {}).get("verdict", "Unknown")
            })
        df_scatter = pd.DataFrame(scatter_data)

        a_col1, a_col2 = st.columns([1.2, 1])
        with a_col1:
            st.markdown("##### 🎯 Experience vs. Match Score Distribution")
            fig_scatter = px.scatter(
                df_scatter,
                x="Experience",
                y="Score",
                color="Verdict",
                size="Skills_Count",
                hover_name="Candidate",
                hover_data=["Fit_Pct", "Skills_Count"],
                color_discrete_map={"Selected": "#34D399", "Good": "#60A5FA", "Moderate": "#FBBF24", "Weak": "#FB7185"},
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font=dict(color='#94A3B8'),
                xaxis=dict(title="Experience (Years)", gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title="Match Score (0-10)", range=[0, 10.5], gridcolor='rgba(255,255,255,0.08)'),
                height=350,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="analytics_scatter_chart")

        with a_col2:
            st.markdown("##### 📈 Score Ranking Distribution")
            fig_bar = px.bar(
                df_scatter.sort_values(by="Score", ascending=True),
                x="Score",
                y="Candidate",
                orientation='h',
                color="Score",
                color_continuous_scale=["#FB7185", "#FBBF24", "#60A5FA", "#34D399"],
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                font=dict(color='#94A3B8'),
                xaxis=dict(range=[0, 10], gridcolor='rgba(255,255,255,0.08)'),
                height=350,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="analytics_bar_chart")

    # ── TAB 4: BIAS MITIGATION & FAIRNESS AUDIT ──
    with tab_bias:
        st.markdown("#### 🛡️ Bias-Aware Ranking & Fairness Audit")
        st.markdown("""
        In traditional hiring workflows, resume screening algorithms frequently amplify unconscious bias
        associated with **candidate names** and **university prestige tiers**. SkillScout incorporates
        dedicated fairness mitigation layers to maintain screening neutrality.
        """)

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown("""
            ##### 🔍 How SkillScout Mitigates Bias
            1. **Decoupled Identity Processing**: Candidate names, email domains, and phone numbers are isolated from the ML inference vectors.
            2. **Categorical College Tier Normalization**: Rather than raw prestige labels, institutions are mapped via an empirical tier encoder trained to penalize prestige bias.
            3. **Evidence-Based Feature Weighting**: 85%+ of scoring weight is anchored strictly on verifiable skill presence, project achievements, and semantic alignment.
            """)
        with b_c2:
            st.markdown("""
            ##### 📋 Compliance & Fairness Checklist
            - ✅ **Name Neutrality**: Zero reliance on demographic proxies.
            - ✅ **Equal Experience Treatment**: Objective tenure calculation through date range parsing.
            - ✅ **Audit Trail**: Every score is accompanied by transparent feedback rationales.
            - ✅ **Blind Review Integration**: Ability to shortlist before viewing candidate demographics.
            """)

    # ── TAB 5: EXPORT CENTER ──
    with tab_export:
        st.markdown("#### 📥 Export Decision Dossiers")
        st.caption("Generate downloadable candidate reports formatted for executive hiring reviews and ATS ingestion.")

        rows_full = []
        for r in results:
            fb = r.get("feedback", {})
            feats = r.get("features", {})
            fn = os.path.basename(r["file_path"])
            status = st.session_state["candidate_decisions"].get(fn, "Under Review")
            rows_full.append({
                "Rank": r["rank"],
                "Candidate File": fn,
                "Recruiter Status": status,
                "Score (0-10)": round(r["predicted_score"], 2),
                "Raw Meta Score": round(r.get("raw_model_score", 0), 2),
                "Verdict": fb.get("verdict", ""),
                "Confidence %": fb.get("ranking_confidence", 0),
                "Requirement Fit %": fb.get("requirement_fit_pct", 0),
                "Experience (Yrs)": feats.get("experience_years", 0),
                "Total Skills": feats.get("num_skills", 0),
                "Matched Skills": ", ".join(fb.get("matched_skills", [])),
                "Missing Skills": ", ".join(fb.get("missing_skills", [])),
                "Missing Mandatory": ", ".join(fb.get("missing_mandatory_skills", [])),
            })
        full_df = pd.DataFrame(rows_full)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.download_button(
                label="📥 Download Full Talent Pool Report (CSV)",
                data=full_df.to_csv(index=False),
                file_name=f"skillscout_full_ranking_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with ex_col2:
            shortlist_df = full_df[full_df["Recruiter Status"] == "⭐ Shortlist"]
            st.download_button(
                label=f"⭐ Download Shortlisted Only ({len(shortlist_df)} Candidates)",
                data=shortlist_df.to_csv(index=False),
                file_name=f"skillscout_shortlisted_{timestamp}.csv",
                mime="text/csv",
                disabled=(len(shortlist_df) == 0),
                use_container_width=True
            )

        st.dataframe(full_df, use_container_width=True, hide_index=True)

# ── Footer ──
st.markdown("---")
st.caption(f"🎯 SkillScout v2.0 Enterprise • Active ML Pipeline • Timestamp: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
