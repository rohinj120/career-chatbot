# O*NET-sourced typical education requirements per occupation.
# Covers the most commonly queried occupations; expand as needed.

EDUCATION_LOOKUP: dict[str, dict] = {
    # ── Software / IT ────────────────────────────────────────────────────────
    "software developers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Software Engineering", "Information Technology"],
        "certifications": ["AWS Certified Developer", "Microsoft Azure", "Google Cloud Professional"],
        "notes": "Some employers accept equivalent work experience or a coding bootcamp in place of a degree.",
    },
    "software quality assurance analysts and testers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Systems"],
        "certifications": ["ISTQB", "Certified Software Tester (CSTE)"],
        "notes": None,
    },
    "computer and information systems managers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Systems", "Business Administration"],
        "certifications": ["PMP", "CISSP", "AWS Solutions Architect"],
        "notes": "Many senior roles prefer or require a master's degree (MBA or MS in IS).",
    },
    "database administrators": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Systems", "Mathematics"],
        "certifications": ["Oracle Certified Professional", "Microsoft Certified: Azure Database Administrator", "MongoDB Certified DBA"],
        "notes": None,
    },
    "network and computer systems administrators": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Technology", "Network Engineering"],
        "certifications": ["CompTIA Network+", "Cisco CCNA", "Microsoft Certified: Azure Administrator"],
        "notes": "An associate degree combined with certifications is often sufficient for entry-level roles.",
    },
    "information security analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Cybersecurity", "Information Assurance", "Information Systems"],
        "certifications": ["CompTIA Security+", "CISSP", "CEH (Certified Ethical Hacker)", "CISA"],
        "notes": "Professional certifications such as CISSP or CISA are highly valued and sometimes substituted for advanced degrees.",
    },
    "cybersecurity analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Cybersecurity", "Computer Science", "Information Systems"],
        "certifications": ["CompTIA Security+", "CISSP", "CEH", "CISA", "CISM"],
        "notes": None,
    },
    "computer systems analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Systems", "Business"],
        "certifications": ["CompTIA A+", "Project Management Professional (PMP)"],
        "notes": None,
    },
    "web developers": {
        "typical_education": "Associate's degree or Bachelor's degree",
        "common_fields": ["Web Development", "Computer Science", "Graphic Design"],
        "certifications": ["Google UX Design Certificate", "Meta Front-End Developer Certificate"],
        "notes": "Portfolio of projects often matters as much as formal education.",
    },
    "cloud engineers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Computer Science", "Information Technology", "Cloud Computing"],
        "certifications": ["AWS Solutions Architect", "Google Cloud Professional Cloud Architect", "Microsoft Azure Solutions Architect"],
        "notes": "Cloud certifications are heavily weighted by employers; hands-on experience is critical.",
    },
    # ── Data / AI ────────────────────────────────────────────────────────────
    "data scientists": {
        "typical_education": "Master's degree (common); Bachelor's degree (entry-level)",
        "common_fields": ["Statistics", "Mathematics", "Computer Science", "Data Science"],
        "certifications": ["Google Professional Data Engineer", "IBM Data Science Professional", "Databricks Certified Associate"],
        "notes": "A PhD is preferred in research-heavy or academic roles. Many practitioners enter with a bachelor's and gain skills through online courses.",
    },
    "data analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Mathematics", "Statistics", "Computer Science", "Economics"],
        "certifications": ["Google Data Analytics Certificate", "Microsoft Power BI Certification", "Tableau Desktop Specialist"],
        "notes": None,
    },
    "machine learning engineers": {
        "typical_education": "Master's degree (common); Bachelor's degree (entry-level)",
        "common_fields": ["Computer Science", "Mathematics", "Electrical Engineering", "AI"],
        "certifications": ["AWS Machine Learning Specialty", "TensorFlow Developer Certificate", "Google Professional ML Engineer"],
        "notes": None,
    },
    "statisticians": {
        "typical_education": "Master's degree",
        "common_fields": ["Statistics", "Mathematics", "Biostatistics", "Economics"],
        "certifications": ["ASA Accredited Statistician (A.Stat)"],
        "notes": "Many federal government positions require a master's degree.",
    },
    # ── Business / Finance ───────────────────────────────────────────────────
    "accountants and auditors": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Accounting", "Finance", "Business Administration"],
        "certifications": ["CPA (Certified Public Accountant)", "CMA (Certified Management Accountant)", "CIA (Certified Internal Auditor)"],
        "notes": "CPA licensure typically requires 150 credit hours (5 years of college).",
    },
    "financial analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Finance", "Accounting", "Economics", "Mathematics"],
        "certifications": ["CFA (Chartered Financial Analyst)", "CFP (Certified Financial Planner)", "FRM"],
        "notes": None,
    },
    "management analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Business Administration", "Management", "Finance", "Economics"],
        "certifications": ["CMC (Certified Management Consultant)"],
        "notes": "Many senior consultants hold an MBA.",
    },
    "market research analysts": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Market Research", "Statistics", "Business", "Communications"],
        "certifications": ["PRC (Professional Researcher Certification)"],
        "notes": None,
    },
    # ── Healthcare ───────────────────────────────────────────────────────────
    "registered nurses": {
        "typical_education": "Associate's degree or Bachelor's degree in Nursing (BSN)",
        "common_fields": ["Nursing", "Biology", "Health Sciences"],
        "certifications": ["RN License (NCLEX-RN)", "BLS/ACLS", "Specialty certifications (CCRN, CEN)"],
        "notes": "BSN is increasingly preferred by hospitals; associate's degree (ADN) is also a valid entry path.",
    },
    "physicians and surgeons": {
        "typical_education": "Doctoral degree (MD or DO) + residency",
        "common_fields": ["Medicine", "Pre-Med Biology/Chemistry"],
        "certifications": ["State Medical License", "Board Certification by specialty"],
        "notes": "Typically 4 years undergraduate + 4 years medical school + 3–7 years residency.",
    },
    # ── Engineering ──────────────────────────────────────────────────────────
    "mechanical engineers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Mechanical Engineering", "Aerospace Engineering", "Physics"],
        "certifications": ["PE (Professional Engineer) License", "Six Sigma", "PMP"],
        "notes": None,
    },
    "electrical engineers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Electrical Engineering", "Electronics", "Computer Engineering"],
        "certifications": ["PE License", "IEEE Certifications"],
        "notes": None,
    },
    "civil engineers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Civil Engineering", "Structural Engineering", "Environmental Engineering"],
        "certifications": ["PE (Professional Engineer) License", "PMP"],
        "notes": "PE licensure is required to offer engineering services directly to the public.",
    },
    # ── Education ────────────────────────────────────────────────────────────
    "elementary school teachers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Education", "Elementary Education", "Child Development"],
        "certifications": ["State Teaching License/Certification"],
        "notes": "Student teaching experience is required during degree programs.",
    },
    "postsecondary teachers": {
        "typical_education": "Doctoral degree (PhD) or Master's degree",
        "common_fields": ["Discipline-specific field (e.g., Chemistry, History)"],
        "certifications": None,
        "notes": "PhD is typically required for tenure-track positions; master's may suffice for community college or adjunct roles.",
    },
    # ── Legal ────────────────────────────────────────────────────────────────
    "lawyers": {
        "typical_education": "Doctoral degree (Juris Doctor, JD)",
        "common_fields": ["Law", "Pre-Law (Political Science, Philosophy, English)"],
        "certifications": ["State Bar License"],
        "notes": "Must pass the bar examination after earning a JD (3 years post-undergraduate).",
    },
    # ── Creative / Design ────────────────────────────────────────────────────
    "graphic designers": {
        "typical_education": "Bachelor's degree",
        "common_fields": ["Graphic Design", "Fine Arts", "Visual Communications"],
        "certifications": ["Adobe Certified Professional"],
        "notes": "Portfolio quality is often weighted equally with formal credentials.",
    },
}

# Alias map for common query variants → canonical key
_ALIASES: dict[str, str] = {
    "software developer": "software developers",
    "software engineer": "software developers",
    "software engineers": "software developers",
    "data scientist": "data scientists",
    "data analyst": "data analysts",
    "database administrator": "database administrators",
    "dba": "database administrators",
    "information security analyst": "information security analysts",
    "cybersecurity analyst": "cybersecurity analysts",
    "cloud engineer": "cloud engineers",
    "web developer": "web developers",
    "machine learning engineer": "machine learning engineers",
    "network administrator": "network and computer systems administrators",
    "systems analyst": "computer systems analysts",
    "financial analyst": "financial analysts",
    "accountant": "accountants and auditors",
    "auditor": "accountants and auditors",
    "registered nurse": "registered nurses",
    "nurse": "registered nurses",
    "physician": "physicians and surgeons",
    "doctor": "physicians and surgeons",
    "mechanical engineer": "mechanical engineers",
    "electrical engineer": "electrical engineers",
    "civil engineer": "civil engineers",
    "lawyer": "lawyers",
    "attorney": "lawyers",
    "graphic designer": "graphic designers",
    "statistician": "statisticians",
    "market research analyst": "market research analysts",
    "management analyst": "management analysts",
    "elementary teacher": "elementary school teachers",
    "professor": "postsecondary teachers",
}


def lookup_education(occupation_title: str) -> dict | None:
    """Return education data for *occupation_title*, or None if not found."""
    key = occupation_title.strip().lower()
    if key in EDUCATION_LOOKUP:
        return EDUCATION_LOOKUP[key]
    if key in _ALIASES:
        return EDUCATION_LOOKUP[_ALIASES[key]]
    # Partial match: if the query title is contained in a known key or vice-versa
    for canonical in EDUCATION_LOOKUP:
        if canonical in key or key in canonical:
            return EDUCATION_LOOKUP[canonical]
    return None


def format_education_response(occupation_title: str, data: dict) -> str:
    """Render education data as a clean bullet-point answer."""
    lines = [f"Education Requirements for {occupation_title}:\n"]
    lines.append(f"- Typical education: {data['typical_education']}")
    if data.get("common_fields"):
        lines.append(f"- Common fields of study: {', '.join(data['common_fields'])}")
    if data.get("certifications"):
        lines.append(f"- Relevant certifications: {', '.join(data['certifications'])}")
    if data.get("notes"):
        lines.append(f"- Notes: {data['notes']}")
    return "\n".join(lines)
