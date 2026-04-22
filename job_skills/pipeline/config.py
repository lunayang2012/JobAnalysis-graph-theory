from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


from enum import Enum

class Mode(str, Enum):
    fast = "fast"
    precise = "precise"


@dataclass
class Config:
    # Base paths
    project_root: Path
    data_dir: Path
    input_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path

    # Input files (defaults can be overridden via env vars)
    hourly_path: Path | None = None
    yearly_path: Path | None = None
    combined_path: Path | None = None

    # Mode: "fast" vs "precise" for FP model
    mode: Mode = Mode.precise

    # Preprocessing knobs
    min_tokens: int = 15

    # ============================================================
    # PREPROCESS: spaCy description normalization (optional)
    # ============================================================
    enable_spacy_clean: bool = False
    spacy_model: str = "en_core_web_sm"
    # 0 = do not truncate; otherwise keep first N tokens
    spacy_max_words: int = 100
    # Optional: disable spaCy pipeline components, e.g. ("ner",)
    spacy_disable: tuple[str, ...] = ()

    # ============================================================
    # PREPROCESS: fuzzy title clustering / collapsing (optional)
    # ============================================================
    enable_title_fuzzy_dedupe: bool = False
    # Similarity threshold in [0,100]; Ricky used 90
    title_fuzzy_similarity: int = 90
    # "token_sort_ratio" (Ricky) or "token_set_ratio" (more forgiving)
    title_fuzzy_scorer: str = "token_sort_ratio"
    # If True: keep only 1 row per canonical title (changes unit of analysis)
    title_fuzzy_collapse: bool = False
    # When collapsing: "first" or "most_recent" (uses date_posted if available)
    title_fuzzy_keep: str = "first"
    # Keep audit columns (cluster id/size) even if collapsing
    keep_title_cluster_cols: bool = True

    # Optional: reference date for computing posting_age_days in preprocess
    reference_date: str | None = None

    # FP model / threshold edits
    #target_keep: int = 10_000  # number of postings you'd like to retain (tunable)
    #tau_grid: tuple[float, float, int] = (0.5, 0.999, 200)

    #fixed prob cutoff
    fp_tau: float = 0.15 

    # Domain filtering
    use_domain_filter: bool = False
    use_soft_domain_filter: bool = False
    domain_allowlist: list[str] = field(default_factory=lambda: [
        "biology", "biotech", "biotechnology", "chemistry", "pharma",
        "pharmaceutical", "lab", "laboratory", "clinical"
    ])
    # ============================================================
    # FIELD-AWARE WEAK LABEL PATTERNS
    # ============================================================

    pos_title_patterns: list[str] = field(default_factory=lambda: [
        # === CORE SCIENTIFIC TITLES ===
        r"\bscientist\b",
        r"\bassociate scientist\b",
        r"\bsenior scientist\b",
        r"\bprincipal scientist\b",
        r"\bstaff scientist\b",
        r"\bresearch scientist\b",
        r"\bresearch associate\b",
        r"\bresearch engineer\b",
        r"\bresearch analyst\b",
        r"\bpostdoctoral\b",
        r"\bpost doctoral\b",
        r"\bpost doc\b",
        r"\bpostdoctoral (research|associate|fellow|scholar)\b",

        # === CHEMISTRY & MATERIALS ===
        r"\bchemist\b",
        r"\banalytical chemist\b",
        r"\borganic chemist\b",
        r"\bmedicinal chemist\b",
        r"\bmaterials (scientist|engineer)\b",
        r"\bpolymer( scientist| chemistry| chemist)\b",

        # === BIOLOGY / BIOMEDICAL ===
        r"\bbiologist\b",
        r"\bmicrobiologist\b",
        r"\bmolecular biologist\b",
        r"\bimmunologist\b",
        r"\btoxicologist\b",

        # === ENGINEERING / COMPUTATIONAL ===
        r"\bbioengineer\b",
        r"\bbioprocess engineer\b",
        r"\bbiomedical engineer\b",
        r"\bcomputational (biolog|chemist|scientist)\b",
        r"\bdata scientist\b",
        r"\bmachine learning (scientist|engineer)\b",

        # === CLINICAL / DIAGNOSTIC (TOGGLEABLE) ===
        r"\bnurse\b",
        r"\bregistered nurse\b",
        r"\brn\b",
        r"\brespiratory therapist\b",
        r"\bdialysis\b",
        r"\bmedical technologist\b",
        r"\bmedical laboratory technician\b",
        r"\bmlt\b",
        r"\bhistotechnician\b",
        r"\bpathologist\b",
        r"\bsurgical technologist\b",

        # NEW: postdoc variants
        r"\bpostdoc\b",
        r"\bpost[- ]?doctoral (fellow|scholar|associate)\b",

        # NEW: generic lab tech roles
        r"\blab tech\b",
        r"\blaboratory tech\b",
        r"\blab(oratory)? technician\b",
        r"\blab(oratory)? technologist\b",

        # NEW: forensic & crime scene
        r"\bforensic (scientist|analyst|technologist)\b",
        r"\bcrime scene (investigator|analyst|technician)\b",

        # NEW: bacteriology / microbiology variants
        r"\bbacteriolog(ist|y)\b",
        r"\bclinical microbiolog(ist|y)\b",

        # NEW: immunology / toxicology / cellular / molecular
        r"\bimmunolog(ist|y)\b",
        r"\btoxicolog(ist|y)\b",
        r"\bcellular (biolog(ist|y)|immunology|microbiology)\b",
        r"\bmolecular (biolog(ist|y)|geneticist|pathology)\b",

        # NEW: bioinformatics
        r"\bbioinformatic(s|ian)\b",

        # NEW: metrology / metrologist
        r"\bmetrologist\b",
        r"\bmetrology (engineer|scientist|specialist|r&d)\b",

        # NEW: cytology / cytotechnologist
        r"\bcytotechnolog(ist|y)\b",

        # NEW: materials / material science variants
        r"\bmaterials? (scientist|engineer|researcher)\b",
        r"\bmaterials? (chemistry|science)\b",

        # NEW: process & application roles explicitly in science context
        r"\b(process|manufacturing) (engineer|scientist|chemist)\b",
        r"\bapplication(s)? analyst\b",
        r"\b(application|systems) (scientist|engineer)\b",

        # NEW: chemical operator in industrial chemistry
        r"\bchemical operator\b",
        r"\bchemical process operator\b",

        # NEW: x-ray domain roles
        r"\bx[- ]?ray (technologist|technician|scientist)\b",

        r"\bgenomic(s)?\b",
        r"\bcytogenetic(s|ist)\b",
        r"\bbiochemist\b",
        r"\bchemical technician\b",
        r"\bchemical engineer\b",
        r"\bsynthetic biology\b",

        r"\bresearch assistant\b",
        r"\bresearch specialist\b",
        r"\bresearch technician\b",
        r"\blab assistant\b",
        r"\blaboratory manager\b",
        r"\blab( oratory)? manager\b",
        r"\blab( oratory)? supervisor\b",
        r"\bchemical engineer\b",
        r"\bproduct development\b",
        r"\bresearch development\b",

        r"\bquality control (analyst|associate|technician|laboratory|chemistry|microbiology)\b",
        r"\bqc (analyst|chemist|chemistry|microbiology|lab)\b",

        # === NEW: research support & lab leadership roles (from dropped FNs) ===
        r"\bresearch assistant\b",
        r"\bresearch specialist\b",
        r"\bresearch technician\b",
        r"\blab assistant\b",
        r"\blaboratory manager\b",
        r"\blab( oratory)? manager\b",
        r"\blab( oratory)? supervisor\b",

        # QC roles in a clearly lab/chem/microbiology context
        r"\bquality control (analyst|associate|technician|laboratory|chemistry|microbiology)\b",
        r"\bqc (analyst|chemist|chemistry|microbiology|lab)\b",

        # Microbiology specialist-like titles
        r"\bmicrobiology specialist\b",

        r"\bvector biology\b",
        r"\bquantitative pharmacolog(?:y|ist|ists)?\b",
        r"\bpharmacometrics?\b",
        r"\bformulation development\b",
        r"\bformulation dev\b",
        r"\badvanced therap(?:y|ies)\b",
        r"\bcell therapy\b",
        r"\bbiological defense\b",
        r"\bprincipal investigator\b",
        r"\blab operations?\b",
        r"\blaboratory supervisor\b",
        r"\bmanager lab\b",
        r"research\s*&\s*development\s*(co-?op|intern(ship)?)",
        r"research and development\s*(co-?op|intern(ship)?)",
        r"\br&d\s*(co-?op|intern(ship)?)",

        ])

    pos_desc_patterns: list[str] = field(default_factory=lambda: [
        # === LAB TECHNIQUES ===
        r"\bassay\b",
        r"\bassay development\b",
        r"\bchromatograph(y|ic)\b",
        r"\bHPLC\b",
        r"\bLC[- ]?MS\b",
        r"\bGC[- ]?MS\b",
        r"\bmass spec\b",
        r"\bspectroscop(y|ic)\b",

        # === MOLECULAR / CELLULAR METHODS ===
        r"\bPCR\b",
        r"\bqPCR\b",
        r"\bRT[- ]?PCR\b",
        r"\bELISA\b",
        r"\bwestern blot\b",
        r"\bSDS[- ]?PAGE\b",
        r"\bIHC\b",
        r"\bimmunohistochemistry\b",
        r"\bflow cytometry\b",
        r"\bFACS\b",
        r"\bmicroscopy\b",
        r"\bcell culture\b",
        r"\btissue culture\b",
        r"\bstem cell\b",

        r"\brna\b",
        r"\bdna\b",
        r"\bNGS\b",
        r"\bvaccine\b",
        r"\bpreclinical\b",

        r"\bmass spectrometry\b",
        r"\bsterile technique\b",
        r"\bcell culture\b",
        r"\bprotein purification\b",
        r"\banalytical chemistry\b",
        


    ])

    neg_title_patterns: list[str] = field(default_factory=lambda: [
        # === SERVICE / MAINTENANCE ===
        r"\bcleaner\b",
        r"\bjanitor\b",
        r"\bjanitorial\b",
        r"\bcustodian\b",
        r"\bhousekeeper\b",
        r"\bhousekeeping\b",
        r"\bmaintenance\b",
        r"\bgroundskeeper\b",
        r"\bparking porter\b",
        r"\bhandyman\b",

        # === RETAIL / FOOD / HOSPITALITY ===
        r"\bretail\b",
        r"\bcashier\b",
        r"\bshoprite\b",
        r"\bsales\b",
        r"\bserver\b",
        r"\bwaiter\b",
        r"\bwaitress\b",
        r"\bbarista\b",
        r"\bline cook\b",
        r"\bchef\b",
        r"\bbanquet\b",
        r"\bhotel\b",
        r"\bguest services?\b",

        # === EDUCATION ===
        r"\bteacher\b",
        r"\binstructor\b",
        r"\btutor\b",
        r"\badjunct\b",
        r"\bprofessor\b",
        r"\blecturer\b",
        r"\bschool\b",

        # === TRANSPORT / DELIVERY ===
        r"\bdriver\b",
        r"\bcourier\b",
        r"\bvalet\b",
        r"\bshuttle\b",
        r"\btransport\b",
    ])

    neg_desc_patterns: list[str] = field(default_factory=list)

    # FP model config
    fp_model_type: Literal["logreg", "linear_svc"] = "logreg"

    # Fast vs precise TF-IDF settings
    max_features_fast: int = 50_000
    max_features_precise: int = 150_000
    min_df_fast: int = 5
    min_df_precise: int = 2
    cv_fast: int = 2
    cv_precise: int = 3
    max_iter_logreg: int = 1000

    # BERTopic settings
    bertopic_len_text_min: int = 200
    bertopic_p_rnd_cutoff: float = 0.5
    bertopic_require_label_weak: bool = False
    bertopic_embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    bertopic_use_embeddings_cache: bool = True
    bertopic_save_embeddings_cache: bool= True
    # BERTopic text mode: "legacy" uses precomputed text_for_bertopic,
    # "combo" builds a richer representation from multiple fields.
    bertopic_text_mode: str = "legacy" #"legacy"= just description, "combo" = multiple fields

    # === LLM topic labeling defaults ===
    topic_labeling_backend: str = "openai"
    topic_labeling_model: str = "gpt-5-mini"
    topic_labeling_max_examples: int = 5
    topic_labeling_max_chars: int = 400
    topic_labeling_temperature: float = 0.2
    topic_labeling_language: str = "en"

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data"
        input_dir = data_dir / "inputs"
        interim_dir = data_dir / "interim"
        processed_dir = data_dir / "processed"
        models_dir = root / "models"

        # Create dirs if missing
        for p in (input_dir, interim_dir, processed_dir, models_dir):
            p.mkdir(parents=True, exist_ok=True)

        # Optional env overrides
        hourly = os.getenv("JOB_SKILLS_HOURLY")
        yearly = os.getenv("JOB_SKILLS_YEARLY")
        combined = os.getenv("JOB_SKILLS_COMBINED")

        raw_mode = os.getenv("JOB_SKILLS_MODE", "precise").lower()
        mode = Mode.fast if raw_mode == "fast" else Mode.precise

        # Optional preprocess env overrides (strings -> bool/int as needed)
        def _env_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}

        enable_spacy_clean = _env_bool("JOB_SKILLS_ENABLE_SPACY_CLEAN", False)
        spacy_model = os.getenv("JOB_SKILLS_SPACY_MODEL", "en_core_web_sm")
        spacy_max_words = int(os.getenv("JOB_SKILLS_SPACY_MAX_WORDS", "100"))

        enable_title_fuzzy_dedupe = _env_bool("JOB_SKILLS_ENABLE_TITLE_FUZZY_DEDUPE", False)
        title_fuzzy_similarity = int(os.getenv("JOB_SKILLS_TITLE_FUZZY_SIMILARITY", "90"))
        title_fuzzy_scorer = os.getenv("JOB_SKILLS_TITLE_FUZZY_SCORER", "token_sort_ratio")
        title_fuzzy_collapse = _env_bool("JOB_SKILLS_TITLE_FUZZY_COLLAPSE", False)
        title_fuzzy_keep = os.getenv("JOB_SKILLS_TITLE_FUZZY_KEEP", "first")
        keep_title_cluster_cols = _env_bool("JOB_SKILLS_KEEP_TITLE_CLUSTER_COLS", True)

        reference_date = os.getenv("JOB_SKILLS_REFERENCE_DATE")

        return cls(
            project_root=root,
            data_dir=data_dir,
            input_dir=input_dir,
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            models_dir=models_dir,
            hourly_path=Path(hourly) if hourly else None,
            yearly_path=Path(yearly) if yearly else None,
            combined_path=Path(combined) if combined else None,
            mode=mode,
            enable_spacy_clean=enable_spacy_clean,
            spacy_model=spacy_model,
            spacy_max_words=spacy_max_words,
            enable_title_fuzzy_dedupe=enable_title_fuzzy_dedupe,
            title_fuzzy_similarity=title_fuzzy_similarity,
            title_fuzzy_scorer=title_fuzzy_scorer,
            title_fuzzy_collapse=title_fuzzy_collapse,
            title_fuzzy_keep=title_fuzzy_keep,
            keep_title_cluster_cols=keep_title_cluster_cols,
            reference_date=reference_date,
        )