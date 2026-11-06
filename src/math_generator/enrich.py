"""Enrich a generated math-QA dataset by sending each problem to a vLLM
serve endpoint and asking the model to rewrite it (and, optionally, judge
its own rewrite).

Usage:
    # 1. Generate a base dataset with the deterministic engine
    python -m math_generator --n 1000 --out base.jsonl

    # 2. Have a vLLM server running, e.g.
    #    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
    #    (default endpoint: http://localhost:8000)

    # 3. Enrich
    python -m math_generator.enrich \\
        --input base.jsonl --output enriched.jsonl \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --n 200 --workers 8

Configuration via environment variables
---------------------------------------
The CLI accepts the following env vars (CLI flags override env, env overrides
built-in defaults). Project-specific names are checked first, then the
standard OPENAI_* names so an existing OpenAI-SDK setup works unchanged.

    MATH_GEN_VLLM_ENDPOINT, OPENAI_BASE_URL, OPENAI_API_BASE
        vLLM /v1 base URL (default http://localhost:8000)
    MATH_GEN_VLLM_MODEL,    OPENAI_MODEL
        model name as served by vLLM (no built-in default — must come from
        env or --model)
    MATH_GEN_VLLM_API_KEY,  OPENAI_API_KEY
        bearer token if vLLM was launched with --api-key
    MATH_GEN_VLLM_TIMEOUT
        per-request timeout in seconds (default 180)

You can either export them ahead of time or source a small .env-style file:

    cat > .env.vllm <<'EOF'
    MATH_GEN_VLLM_ENDPOINT=http://my-vllm-host:8000
    MATH_GEN_VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    MATH_GEN_VLLM_API_KEY=sk-...
    EOF
    set -a; source .env.vllm; set +a
    python -m math_generator.enrich --input base.jsonl --output enriched.jsonl

The script also accepts `--env-file PATH` to read a `.env`-style file with
no external dependencies (lines like `KEY=value`, `#` comments and blanks
ignored). Existing process env wins over the file.

The script picks one of several rewriter prompts at random for each record.
Different prompts encourage different rewriting *modes* — pure word-problem
framing, narrative, engineering scenario, deliberately added irrelevant
information, hidden operation (forces the solver to figure out *what* to
compute), or a strict formal restatement. After the rewrite, the same model
is asked to rate the rewrite on a few criteria; a record is marked
`keep=True` iff the judge says it is correct, natural, and not trivialised.

Output JSONL fields per record:
    seed_problem, seed_answer, seed_meta
    rewrite_prompt_name, rewrite_temperature
    rewritten_problem, rewritten_solution
    judge: {correctness, correctness_reason, naturalness, clarity,
            difficulty_preservation, overall_keep}
    rewrite_raw, judge_raw           (full model outputs for debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests


# ─────────────────────────── ENV / CONFIG ───────────────────────────

DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_TIMEOUT = 600


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable from `names`."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def load_env_file(path: str) -> int:
    """Tiny .env loader: `KEY=value` lines, # comments and blank lines ignored.
    Existing os.environ entries are NOT overwritten. Returns the number of
    new variables actually set."""
    set_count = 0
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
                    set_count += 1
    except FileNotFoundError:
        sys.exit(f"--env-file: file not found: {path}")
    return set_count


# ─────────────────────────── DIVERSITY POOLS ───────────────────────────
# Gemma's default rewrites collapse onto a tiny cast ("Leo" appeared in ~47%
# of a 500-record sample, with Maya and Sarah next). We append a per-record
# style constraint to the user prompt so each rewrite gets a fresh name +
# scenario. `formal` and `mcq` modes intentionally get no hint — they are
# either textbook restatement or pure question-form, where a cast is wrong.

FIRST_NAMES = [
    # original 47
    "Aarav", "Adaeze", "Aiko", "Alejandro", "Amaru", "Anika", "Bashir",
    "Beatriz", "Cassian", "Chiamaka", "Diego", "Dimitra", "Elif", "Esme",
    "Farhan", "Fatima", "Felix", "Giulia", "Hana", "Ines", "Ivan", "Jasper",
    "Jiang", "Kavya", "Khadija", "Lakshmi", "Lior", "Magnus", "Mateo",
    "Nadia", "Niamh", "Nikola", "Omar", "Priya", "Quynh", "Rafael", "Ravi",
    "Renata", "Salma", "Soren", "Takeshi", "Thandiwe", "Tobias", "Wren",
    "Xiulan", "Yuki", "Zara",
    # added pool
    "Adaora", "Aditi", "Akira", "Amir", "Anders", "Anouk", "Arjun", "Asha",
    "Astrid", "Aurora", "Bilal", "Bjorn", "Camila", "Catalina", "Celeste",
    "Chinedu", "Demetri", "Devansh", "Eitan", "Eira", "Emeka", "Esperanza",
    "Estela", "Folake", "Genevieve", "Henrik", "Hiroshi", "Ifeoma", "Imani",
    "Imogen", "Ishaan", "Jelani", "Joaquin", "Kaito", "Karim", "Kenji",
    "Kiana", "Kofi", "Krishna", "Layla", "Leilani", "Lev", "Linh", "Lucia",
    "Mahir", "Marta", "Meera", "Mei", "Milena", "Mirembe", "Nasrin",
    "Nia", "Nikhil", "Noor", "Pari", "Petra", "Rashid", "Rocio", "Saanvi",
    "Saoirse", "Selma", "Sora", "Stellan", "Tanvi", "Tariq", "Tendai",
    "Themba", "Tomoko", "Valentina", "Vihaan", "Yara", "Yasmin", "Yusuf",
    "Zaynab", "Zola", "Zuri",
    # bulk pool — kept distinct from the lists above for code review,
    # but indistinguishable to the sampler. Aim ~250 names so per-name
    # repetition stays under 200× even at 50 k story-mode records.
    "Abebe", "Adetola", "Adrian", "Aleksei", "Alessia", "Amani", "Amelie",
    "Amina", "Anaya", "Andrei", "Aneeka", "Angeline", "Antonella", "Aoife",
    "Arman", "Arnav", "Aroha", "Asma", "Atharva", "Aviva", "Aya", "Ayaan",
    "Ayano", "Ayodele", "Bao", "Basir", "Carmen", "Chiara", "Chinwe", "Cira",
    "Clemence", "Cyrus", "Dahlia", "Daniela", "Davinder", "Dilnoza", "Domenico",
    "Edmund", "Elara", "Eloise", "Elliot", "Emil", "Emiliano", "Enrique",
    "Erika", "Esi", "Estefania", "Ezra", "Faina", "Fiona", "Florian", "Galina",
    "Gemma", "Hailey", "Hamid", "Haruki", "Hassan", "Helena", "Hugo", "Idris",
    "Inara", "Iona", "Isadora", "Itzel", "Jamila", "Janneke", "Jelena",
    "Jorgen", "Josefina", "Junko", "Kaida", "Kaspar", "Keiko", "Kenta",
    "Kgomotso", "Konstantin", "Lakeisha", "Lars", "Leandro", "Lila", "Liyana",
    "Luca", "Maeve", "Manon", "Maral", "Marcelo", "Marisol", "Markus", "Mashal",
    "Matilde", "Maximilian", "Mayuko", "Mira", "Miran", "Misaki", "Mosi",
    "Naima", "Naledi", "Nayeli", "Nilufar", "Nino", "Nuria", "Octavia",
    "Olamide", "Oluwaseun", "Orla", "Padma", "Pavel", "Phong", "Rania", "Reema",
    "Riya", "Romina", "Rumi", "Samar", "Sana", "Sergei", "Shahid", "Shoshana",
    "Sif", "Simba", "Soraya", "Tahir", "Tamiko", "Tarik", "Theo", "Tomas",
    "Ulla", "Vasilisa", "Wei", "Xochitl", "Yael", "Yannis", "Yuna", "Zahara",
    "Zahra",
]

SCENARIOS = [
    # original specialist set (kept for variety)
    "a marine biologist tagging dolphins",
    "a luthier fitting violin strings",
    "a barista calibrating an espresso grinder",
    "a civic planner sizing a bike-share station",
    "a vineyard manager scheduling harvest",
    "an archaeologist cataloguing pottery shards",
    "a glassblower measuring kiln temperatures",
    "a costume designer ordering fabric for a stage production",
    "a chess coach reviewing tournament scores",
    "a beekeeper tracking hive yields",
    "a roastery owner blending coffee beans",
    "a stage manager timing scene changes",
    "a falconer training a young kestrel",
    "a baker proofing sourdough loaves",
    "a sailmaker cutting Dacron panels",
    "a wildlife photographer surveying bird migration",
    "a perfumer formulating a bergamot–cedar blend",
    "a zoo nutritionist balancing feed rations",
    "a railway dispatcher routing freight cars",
    "a roofer ordering slate tiles",
    "a tea merchant grading oolong leaves",
    "a sommelier curating a wine cellar",
    "an auction-house cataloguer pricing prints",
    "a librarian re-shelving a returned cart",
    "a cartographer drafting a trail map",
    "a paramedic stocking an ambulance",
    "a ceramicist firing a batch of stoneware",
    "a dog trainer scheduling agility sessions",
    "a film editor cutting a documentary",
    "a park ranger patrolling a trail loop",
    "a textile designer dyeing wool skeins",
    "a watchmaker servicing pocket watches",
    "a bookbinder stitching a hand-bound journal",
    "a goat-cheese maker rotating wheels in a cave",
    "an ornithologist banding migratory warblers",
    # everyday / relatable settings — these recover naturalness while still
    # being varied; weighted heavier than the specialist set above by sheer
    # count, so the average rewrite leans more grounded.
    "a school principal organising a field trip for two grade levels",
    "a soccer coach planning a team's away-game travel",
    "a delivery driver routing afternoon stops across the city",
    "a parent packing lunches for the school week",
    "a homeowner pricing materials for a kitchen renovation",
    "a renter splitting groceries with two roommates",
    "a high-school student building a science-fair display",
    "a small bakery owner ordering flour for the month",
    "a community gardener planting a row of tomatoes",
    "a contractor pouring concrete for a driveway",
    "a fitness instructor scheduling weekly studio classes",
    "a nurse stocking medication carts on a hospital ward",
    "a family planning a road trip across three states",
    "a runner training for a half marathon",
    "a bookshop owner reordering bestsellers",
    "a small-business accountant balancing the monthly books",
    "a music teacher arranging a student recital programme",
    "a pet-sitter walking three dogs on a single route",
    "a city bus driver tracking weekly fuel costs",
    "a hospital administrator scheduling on-call shifts",
    "a tutor preparing a maths review session",
    "a wedding planner seating guests at round tables",
    "a college freshman budgeting a semester's expenses",
    "a roommate splitting a utility bill four ways",
    "a hiker mapping out a multi-day backpacking route",
    "a video creator editing a series of cooking tutorials",
    "a yoga-studio owner rotating weekly schedules",
    "a startup founder estimating runway from monthly burn",
    "a substitute teacher covering classes across grades",
    "a farmers'-market vendor selling apples by the pound",
    "a tour guide planning a daily city itinerary",
    "a junior accountant reconciling expense receipts",
    "a software developer benchmarking a new release",
    "a museum docent timing tour groups through galleries",
    "a non-profit volunteer sorting donated clothes",
    "a board-game designer playtesting a card mechanic",
    "an apartment manager scheduling maintenance visits",
    "a swim coach timing relay practices at the pool",
    "a Sunday brunch host serving family-style portions",
    "a home baker scaling a recipe for a bake sale",
    "a youth-orchestra librarian distributing sheet music",
    "a high-school robotics team building a competition robot",
    "a community-centre director planning a summer camp",
    "a freelance translator estimating a project's word count",
    "a personal trainer designing a four-week strength block",
    "a parents' association raising funds for a playground",
    "a barber shop owner scheduling Saturday appointments",
    "a thrift-store manager pricing the week's intake",
    "a food-bank coordinator packing weekly grocery boxes",
    "a Scout troop leader planning a weekend camping trip",
    # bulk pool — additional everyday + niche occupations to push total
    # toward ~200, so per-scenario repetition stays manageable at 30 k+.
    # — workplace / office —
    "a junior auditor reconciling vendor expense reports",
    "a city clerk filing weekly zoning permits",
    "an insurance adjuster reviewing storm-damage claims",
    "a grant writer drafting a foundation proposal",
    "a real-estate agent staging weekend open houses",
    "a UX researcher running usability sessions for a mobile app",
    "a recruiter scheduling candidate interview loops",
    "a payroll administrator running a biweekly cycle",
    "a benefits coordinator enrolling new hires for open season",
    "a court stenographer transcribing testimony",
    # — food —
    "a school-cafeteria nutritionist planning a weekly menu",
    "a catering manager portioning a wedding buffet",
    "a sushi chef ordering bluefin loins",
    "a fishmonger pricing the morning catch",
    "a food-truck owner restocking ingredients between shifts",
    "a chocolatier tempering dark chocolate for a Valentine's run",
    "a cheesemonger cutting truckle wheels for a tasting flight",
    "a cocktail-bar manager tracking pour costs",
    "a charcuterie-board maker assembling a holiday-party order",
    "a brewery brewmaster mashing in a pale-ale batch",
    # — nature / outdoors —
    "a national-park biologist counting elk herds at dawn",
    "a fly-fishing guide planning a week of guided trips",
    "a mountaineering instructor leading a glacier-skills course",
    "a kayak-tour guide running a coastal day trip",
    "a tide-pool researcher inventorying intertidal species",
    "a forester scheduling a controlled burn",
    "a stable manager rotating pasture for dairy cattle",
    "a saltwater-aquaculture technician feeding farmed salmon",
    "a beach lifeguard logging rip-current incidents",
    "a vineyard sommelier blending a barrel sample",
    # — arts / crafts —
    "a printmaker editioning a linocut",
    "a jeweller setting a row of pavé diamonds",
    "a stained-glass artist cutting a rose-window panel",
    "a tattoo artist scheduling a sleeve across multiple sittings",
    "a metalworker forging a pattern-welded blade",
    "a saxophone repair tech overhauling a vintage horn",
    "a kite-maker stitching ripstop nylon panels",
    "an illustrator inking a children's-book spread",
    "a costume seamster fitting a period-drama costume",
    "a piano tuner tempering a concert grand",
    # — sports / fitness —
    "a marathon coach building a periodised training plan",
    "a CrossFit programmer designing the week's workouts",
    "a tennis stringer restringing a tour pro's racquets",
    "a cycling-team mechanic prepping bikes for a stage race",
    "a golf-course superintendent planning aeration cycles",
    "a downhill-ski coach scheduling slalom drills",
    "a high-school wrestling coach managing a weight-cut week",
    "a basketball referee scheduling weekend tournament games",
    "a sailing instructor logging student progressions",
    "a triathlon-club organiser staging a sprint event",
    # — transport / travel —
    "a freight forwarder consolidating an LCL container",
    "a charter-flight dispatcher filing a flight plan",
    "a tugboat captain timing a harbour shift",
    "a customs broker classifying an inbound shipment",
    "a travel agent assembling a multi-city itinerary",
    "a long-haul trucker logging hours under HOS rules",
    "a cruise-ship excursion planner timing port calls",
    "a school-bus dispatcher routing morning pickups",
    "a parking-garage manager pricing event-day rates",
    "a rental-car agent reconciling fleet returns",
    # — household / community —
    "a home brewer planning a five-gallon batch",
    "a CSA-farm manager packing weekly subscriber boxes",
    "a yarn-store owner pricing a new fibre line",
    "a quilting-guild leader assigning blocks for a community quilt",
    "a crisis-line volunteer logging Saturday-night calls",
    "a city-council interpreter translating a public meeting",
    "a homeowners' association treasurer reconciling dues",
    "a community-pool manager scheduling lifeguard shifts",
    "a local-paper editor laying out the weekend edition",
    "a polling-place coordinator stocking a precinct",
    # — science / data —
    "a citizen scientist logging back-yard bird counts",
    "a museum conservator stabilising a 19th-century oil painting",
    "a weather-balloon technician launching the morning sounding",
    "a planetarium operator scripting a constellation show",
    "a forensic accountant tracing a trail of invoices",
    "a survey crew running a property-line traverse",
    "a soil scientist sampling a vineyard's terroir",
    "a data-centre operator capacity-planning the next quarter",
    "a clinical-research coordinator screening trial candidates",
    "a librarian cataloguing a special-collections donation",
    # — services —
    "a fire-department captain rotating four shifts",
    "a hospital pharmacy tech filling prescriptions on a ward",
    "a hospice nurse scheduling weekly home visits",
    "a dental hygienist cleaning a packed Wednesday schedule",
    "a veterinary surgeon booking spay/neuter days",
    "a bike-share mechanic rebalancing a station overnight",
    "a public-transit planner reviewing weekday ridership",
    "a 911 dispatcher logging a busy summer shift",
    "an EMT restocking trauma kits between calls",
    "a wildfire-crew foreman cutting fireline ahead of a blaze",
    # — teaching / learning —
    "a kindergarten teacher organising centres for the day",
    "a high-school physics teacher building a pendulum lab",
    "a college TA grading a stack of problem sets",
    "a community-college dean planning a semester's course schedule",
    "a private-school admissions officer reviewing applications",
    "a music conservatory secretary scheduling juries",
    "a daycare provider planning weekly snacks",
    "a piano teacher arranging a winter recital order",
    "an SAT-prep tutor designing a six-week curriculum",
    "a writing-centre coordinator staffing tutoring hours",
    # — media / events —
    "a small-press publisher pricing a debut paperback run",
    "a podcast producer scheduling a season's interview slots",
    "a synagogue caretaker preparing a holiday meal",
    "a wedding officiant blocking out a season of ceremonies",
    "a community-radio DJ rotating playlists across timeslots",
]

ENGINEERING_DOMAINS = [
    # — engineering disciplines —
    "structural engineering on a pedestrian bridge",
    "electrical engineering on a solar inverter",
    "thermal engineering on a heat exchanger",
    "fluid mechanics on a wind-tunnel experiment",
    "materials science on a composite layup",
    "control systems on a quadcopter stabiliser",
    "RF engineering on a satellite uplink",
    "biomedical engineering on a dialysis pump",
    "aerospace engineering on a turbofan blade",
    "chemical engineering on a distillation column",
    "geotechnical engineering on a retaining wall",
    "acoustic engineering on a concert-hall renovation",
    "automotive engineering on a regenerative brake",
    "naval architecture on a hydrofoil hull",
    "robotics engineering on a pick-and-place arm",
    "power-systems engineering on a grid-tied battery",
    "optical engineering on a telescope mirror",
    "environmental engineering on a stormwater basin",
    "petroleum engineering on a wellhead choke",
    "nuclear engineering on a reactor coolant loop",
    "mechanical engineering on a planetary gearbox",
    "industrial engineering on an assembly-line throughput study",
    "manufacturing engineering on a CNC tolerance audit",
    "mining engineering on a haul-truck cycle-time analysis",
    "mechatronics engineering on a six-axis welding cell",
    "metallurgical engineering on a steel-quenching schedule",
    "agricultural engineering on a centre-pivot irrigation system",
    "transportation engineering on a freeway-interchange traffic study",
    "hydraulic engineering on a dam spillway",
    "bioprocess engineering on a fermenter scale-up",
    "food engineering on a continuous-flow pasteuriser",
    "pharmaceutical engineering on a tablet-press throughput target",
    "semiconductor engineering on a photolithography stepper",
    "MEMS engineering on a capacitive accelerometer",
    "photonics engineering on a fibre-optic transceiver",
    "network engineering on a campus-backbone capacity plan",
    "packaging engineering on a corrugated shipping crate",
    "forensic engineering on a fatigue-failure analysis",
    "tissue engineering on a vascularised scaffold",
    "wind engineering on a turbine-blade load case",
    "marine engineering on a propeller cavitation study",
    "rail engineering on a high-speed track alignment",
    "geomatic engineering on a LiDAR survey of a quarry",
    # — physics subfields —
    "astrophysics on a stellar light curve",
    "plasma physics on a tokamak diverter coil",
    "atomic physics on a rubidium magneto-optical trap",
    "condensed-matter physics on a superconducting wire",
    "particle physics on a calorimeter calibration",
    "biophysics on a single-molecule force-extension curve",
    "geophysics on a seismic-survey grid",
    "atmospheric physics on a hurricane simulation",
    "ocean physics on a thermohaline-circulation profile",
    "gravitational physics on a LIGO mirror suspension",
    "medical physics on a radiotherapy beam calibration",
    "quantum optics on a single-photon detector",
    "nuclear physics on a half-life measurement",
    "statistical mechanics on a polymer-chain Monte-Carlo run",
    "solid-state physics on a piezoelectric stack",
    "laser physics on a Fabry–Pérot cavity alignment",
    "thermodynamics on a Stirling-engine cycle",
    "electromagnetism on a coilgun acceleration stage",
    "high-energy physics on a beam-loss monitor",
    "fluid dynamics on a Reynolds-number transition study",
    # — applied sciences with quantitative contexts —
    "meteorology on a numerical weather-prediction grid",
    "climatology on a regional sea-level-rise model",
    "seismology on an earthquake P-wave arrival",
    "oceanography on a Gulf Stream current profile",
    "volcanology on a gas-flux monitoring station",
    "glaciology on an ice-core melt-layer count",
    "hydrology on a river-discharge gauge",
    "limnology on a lake-stratification temperature profile",
    "ecology on a salmon-population dynamics model",
    "epidemiology on a respiratory-virus transmission model",
    "pharmacokinetics on a drug-clearance curve",
    "neuroscience on an action-potential firing-rate analysis",
    "computational chemistry on a reaction-rate calculation",
    "crystallography on a lattice-spacing refinement",
    "biomechanics on a gait-analysis platform",
    "geomorphology on a coastal-erosion rate study",
    "remote sensing on a NDVI vegetation index",
    "metrology on a coordinate-measuring-machine calibration",
    "computational fluid dynamics on a heat-sink airflow simulation",
    "data engineering on a streaming-pipeline throughput benchmark",
    "cryogenics on a liquid-helium boil-off measurement",
    "nanotechnology on a graphene sheet conductivity test",
]

_STORY_MODES = {"word_problem", "narrative", "hidden_operation", "red_herring"}
_ENGINEERING_MODES = {"engineering_physics"}

# Modes that don't try to wrap the math in a real-world setting. Used as
# a fallback for categories whose math is pure symbol manipulation.
_NON_NARRATIVE_MODES = ("formal", "mcq")


# Categories whose math is pure symbol manipulation — there is no
# physical referent for the symbols involved (a trig identity is not a
# "fluctuation"; sin/cos isn't a "spending model"). Empirically, when
# Gemma 4 is forced to embed these in a story or engineering scenario,
# it pastes the symbolic expression verbatim into the prose ("the
# variance factor equals cos(2x)·tan(2x)") and the judge knocks
# naturalness to 2. Restricting these seeds to formal + mcq removes
# the failure mode entirely. See thinking-trace audit on run #3.
STORY_INCOMPATIBLE_CATEGORIES = frozenset({
    "trig_identity",
    "latex_simplify_trig",
    "latex_simplify_rational",
    "latex_simplify_partial_fractions",
    "latex_simplify_polynomial",
    "latex_simplify_factor",
})


# Categories where the rewriter benefits from a thinking pass. Either
# symbolic (LaTeX-heavy outputs the rewriter has to reproduce verbatim)
# or multi-step (the model must verify its arithmetic before committing
# to a final answer). Empirically these are the categories with sub-75%
# keep-rate on Gemma 4 31B without thinking.
THINKING_CATEGORIES = frozenset({
    # calculus
    "derivative_basic", "derivative_poly",
    "integral_definite", "integral_indefinite", "integral_indefinite_sympy",
    "limit", "ode_separable", "taylor_series", "maclaurin",
    # LaTeX simplifiers
    "latex_simplify_trig", "latex_simplify_rational",
    "latex_simplify_partial_fractions", "latex_simplify_polynomial",
    "latex_simplify_factor",
    # trigonometry
    "trig_identity", "trig_equation", "trig_inverse", "trig_exact",
    # linear algebra
    "matrix_multiply", "matrix_add", "matrix_transpose", "matrix_inverse",
    "determinant_2x2", "eigenvalue_2x2",
    "dot_product", "vector_scale", "vector_add",
    # algebraic equations
    "quadratic", "cubic_roots", "system_2x2", "system_3x3",
    "linear_equation", "inequality_linear",
    # functions
    "function_domain", "function_composition", "function_inverse",
    # complex / number theory
    "complex_arithmetic", "complex_polar", "complex_modulus",
    "extended_euclidean", "chinese_remainder", "fermat", "modular_inverse",
    "euler_totient", "prime_factorization",
    # boolean / probability
    "boolean_simplify", "boolean_truth_table", "sat_witness",
    "prob_conditional", "inclusion_exclusion",
    # tricks / adversarial
    "trick_natural_zero", "trick_lin_inconsistent", "trick_abs_negative",
    "adversarial_sqrt_x2", "adversarial_div_zero",
})


def wants_thinking(seed_meta: dict, mode: str, policy: str) -> bool:
    """Should we enable Gemma 4's thinking mode for this record?

    policy: 'off' | 'auto' | 'on'
      off  — never (default; thinking adds latency + tokens).
      auto — curated category set, plus a LaTeX-answer heuristic.
      on   — always, except 'formal' / 'mcq' where the output is too
             short for a thinking pass to pay back its cost.
    """
    if policy == "off":
        return False
    if mode in ("formal", "mcq"):
        return False
    if policy == "on":
        return True
    cat = (seed_meta or {}).get("category", "")
    return cat in THINKING_CATEGORIES


def diversity_hint(prompt_name: str, rng: random.Random) -> str:
    """Style constraint appended to the user prompt to force a fresh
    name + scenario per record. Empty for modes where it would fight
    the prompt's intent (formal restatement, MCQ shell)."""
    if prompt_name in _STORY_MODES:
        n = rng.choice([1, 1, 2])
        names = rng.sample(FIRST_NAMES, n)
        scenario = rng.choice(SCENARIOS)
        if n == 1:
            who = f"the protagonist named {names[0]}"
        else:
            who = f"two characters named {names[0]} and {names[1]}"
        return (
            "\n\nSTYLE CONSTRAINT (must be obeyed):\n"
            f"- Set the scene around {scenario}.\n"
            f"- Use {who}. Do not invent different names.\n"
            "- Do not reuse stock names like Leo, Maya, Sarah, Alice, Bob."
        )
    if prompt_name in _ENGINEERING_MODES:
        domain = rng.choice(ENGINEERING_DOMAINS)
        return (
            "\n\nSTYLE CONSTRAINT (must be obeyed):\n"
            f"- Frame the problem inside {domain}; stay specific to that "
            "subfield rather than writing a generic 'engineer' scenario."
        )
    return ""


# ─────────────────────────── PROMPTS ───────────────────────────

# ─────────────────────────────────────────────────────────────────
# Multilingual extension (added 2026-05-06)
# Each rewrite picks one of these languages; system prompt is wrapped with
# a language instruction. Output records carry `rewrite_language`.
# ─────────────────────────────────────────────────────────────────
LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "ru": "Russian",
    "uk": "Ukrainian",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "hi": "Hindi",
    "sv": "Swedish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    "da": "Danish",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "ca": "Catalan",
}


def _language_preamble(lang_code: str) -> str:
    """Returns a system-prompt suffix that pins the rewrite to one language.
    For 'en' returns empty (no change vs original behaviour).

    Mathematical notation, numbers and 'Answer:' marker stay in standard
    ASCII/LaTeX regardless of language."""
    if lang_code == "en":
        return ""
    name = LANGUAGES.get(lang_code, lang_code)
    return (
        f"\n\nIMPORTANT — LANGUAGE INSTRUCTION:\n"
        f"Write the rewritten problem AND the step-by-step solution "
        f"entirely in {name}. The text must be idiomatic {name}, not a "
        f"word-for-word translation.\n"
        f"Keep mathematical notation in standard ASCII/LaTeX (e.g. x^2, "
        f"\\frac{{a}}{{b}}, =, +) regardless of language. Numbers stay "
        f"as digits. Keep the literal token 'Answer:' in {name} "
        f"(use the natural translation, e.g. 'Réponse:' / 'Antwort:' / "
        f"'Respuesta:' / '答案:' / etc.) followed by the same value as "
        f"the original answer."
    )


REWRITE_PROMPTS = [
    {
        "name": "word_problem",
        "system": (
            "You are a creative math teacher. Your job is to take a bare math "
            "problem and rewrite it as a vivid word problem set in a real-world "
            "context (shopping, travel, sport, science, daily life, …). "
            "The underlying math, the numbers, and the final answer must be "
            "preserved exactly. After the problem, give a clear step-by-step "
            "solution that ends with 'Answer: <value>'."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as a real-world word problem and provide a step-by-step "
            "solution.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<your rewritten word problem>\n\n"
            "SOLUTION:\n<step-by-step reasoning, ending with 'Answer: <value>'>"
        ),
    },
    {
        "name": "red_herring",
        "system": (
            "You are an exam-question writer. Take a math problem and rewrite "
            "it as a real-world word problem that **deliberately contains one "
            "or two extra numerical facts that are NOT needed to solve it** — "
            "students must identify which facts matter. Same numbers and same "
            "final answer for the operative facts; same arithmetic. In the "
            "solution, explicitly point out which fact(s) are red herrings."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as a word problem with 1-2 irrelevant numerical facts "
            "added, then solve it while flagging the red herrings.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<rewritten problem with extra irrelevant info>\n\n"
            "SOLUTION:\n<solution that names the red herrings, then computes; "
            "end with 'Answer: <value>'>"
        ),
    },
    {
        "name": "hidden_operation",
        "system": (
            "You are a creative math teacher. Take a math problem and rewrite "
            "it as a story problem where the reader must figure out which "
            "arithmetic operation (or sequence of operations) to apply. "
            "**Do NOT use the words 'add', 'subtract', 'sum', 'difference', "
            "'multiply', 'product', 'divide', 'quotient', or any direct "
            "synonym in the problem text.** Use a realistic context. Same "
            "numbers and same final answer."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as a story problem with no operation words, then "
            "solve it: the solution should first identify which operation is "
            "needed, then compute the answer.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<story problem with no operation words>\n\n"
            "SOLUTION:\n<solution that first identifies the correct operation, "
            "then computes; end with 'Answer: <value>'>"
        ),
    },
    {
        "name": "engineering_physics",
        "system": (
            "You are an engineer writing practical exercises. Rewrite the "
            "given math problem as an engineering or physics scenario "
            "(forces, voltages, speeds, mass, flow rate, materials, "
            "tolerances, …). Keep the numbers and final answer identical."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as an engineering or physics problem and solve it "
            "step by step.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<engineering / physics version>\n\n"
            "SOLUTION:\n<step-by-step solution; end with 'Answer: <value>'>"
        ),
    },
    {
        "name": "narrative",
        "system": (
            "You are a fiction writer who incorporates math into short "
            "stories. Take the math problem and rewrite it as a 3-5 sentence "
            "narrative involving 2-3 named characters interacting realistically. "
            "The math should arise organically from the story. Same numbers, "
            "same final answer."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as a short narrative problem and solve it.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<narrative problem>\n\n"
            "SOLUTION:\n<step-by-step solution; end with 'Answer: <value>'>"
        ),
    },
    {
        "name": "formal",
        "system": (
            "You are a textbook author. Restate the given math problem in "
            "clean, formal mathematical language — the kind that would appear "
            "in a textbook exercise. No casual phrasing, no story, no real-"
            "world dressing. The math content stays the same."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Restate this problem formally and provide a rigorous solution.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<formal restatement>\n\n"
            "SOLUTION:\n<rigorous step-by-step solution; end with 'Answer: <value>'>"
        ),
    },
    {
        # Sampled rarely (see `weight` below) — most records should remain
        # free-form. The model is instructed to emit the final answer as
        # "Answer: <letter>) <value>" so the judge can still match the
        # numeric ORIGINAL ANSWER without special-casing the format.
        "name": "mcq",
        "weight": 0.4,
        "system": (
            "You are an exam author. Take a math problem and rewrite it as a "
            "4-way multiple-choice question. Generate three plausible "
            "distractors alongside the correct answer (off-by-one, sign flip, "
            "common arithmetic slip, swapped operands — wrong but tempting, "
            "never absurd). Shuffle the four options and label them A, B, C, "
            "D so the correct one is not always in the same slot. The "
            "underlying math, numbers, and correct numeric answer must be "
            "preserved exactly. After the problem, give step-by-step "
            "reasoning that ends with a line of the form "
            "'Answer: <letter>) <value>' so both the chosen letter and the "
            "value it stands for are explicit."
        ),
        "user": (
            "ORIGINAL PROBLEM:\n{problem}\n\n"
            "ORIGINAL ANSWER: {answer}\n"
            "ORIGINAL SOLUTION STEPS: {steps}\n\n"
            "Rewrite this as a 4-way multiple-choice question and solve it.\n\n"
            "Output strictly in this format:\n"
            "PROBLEM:\n<question text>\n"
            "A) <option>\n"
            "B) <option>\n"
            "C) <option>\n"
            "D) <option>\n\n"
            "SOLUTION:\n<step-by-step reasoning, ending with "
            "'Answer: <letter>) <value>'>"
        ),
    },
]


JUDGE_SYSTEM = (
    "You are a meticulous math teacher reviewing a student-written word "
    "problem. Rate the rewrite on four criteria and emit STRICT JSON only "
    "(no commentary outside the JSON object)."
)

JUDGE_USER_TEMPLATE = """ORIGINAL PROBLEM:
{original_problem}
ORIGINAL ANSWER: {original_answer}

REWRITTEN PROBLEM:
{rewritten_problem}

REWRITTEN SOLUTION:
{rewritten_solution}

Rate the rewrite on these criteria:
1. CORRECTNESS  — does the rewritten problem still admit the same final answer? "yes" or "no"
2. NATURALNESS  — does it read like a realistic problem (1-5)
3. CLARITY      — can a competent student tell what is being asked, after ignoring any red-herring info (1-5)
4. DIFFICULTY_PRESERVATION — was the underlying difficulty preserved (or made harder), without becoming trivial (1-5)
5. OVERALL_KEEP — your single yes/no judgement on whether this rewrite is worth keeping in a training set

Output STRICTLY this JSON shape and nothing else:
{{
  "correctness": "yes" or "no",
  "correctness_reason": "<one sentence>",
  "naturalness": <1-5>,
  "clarity": <1-5>,
  "difficulty_preservation": <1-5>,
  "overall_keep": true or false
}}"""


# ─────────────────────────── vLLM CLIENT ───────────────────────────

class VLLMClient:
    def __init__(self, endpoint: str, model: str, *,
                 api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, system: str, user: str, *,
              temperature: float, max_tokens: int, retries: int,
              extra_body: Optional[dict]) -> dict:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = self._session.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]
            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"vLLM request failed after {retries+1} tries: {last_err}")

    def chat(self, system: str, user: str, *,
             temperature: float = 0.8, max_tokens: int = 1024,
             retries: int = 2, extra_body: Optional[dict] = None) -> str:
        msg = self._post(system, user, temperature=temperature,
                         max_tokens=max_tokens, retries=retries,
                         extra_body=extra_body)
        return msg.get("content", "") or ""

    def chat_with_reasoning(self, system: str, user: str, *,
                            temperature: float = 0.8, max_tokens: int = 1024,
                            retries: int = 2,
                            extra_body: Optional[dict] = None
                            ) -> tuple[str, Optional[str]]:
        """Return (content, reasoning). `reasoning` comes from vLLM's
        --reasoning-parser output (Gemma 4 exposes it as message.reasoning
        when the server is launched with --reasoning-parser gemma4)."""
        msg = self._post(system, user, temperature=temperature,
                         max_tokens=max_tokens, retries=retries,
                         extra_body=extra_body)
        return msg.get("content", "") or "", msg.get("reasoning")


# ─────────────────────────── PARSERS ───────────────────────────

_PROBLEM_RE = re.compile(r"PROBLEM\s*:\s*(.+?)(?=\n\s*SOLUTION\s*:|\Z)",
                         re.DOTALL | re.IGNORECASE)
_SOLUTION_RE = re.compile(r"SOLUTION\s*:\s*(.+)", re.DOTALL | re.IGNORECASE)


def parse_rewrite(text: str) -> tuple[Optional[str], Optional[str]]:
    """Pull PROBLEM / SOLUTION sections out of the model's reply."""
    p = _PROBLEM_RE.search(text)
    s = _SOLUTION_RE.search(text)
    p_text = p.group(1).strip() if p else None
    s_text = s.group(1).strip() if s else None
    if not p_text and not s_text:
        # fallback: try splitting on the word "Solution"
        parts = re.split(r"\n\s*solution\s*:?\s*", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return p_text, s_text


def parse_judge(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from the judge response."""
    try:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return None


# ─────────────────────────── ENRICH ONE RECORD ───────────────────────────

def enrich_one(record: dict, client: VLLMClient, *,
               temperature: float, max_tokens: int,
               do_judge: bool, prompt_pool: list[dict],
               rng: random.Random,
               thinking_policy: str = "off",
               thinking_max_tokens: int = 4096,
               languages: list[str] = ["en"]) -> dict:
    seed_problem = record["problem"]
    seed_answer = record["answer"]
    seed_meta = record.get("meta", {})
    seed_steps = seed_meta.get("step_text", "(no step text in seed)")

    # Restrict the pool for purely-symbolic seeds: forcing them into a
    # story or engineering scenario reliably produces "formula = pasted
    # expression" prose that the judge rejects on naturalness.
    cat = seed_meta.get("category", "")
    if cat in STORY_INCOMPATIBLE_CATEGORIES:
        safe = [p for p in prompt_pool if p["name"] in _NON_NARRATIVE_MODES]
        if safe:
            prompt_pool = safe

    weights = [p.get("weight", 1.0) for p in prompt_pool]
    prompt = rng.choices(prompt_pool, weights=weights, k=1)[0]
    user = prompt["user"].format(problem=seed_problem, answer=seed_answer, steps=seed_steps)
    user += diversity_hint(prompt["name"], rng)

    # Pick a language for this record (uniform over `languages`); inject
    # language preamble into the system prompt. For en (default) this is a no-op.
    lang_code = rng.choice(languages)
    system_msg = prompt["system"] + _language_preamble(lang_code)

    use_thinking = wants_thinking(seed_meta, prompt["name"], thinking_policy)
    if use_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
        rewrite_raw, reasoning = client.chat_with_reasoning(
            system_msg, user,
            temperature=temperature,
            max_tokens=max(max_tokens, thinking_max_tokens),
            extra_body=extra_body,
        )
    else:
        rewrite_raw = client.chat(system_msg, user,
                                  temperature=temperature, max_tokens=max_tokens)
        reasoning = None
    new_problem, new_solution = parse_rewrite(rewrite_raw)

    out = {
        "seed_id": record.get("id"),
        "seed_problem": seed_problem,
        "seed_answer": seed_answer,
        "seed_meta": seed_meta,
        "rewrite_prompt_name": prompt["name"],
        "rewrite_temperature": temperature,
        "rewrite_thinking_used": use_thinking,
        "rewrite_language": lang_code,
        "rewritten_problem": new_problem,
        "rewritten_solution": new_solution,
        "rewrite_raw": rewrite_raw,
    }
    if reasoning:
        out["rewrite_reasoning"] = reasoning

    if do_judge and new_problem and new_solution:
        judge_user = JUDGE_USER_TEMPLATE.format(
            original_problem=seed_problem,
            original_answer=seed_answer,
            rewritten_problem=new_problem,
            rewritten_solution=new_solution,
        )
        judge_raw = client.chat(JUDGE_SYSTEM, judge_user,
                                temperature=0.1, max_tokens=512)
        out["judge_raw"] = judge_raw
        out["judge"] = parse_judge(judge_raw)

    return out


# ─────────────────────────── CLI ───────────────────────────

def main():
    # Pre-parse only --env-file so the env is populated before we read defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file:
        n_set = load_env_file(pre_args.env_file)
        print(f"Loaded {n_set} variable(s) from {pre_args.env_file}", file=sys.stderr)

    # Now read env defaults (project-specific names first, then OPENAI_*).
    env_endpoint = _env_first("MATH_GEN_VLLM_ENDPOINT", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                              default=DEFAULT_ENDPOINT)
    env_model = _env_first("MATH_GEN_VLLM_MODEL", "OPENAI_MODEL", default=None)
    env_api_key = _env_first("MATH_GEN_VLLM_API_KEY", "OPENAI_API_KEY", default=None)
    env_timeout = int(os.environ.get("MATH_GEN_VLLM_TIMEOUT", DEFAULT_TIMEOUT))

    ap = argparse.ArgumentParser(
        description="Rewrite math problems via vLLM and (optionally) judge them. "
                    "All connection settings can be supplied via env vars; "
                    "see the module docstring for the full list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="JSONL produced by `python -m math_generator`")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--env-file", default=None,
                    help="path to a .env-style file (KEY=value lines). Existing env wins.")
    ap.add_argument("--endpoint", default=env_endpoint,
                    help=f"vLLM /v1 base URL (default from env or '{DEFAULT_ENDPOINT}')")
    ap.add_argument("--model", default=env_model,
                    help="model name as served by vLLM (env: MATH_GEN_VLLM_MODEL / OPENAI_MODEL)")
    ap.add_argument("--api-key", default=env_api_key,
                    help="bearer token for vLLM (env: MATH_GEN_VLLM_API_KEY / OPENAI_API_KEY)")
    ap.add_argument("--timeout", type=int, default=env_timeout,
                    help="per-request timeout in seconds (env: MATH_GEN_VLLM_TIMEOUT)")
    ap.add_argument("--n", type=int, default=0, help="max records to process (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="temperature for the rewrite call")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the judge pass (faster, half as many calls)")
    ap.add_argument("--prompts", default="all",
                    help=f"comma-separated prompt names; 'all' for everything. "
                         f"Available: {', '.join(p['name'] for p in REWRITE_PROMPTS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--thinking", choices=["off", "auto", "on"], default="off",
                    help="Gemma 4 thinking mode. off=never (default). "
                         "auto=enable for curated symbolic/multi-step categories "
                         "(see THINKING_CATEGORIES in enrich.py). "
                         "on=enable for everything except formal/mcq. "
                         "Requires the vLLM server to be started with "
                         "--reasoning-parser gemma4.")
    ap.add_argument("--languages", default="en",
                    help=("Comma-separated language codes for the rewrite output. "
                          "Each record picks one uniformly at random. Default 'en' "
                          "preserves original behaviour. See LANGUAGES dict in "
                          "enrich.py for the supported set (en, fr, de, es, it, pt, "
                          "nl, pl, cs, ru, uk, zh, ja, ko, ar, tr, hi, …). "
                          "When non-en, system prompt is appended with a language "
                          "instruction; mathematical notation stays ASCII/LaTeX."))
    ap.add_argument("--thinking-max-tokens", type=int, default=4096,
                    help="max_tokens override when thinking is enabled "
                         "(thinking eats budget; default 4096).")
    args = ap.parse_args()

    if not args.model:
        ap.error("model is required: pass --model, or set MATH_GEN_VLLM_MODEL "
                 "(or OPENAI_MODEL) in the environment.")

    # Pick prompt pool
    if args.prompts == "all":
        prompt_pool = REWRITE_PROMPTS
    else:
        wanted = {p.strip() for p in args.prompts.split(",")}
        prompt_pool = [p for p in REWRITE_PROMPTS if p["name"] in wanted]
        if not prompt_pool:
            sys.exit(f"No matching prompts in --prompts={args.prompts}; "
                     f"available: {', '.join(p['name'] for p in REWRITE_PROMPTS)}")

    with open(args.input) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.n and args.n < len(records):
        rng_outer = random.Random(args.seed)
        records = rng_outer.sample(records, args.n)

    client = VLLMClient(args.endpoint, args.model,
                        api_key=args.api_key, timeout=args.timeout)

    if not args.quiet:
        auth_note = " (with API key)" if args.api_key else ""
        print(f"Enriching {len(records)} records via {args.endpoint}{auth_note} "
              f"(model={args.model}, workers={args.workers}, "
              f"prompts={[p['name'] for p in prompt_pool]}, judge={not args.no_judge}, "
              f"thinking={args.thinking})")

    rng = random.Random(args.seed)

    written = 0
    failed = 0
    kept = 0
    t0 = time.time()
    with open(args.output, "w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(enrich_one, rec, client,
                      temperature=args.temperature, max_tokens=args.max_tokens,
                      do_judge=not args.no_judge, prompt_pool=prompt_pool,
                      rng=random.Random(rng.randrange(1 << 31)),
                      thinking_policy=args.thinking,
                      thinking_max_tokens=args.thinking_max_tokens,
                      languages=[c.strip() for c in args.languages.split(",") if c.strip()]): i
            for i, rec in enumerate(records)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                enriched = fut.result()
            except Exception as e:
                failed += 1
                if not args.quiet:
                    print(f"  [{i}] FAILED: {type(e).__name__}: {e}")
                continue
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            j = enriched.get("judge") or {}
            if j.get("overall_keep"):
                kept += 1
            if not args.quiet and written % 10 == 0:
                rate = written / max(time.time() - t0, 1e-3)
                print(f"  [{written}/{len(records)}] kept={kept} failed={failed} "
                      f"({rate:.2f} rec/s)")

    elapsed = time.time() - t0
    print(f"Wrote {written} records to {args.output} "
          f"(kept={kept}, failed={failed}, elapsed={elapsed:.1f}s).")


if __name__ == "__main__":
    main()
