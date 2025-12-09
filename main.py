# main.py
import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from openai import OpenAI

import json
import re

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TMDB_API_KEY = os.environ["TMDB_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

## app = FastAPI(title="Movie Recommender AI Backend")


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Allow your Next.js frontend to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------

class UserPreferenceInput(BaseModel):
    user_id: Optional[str] = None
    free_text: str
    must_include_genres: List[str] = []
    must_exclude_genres: List[str] = []
    watching_with: Optional[str] = None  # solo / family / date / friends
    max_runtime_minutes: Optional[int] = None


class PreferenceProfile(BaseModel):
    target_genres: List[str]
    avoid_genres: List[str]
    tone: str  # comforting | intense | emotional | silly
    year_range: List[int]  # [start, end]
    language: str
    violence_tolerance: str  # low | medium | high


class SearchStrategy(BaseModel):
    tmdb_genre_ids: List[int]
    min_vote_average: float
    min_vote_count: int
    year_from: int
    year_to: int
    include_adult: bool


class TmdbMovie(BaseModel):
    id: int
    title: str
    overview: str
    release_date: str
    vote_average: float
    vote_count: int
    genre_ids: List[int]
    original_language: str


class MovieScore(BaseModel):
    movie_id: int
    score: float
    reasons: str
    disqualified: bool


class PlaylistResult(BaseModel):
    main_picks: List[Dict[str, Any]]
    wildcard_picks: List[Dict[str, Any]]
    notes: str


class RecommendResponse(BaseModel):
    preferences: PreferenceProfile
    strategy: SearchStrategy
    candidates_considered: int
    movie_scores: List[MovieScore]
    playlist: PlaylistResult


# ---------- Utility: OpenAI JSON call ----------

from typing import Any, Dict

def call_openai_json(system: str, user: str, schema: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Call OpenAI and get back a JSON object.

    We use response_format="json_object" and explicitly tell the model
    to return ONLY a JSON object. Includes a fallback JSON extractor.
    """

    # Make sure the system message literally mentions JSON, to satisfy the API requirement
    system_with_hint = (
        system
        + "\n\nIMPORTANT: You MUST respond with a single valid JSON object only. "
          "Do not include any explanation or text outside the JSON. The response must be valid JSON."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_with_hint},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content

    # First attempt: direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to salvage any JSON-looking {...} substring
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            return json.loads(match.group(0))
        # If still broken, raise a clearer error so we can see what the model sent
        raise ValueError(f"Model returned non-JSON content: {content!r}")

# ---------- Agent 1: Preference Profiler ----------

def agent_preference_profiler(inp: UserPreferenceInput) -> PreferenceProfile:
    system = (
        "You are an assistant that converts messy movie preferences into a clean profile.\n"
        "You MUST respond with a single JSON object only.\n"
        "Fields:\n"
        "- target_genres: array of strings (genres the user wants, e.g. ['comedy','romance'])\n"
        "- avoid_genres: array of strings (genres to avoid, e.g. ['horror'])\n"
        "- tone: string (one of 'comforting','intense','emotional','silly','neutral')\n"
        "- year_range: array of two integers [start_year, end_year]\n"
        "- language: string (primary language like 'English')\n"
        "- violence_tolerance: string ('low','medium','high')\n\n"
        "Return ONLY a JSON object with exactly those keys. No extra text."
    )

    # schema is currently ignored by call_openai_json, but we keep the param signature
    schema = None

    user = (
        f"User description: {inp.free_text}\n"
        f"Must include genres (from UI): {inp.must_include_genres}\n"
        f"Must exclude genres (from UI): {inp.must_exclude_genres}\n"
        f"Watching with: {inp.watching_with}\n"
        f"Max runtime: {inp.max_runtime_minutes}\n"
        "Infer a reasonable year range like [2010,2025] etc.\n"
        "If something is not specified, make a reasonable guess."
    )

    raw = call_openai_json(system, user, schema)

    # ---- Normalization: make sure raw matches PreferenceProfile fields ----

    def ensure_list_str(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    def ensure_str(value, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, list):
            # take the first or join
            return " ".join(str(v) for v in value)
        return str(value)

    def ensure_year_range(value) -> list[int]:
        if isinstance(value, list) and len(value) == 2:
            try:
                return [int(value[0]), int(value[1])]
            except Exception:
                pass
        # fallback default
        return [2010, 2025]

    # The model might use different keys; be defensive.
    normalized = {
        "target_genres": ensure_list_str(
            raw.get("target_genres") or raw.get("genres") or raw.get("include_genres")
        ),
        "avoid_genres": ensure_list_str(
            raw.get("avoid_genres") or raw.get("exclude_genres") or raw.get("blocked_genres")
        ),
        "tone": ensure_str(raw.get("tone"), "neutral"),
        "year_range": ensure_year_range(raw.get("year_range")),
        "language": ensure_str(raw.get("language"), "English"),
        "violence_tolerance": ensure_str(raw.get("violence_tolerance"), "medium"),
    }

    return PreferenceProfile(**normalized)

# ---------- Agent 2: Search Strategy Planner ----------

GENRE_MAP = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science fiction": 878,
    "tv movie": 10770,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}


def agent_search_strategy(pref: PreferenceProfile) -> SearchStrategy:
    system = (
        "You are a movie search strategy planner.\n"
        "You MUST respond with a single JSON object only.\n\n"
        "Your goal is to design parameters for calling the TMDB discover API, but your output JSON\n"
        "must use these exact fields:\n"
        "- tmdb_genre_ids: array of integers (TMDB genre IDs to include)\n"
        "- min_vote_average: number (e.g. 6.8)\n"
        "- min_vote_count: integer (e.g. 200 or 500)\n"
        "- year_from: integer (start year)\n"
        "- year_to: integer (end year)\n"
        "- include_adult: boolean\n\n"
        "Return ONLY a JSON object with exactly those keys.\n"
    )

    # We keep schema param in signature but don't rely on it
    schema = None

    user = (
        "Preference profile:\n"
        f"{pref.model_dump_json(indent=2)}\n\n"
        "Here is a mapping from genre names to TMDB IDs:\n"
        f"{GENRE_MAP}\n\n"
        "Pick relevant TMDB genre IDs based on target_genres.\n"
        "Use min_vote_average around 6.8–7.3 for most cases.\n"
        "Use higher min_vote_average for 'comforting' tone / family-friendly movies.\n"
        "Use min_vote_count at least 100 to avoid obscure movies.\n"
        "Year range should roughly match the user's year_range but you can slightly widen it.\n"
        "Set include_adult = false unless the user clearly wants adult content.\n"
    )

    raw = call_openai_json(system, user, schema)

    # ---- Normalization: map whatever we got into SearchStrategy fields ----

    def ensure_int(value, default: int) -> int:
        try:
            if isinstance(value, (int, float, str)):
                return int(value)
        except Exception:
            pass
        return default

    def ensure_float(value, default: float) -> float:
        try:
            if isinstance(value, (int, float, str)):
                return float(value)
        except Exception:
            pass
        return default

    def ensure_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in ["true", "yes", "1"]:
                return True
            if lower in ["false", "no", "0"]:
                return False
        return default

    def ensure_genre_ids(value) -> list[int]:
        # If already a list of ints
        if isinstance(value, list):
            cleaned = []
            for v in value:
                try:
                    cleaned.append(int(v))
                except Exception:
                    pass
            if cleaned:
                return cleaned

        # If a single int
        if isinstance(value, int):
            return [value]

        # If it's a list of strings (genre names), map via GENRE_MAP
        if isinstance(value, list):
            ids = []
            for v in value:
                name = str(v).lower()
                if name in GENRE_MAP:
                    ids.append(GENRE_MAP[name])
            if ids:
                return ids

        # Fallback: use genres from the preference profile
        ids: list[int] = []
        for g in pref.target_genres:
            gid = GENRE_MAP.get(g.lower())
            if gid:
                ids.append(gid)
        # If still none, default to something generic like drama + comedy
        if not ids:
            ids = [GENRE_MAP["drama"], GENRE_MAP["comedy"]]
        return ids

    # Year range: prefer the model's suggestion if present, otherwise fallback to pref.year_range
    def derive_years(r: dict) -> tuple[int, int]:
        # Model might give year_from/year_to directly
        yf = r.get("year_from")
        yt = r.get("year_to")

        # Or primary_release_date.gte/lte
        gte = r.get("primary_release_date.gte") or r.get("release_date_from")
        lte = r.get("primary_release_date.lte") or r.get("release_date_to")

        start = None
        end = None

        if yf is not None and yt is not None:
            start = ensure_int(yf, pref.year_range[0])
            end = ensure_int(yt, pref.year_range[1])
        else:
            # Parse from date strings like '2010-01-01'
            if isinstance(gte, str) and len(gte) >= 4:
                try:
                    start = int(gte[:4])
                except Exception:
                    pass
            if isinstance(lte, str) and len(lte) >= 4:
                try:
                    end = int(lte[:4])
                except Exception:
                    pass

        # fallback to pref if still None
        if start is None:
            start = pref.year_range[0]
        if end is None:
            end = pref.year_range[1]

        # sanity: ensure start <= end
        if start > end:
            start, end = end, start

        return start, end

    # Try to get something sensible out of whatever the model sent
    genre_source = (
        raw.get("tmdb_genre_ids")
        or raw.get("genre_ids")
        or raw.get("with_genres")
        or raw.get("genres")
    )

    min_vote_avg_source = raw.get("min_vote_average") or raw.get("vote_average.gte")
    min_vote_count_source = raw.get("min_vote_count") or raw.get("vote_count.gte")

    year_from, year_to = derive_years(raw)

    include_adult_source = raw.get("include_adult")
    # Maybe the model used TMDB-style 'certification.lte' to imply non-adult
    if include_adult_source is None and raw.get("certification.lte"):
        include_adult_source = False

    normalized = {
        "tmdb_genre_ids": ensure_genre_ids(genre_source),
        "min_vote_average": ensure_float(min_vote_avg_source, 6.8),
        "min_vote_count": ensure_int(min_vote_count_source, 100),
        "year_from": year_from,
        "year_to": year_to,
        "include_adult": ensure_bool(include_adult_source, False),
    }

    return SearchStrategy(**normalized)


# ---------- Tool: TMDB Fetch ----------

async def fetch_tmdb_movies(strategy: SearchStrategy) -> List[TmdbMovie]:
    params = {
        "api_key": TMDB_API_KEY,
        "sort_by": "vote_average.desc",
        "with_genres": ",".join(map(str, strategy.tmdb_genre_ids)),
        "vote_average.gte": strategy.min_vote_average,
        "vote_count.gte": strategy.min_vote_count,
        "include_adult": str(strategy.include_adult).lower(),
        "language": "en-US",
        "page": 1,
        "primary_release_date.gte": f"{strategy.year_from}-01-01",
        "primary_release_date.lte": f"{strategy.year_to}-12-31",
    }

    async with httpx.AsyncClient(timeout=10) as client_http:
        res = await client_http.get("https://api.themoviedb.org/3/discover/movie", params=params)
        res.raise_for_status()
        data = res.json()
        results = data.get("results", [])

    return [TmdbMovie(**m) for m in results]


# ---------- Agent 3: Candidate Evaluator / Critic ----------

def agent_candidate_evaluator(
    pref: PreferenceProfile, movies: List[TmdbMovie], avoid_dark_content: bool
) -> List[MovieScore]:
    system = (
        "You are a movie critic AI. Score each movie for how well it matches the user's preferences.\n"
        "You MUST respond with JSON only.\n\n"
        "Output format:\n"
        "- Either a JSON object with key 'scores' containing an array of objects, OR\n"
        "- A JSON array of score objects.\n\n"
        "Each score object must have:\n"
        "- movie_id: integer (TMDB id)\n"
        "- score: number between 0 and 10\n"
        "- reasons: string explanation\n"
        "- disqualified: boolean (true if it violates hard constraints like tone/genre)\n\n"
        "No text outside JSON."
    )

    schema = None  # we don't rely on schema enforcements here

    user = (
        "User preference profile:\n"
        f"{pref.model_dump_json(indent=2)}\n\n"
        f"Avoid dark / violent content? {avoid_dark_content}\n\n"
        "Movies (array of TMDB results):\n"
        f"{[m.model_dump() for m in movies]}\n\n"
        "For each movie, give a score between 0 and 10.\n"
        "Set disqualified = true if it violates hard constraints (tone, genre avoidance, etc.)."
    )

    raw = call_openai_json(system, user, schema)

    # ---- Normalize model output into a list of dicts ----

    # raw can be:
    # - {"scores": [ {...}, {...} ]}
    # - [ {...}, {...} ]
    if isinstance(raw, dict) and "scores" in raw:
        items = raw["scores"]
    elif isinstance(raw, list):
        items = raw
    else:
        # Debug-friendly error
        raise ValueError(f"Unexpected response format from candidate evaluator: {raw!r}")

    if not isinstance(items, list):
        raise ValueError(f"'scores' must be a list, got: {type(items)}")

    normalized_scores: List[MovieScore] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        movie_id = item.get("movie_id")
        score = item.get("score", 0)
        reasons = item.get("reasons", "")
        disqualified = item.get("disqualified", False)

        # Basic sanity: movie_id must be present and valid
        try:
            movie_id_int = int(movie_id)
        except Exception:
            # skip entries without a valid movie_id
            continue

        # Coerce types safely
        try:
            score_float = float(score)
        except Exception:
            score_float = 0.0

        reasons_str = str(reasons)
        disqualified_bool = bool(disqualified)

        normalized_scores.append(
            MovieScore(
                movie_id=movie_id_int,
                score=score_float,
                reasons=reasons_str,
                disqualified=disqualified_bool,
            )
        )

    if not normalized_scores:
        raise ValueError(f"No valid movie scores produced from response: {raw!r}")

    return normalized_scores

# ---------- Agent 4: Playlist / Watch Plan Composer ----------

def agent_playlist_builder(
    movies: List[TmdbMovie], scores: List[MovieScore], pref: PreferenceProfile
) -> PlaylistResult:
    system = (
        "You are a movie night planner AI. "
        "Given scored movies and user preferences, create a watch list with main picks and wildcards."
    )

    schema = {
        "name": "PlaylistResult",
        "schema": {
            "type": "object",
            "properties": {
                "main_picks": {"type": "array", "items": {"type": "object"}},
                "wildcard_picks": {"type": "array", "items": {"type": "object"}},
                "notes": {"type": "string"},
            },
            "required": ["main_picks", "wildcard_picks", "notes"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    # Convert movies+scores into a simplified list
    movie_score_map = {s.movie_id: s for s in scores}
    simplified = []
    for m in movies:
        ms = movie_score_map.get(m.id)
        if not ms:
            continue
        simplified.append(
            {
                "id": m.id,
                "title": m.title,
                "overview": m.overview,
                "release_date": m.release_date,
                "vote_average": m.vote_average,
                "vote_count": m.vote_count,
                "score": ms.score,
                "reasons": ms.reasons,
                "disqualified": ms.disqualified,
            }
        )

    user = (
        "User preference profile:\n"
        f"{pref.model_dump_json(indent=2)}\n\n"
        "Candidate movies with scores:\n"
        f"{simplified}\n\n"
        "Choose ~3–5 main picks that are safest + best fit.\n"
        "Choose 1–3 wildcard picks that are riskier or more unusual.\n"
        "Add brief notes explaining the watch plan."
    )

    raw = call_openai_json(system, user, schema)
    return PlaylistResult(**raw)


# ---------- Main Orchestration Route (with branches) ----------

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(input_data: UserPreferenceInput):
    # Agent 1: preference profile
    pref = agent_preference_profiler(input_data)

    # Branch B: avoid dark content if tone is comforting or violence_tolerance low
    avoid_dark = pref.tone.lower() in ["comforting", "light", "silly"] or pref.violence_tolerance.lower() == "low"

    # Agent 2: search strategy
    strategy = agent_search_strategy(pref)

    # Call TMDB
    movies = await fetch_tmdb_movies(strategy)

    # Branch A: if too few movies, relax strategy
    if len(movies) < 10:
        # Relax min_vote_average and widen year range in code
        strategy.min_vote_average = max(strategy.min_vote_average - 0.5, 5.5)
        strategy.year_from = strategy.year_from - 5
        # Retry TMDB
        movies = await fetch_tmdb_movies(strategy)

    # If still nothing, fail gracefully
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found for these preferences.")

    # Take top N for evaluation
    movies_subset = movies[:20]

    # Agent 3: candidate evaluator (uses Branch B via avoid_dark)
    scores = agent_candidate_evaluator(pref, movies_subset, avoid_dark)

    # Filter out disqualified, sort by score
    valid = [s for s in scores if not s.disqualified]
    valid_sorted = sorted(valid, key=lambda s: s.score, reverse=True)

    # Align with movie data
    id_to_movie = {m.id: m for m in movies_subset}
    ordered_movies = [id_to_movie[s.movie_id] for s in valid_sorted if s.movie_id in id_to_movie]

    # Agent 4: playlist builder
    playlist = agent_playlist_builder(ordered_movies, valid_sorted, pref)

    return RecommendResponse(
        preferences=pref,
        strategy=strategy,
        candidates_considered=len(movies_subset),
        movie_scores=valid_sorted,
        playlist=playlist,
    )