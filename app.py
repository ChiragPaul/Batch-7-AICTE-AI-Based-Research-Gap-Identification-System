from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, NotFound
import arxiv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
import json
import re
import traceback


load_dotenv()

app = Flask(__name__)
CORS(app)


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("HuggingFace Token Loaded Successfully")
else:
    print("Warning: HF_TOKEN not found in .env file - using public model access")


API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
ANALYSIS_PROVIDER = os.getenv("ANALYSIS_PROVIDER", "auto").strip().lower()

if API_KEY:
    genai.configure(api_key=API_KEY)
    print("Gemini API Key Loaded Successfully")
else:
    print("Warning: GEMINI_API_KEY not found in .env file")



model_name = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print(f"Logged in to HuggingFace Hub with model: {model_name}")

embedding_model = SentenceTransformer(model_name, use_auth_token=HF_TOKEN)



HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

SELECTED_GEMINI_MODEL = None


def normalize_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    return name if name.startswith("models/") else f"models/{name}"


def list_generate_content_models():
    models = []
    for model in genai.list_models():
        methods = getattr(model, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            models.append(model.name)
    return models


def resolve_gemini_model_name():
    """Pick a valid model for this API key/account."""
    global SELECTED_GEMINI_MODEL
    if SELECTED_GEMINI_MODEL:
        return SELECTED_GEMINI_MODEL

    available = list_generate_content_models()
    if not available:
        raise RuntimeError("No Gemini models with generateContent access were found for this API key")

    preferred = normalize_model_name(GEMINI_MODEL)
    if preferred and preferred in available:
        SELECTED_GEMINI_MODEL = preferred
        return SELECTED_GEMINI_MODEL


    flash = [m for m in available if "flash" in m.lower()]
    SELECTED_GEMINI_MODEL = flash[0] if flash else available[0]
    return SELECTED_GEMINI_MODEL


def extract_json_from_text(text: str):
    """Extract first valid JSON object from model output."""
    text = (text or "").strip()

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(re.findall(r"\{[\s\S]*\}", text))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue

    return None


def response_text_or_none(response):
    if not response:
        return None
    try:
        if getattr(response, "text", None):
            return response.text.strip()
    except Exception:
        pass

    parts = []
    candidates = getattr(response, "candidates", []) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)

    return "\n".join(parts).strip() if parts else None


def choose_analysis_provider():
    """Resolve analysis provider from env and available keys."""
    if ANALYSIS_PROVIDER in {"gemini", "huggingface"}:
        return ANALYSIS_PROVIDER
    if HF_TOKEN:
        return "huggingface"
    if API_KEY:
        return "gemini"
    raise RuntimeError(
        "No valid analysis provider configured. Set ANALYSIS_PROVIDER and API keys in .env"
    )


def build_analysis_prompt(topic, abstracts, trends, language):
    return f"""
Topic: {topic}
Preferred Output Language: {language}

Abstracts:
{' '.join(abstracts[:5])}

Current Trends:
{trends}

Return ONLY valid JSON with this exact schema:
{{
  "trends": ["3-5 current trends"],
  "gaps": ["3 research gaps"],
  "ideas": ["3 project ideas"],
  "future_work": ["2-3 future directions"],
  "gap_score": "score out of 10"
}}

Important rules:
- Keep all values in "{language}".
- Keep keys exactly as shown in the schema.
- Do not include markdown fences or extra text.
"""


def analyze_with_gemini(prompt):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing but Gemini provider was selected")

    model_name = resolve_gemini_model_name()
    gemini_model = genai.GenerativeModel(model_name)

    try:
        response = gemini_model.generate_content(prompt)
    except NotFound:
        global SELECTED_GEMINI_MODEL
        SELECTED_GEMINI_MODEL = None
        model_name = resolve_gemini_model_name()
        gemini_model = genai.GenerativeModel(model_name)
        response = gemini_model.generate_content(prompt)

    response_text = response_text_or_none(response)
    if not response_text:
        raise RuntimeError("Empty response from Gemini")

    return response_text, model_name


def analyze_with_huggingface(prompt):
    if not hf_client:
        raise RuntimeError("HF_TOKEN is missing but HuggingFace provider was selected")

    completion = hf_client.chat_completion(
        model=HF_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research analysis assistant. Respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    response_text = completion.choices[0].message.content if completion and completion.choices else None
    if not response_text:
        raise RuntimeError("Empty response from HuggingFace model")

    return response_text, HF_LLM_MODEL


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Research Gap API is running", "health": "/health", "analyze": "/analyze"})



@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True)

        if not data or "topic" not in data:
            return jsonify({"error": "Topic is required"}), 400

        topic = str(data["topic"]).strip()
        language = str(data.get("language", "English")).strip()
        if not topic:
            return jsonify({"error": "Topic cannot be empty"}), 400
        if not language:
            language = "English"


        client_arxiv = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)
        search = arxiv.Search(query=topic, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)

        papers = list(client_arxiv.results(search))
        abstracts = [paper.summary.strip() for paper in papers if paper.summary]

        if not abstracts:
            return jsonify({"error": "No papers found for this topic"}), 404


        embeddings = embedding_model.encode(abstracts)
        num_clusters = min(3, len(abstracts))
        trends = []

        if num_clusters > 1:
            kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            kmeans.fit(embeddings)

            clusters = {}
            for i, label in enumerate(kmeans.labels_):
                clusters.setdefault(int(label), []).append(abstracts[i])

            for cluster_id, cluster_items in sorted(clusters.items()):
                trends.append(f"Cluster {cluster_id}: {len(cluster_items)} papers")
        else:
            trends.append("Cluster 0: 1 paper")


        prompt = build_analysis_prompt(topic, abstracts, trends, language)
        provider = choose_analysis_provider()

        if provider == "huggingface":
            response_text, model_name = analyze_with_huggingface(prompt)
        else:
            response_text, model_name = analyze_with_gemini(prompt)


        result = extract_json_from_text(response_text)
        if not isinstance(result, dict):
            return jsonify(
                {
                    "error": "Gemini response did not contain valid JSON object",
                    "raw_response": response_text,
                }
            ), 500

        required_keys = ["trends", "gaps", "ideas", "future_work", "gap_score"]
        for key in required_keys:
            result.setdefault(key, [] if key != "gap_score" else "N/A")

        result["model_used"] = model_name
        result["provider_used"] = provider
        result["language_used"] = language
        return jsonify(result)

    except GoogleAPICallError as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
