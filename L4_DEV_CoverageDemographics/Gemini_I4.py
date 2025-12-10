import os
from openai import OpenAI
import json


client = OpenAI(
    api_key='',#Input yourAPI key here(Deepseek)
    base_url="https://api.deepseek.com")



def read_txt(path: str, max_chars: int | None = None) -> str:
    print(f"[read_txt] trying to read: {path} | exists: {os.path.exists(path)}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED BY SCRIPT...]"
    return text


fmt_index_text = read_txt("The_Foundation_Model_Transparency_Index_v1.1.txt", max_chars=150000)
gemini_model_card_text = read_txt("868214523-Gemini-2-5-Pro-Preview-Model-Card.txt", max_chars=150000)
gemini_report_text     = read_txt("gemini_v2_5_report.txt", max_chars=150000)


# ========= Gemini 2.0 Pro：L4-2 Known gaps and limitations documented with mitigation plan =========

system_prompt_gaps_gemini = """
You are an expert evaluator of AI model transparency, dataset quality, and representativeness.

You will evaluate the Gemini 2.0 Pro model on the following indicator from the AI Ethics Index Category Tree:

- L4-2 (Dataset Quality & Representativeness):
  "Known gaps and limitations documented with mitigation plan."

This is a documentation-focused indicator: you must assess whether the model developer
explicitly documents (a) known gaps/limitations in data or model behavior across
populations/locales, AND (b) corresponding mitigation plans or strategies.

Use the following 0 / 0.5 / 1 rubric for L4-2, based on documentation (ADA):

- 1.0 (High):
  The documentation clearly and explicitly:
    * Describes concrete gaps or limitations (e.g., underrepresented languages, regions,
      demographics, failure modes, or fairness issues), AND
    * Describes concrete mitigation measures or plans tied to those gaps
      (e.g., data collection plans, fine-tuning or RLHF targeted at these groups,
      usage restrictions, safety layers, or other specific interventions).

- 0.5 (Medium / Partial):
  The documentation has some relevant content, but is incomplete:
    * It may mention high-level limitations or bias risks without specifying which
      groups/locales/tasks are affected, OR
    * It may discuss general mitigation (e.g., "we continuously improve", "we monitor bias")
      without clearly tying those measures to specific known gaps or limitations.

- 0.0 (Low / None):
  There is no meaningful disclosure of known gaps/limitations together with mitigation:
    * Either gaps/limitations are not discussed at all, OR
    * Only vague, generic disclaimers appear (e.g., "the model may be biased") with no
      concrete details, AND no clear mitigation plan is articulated.

Important:
- Explicit, concrete descriptions of both gaps and mitigation are required for 1.0.
- Partial or generic statements justify at most 0.5.
- If you cannot find any substantive disclosure, assign 0.0.

Your tasks:

1. From the FMTI text, infer and summarize the relevant principles for this indicator:
   - How FMTI treats transparency about known limitations, risks, and mitigation
     (even if FMTI does not use exactly this L4 label).
   - Summarize these principles in your own words as the "rubric_summary" for L4-2.

2. Read the Gemini-related documents (model card, technical/analysis report) and extract verbatim evidence
   related to:
   - known gaps or limitations in data or model behavior across demographics/locales,
   - bias, fairness, or representativeness limitations,
   - any mitigation plans, safeguards, or ongoing improvement plans that address these gaps.

   For each evidence item, record:
   - which document it comes from,
   - any location information you can infer (page/section if mentioned, or 'unknown'),
   - a short verbatim English quote (do NOT paraphrase).

3. Using ONLY the provided documents, assign a score for this indicator:
   - raw_score must be one of: 0, 0.5, or 1.
   - normalized_score_0_1 should be identical to raw_score.
   - Justify your score in 3–6 sentences:
     * Refer to the rubric above (explicit vs. partial vs. none).
     * Explain why the evidence supports 0 / 0.5 / 1.

Output:

You MUST output a single JSON object with the following structure:

{
  "model": "Gemini 2.0 Pro",
  "indicator_L4_gaps": {
    "rubric_summary": "Short description of the scoring rules for L4-2 in your own words.",
    "raw_score": 0,
    "raw_scale": "0, 0.5, or 1 (documentation-based gaps & mitigation disclosure)",
    "normalized_score_0_1": 0.0,
    "justification": "3–6 sentences explaining how Gemini 2.0 Pro meets or fails the criteria for L4-2.",
    "evidence": [
      {
        "doc": "Gemini 2.0 Pro Model Card / Report",
        "location": "page/section if available, or 'unknown'",
        "quote": "verbatim English quote"
      }
    ]
  }
}

Constraints:
- DO NOT invent documents or citations.
- ONLY use information contained in the provided texts.
- If you find no relevant disclosure for this indicator, set raw_score = 0 and explain why.
- All JSON keys and values must be in English.
- Your entire response must be valid JSON. Do NOT include any commentary outside the JSON object.
"""

user_content_gaps_gemini = f"""
Below are the documents you can use. They are plain-text extractions from PDFs or web pages.

[DOCUMENT: Gemini 2.0 Pro Model Card / Preview Model Card]
{gemini_model_card_text}

[DOCUMENT: Gemini v2.5 Report]
{gemini_report_text}

[DOCUMENT: Foundation Model Transparency Index v1.1]
{fmt_index_text}

Please now evaluate Gemini 2.0 Pro on the indicator
'L4-2: Known gaps and limitations documented with mitigation plan'
and output ONLY the JSON object.
"""

response_gaps_gemini = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": system_prompt_gaps_gemini},
        {"role": "user", "content": user_content_gaps_gemini},
    ],
    temperature=0.1,
    stream=False,
)

raw_output_gaps_gemini = response_gaps_gemini.choices[0].message.content
print("Raw model output (Gemini L4-2 gaps):")
print(raw_output_gaps_gemini)

try:
    scores_gaps_gemini = json.loads(raw_output_gaps_gemini)
    print("\nParsed scores (Gemini L4-2 gaps):")
    print(json.dumps(scores_gaps_gemini, indent=2, ensure_ascii=False))
    with open("gemini2_L4_gaps_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores_gaps_gemini, f, indent=2, ensure_ascii=False)
except json.JSONDecodeError:
    print("\n⚠️ The model output is not valid JSON (Gemini L4-2 gaps). Please manually check it or do a bit of cleaning before parsing.")
