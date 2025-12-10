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

# ========= Gemini 2.0 Pro 的 L4-1：Coverage across demographics or locales is characterized =========

system_prompt_coverage_gemini = """
You are an expert evaluator of AI model transparency, dataset quality, and representativeness.

You will evaluate the Gemini 2.0 Pro model on the following indicator from the AI Ethics Index Category Tree:

- L4-1 (Dataset Quality & Representativeness):
  "Coverage across demographics or locales is characterized (no raw protected attributes required)."

Important:
- The provided "Foundation Model Transparency Index v1.1" (FMTI) text is a primary reference for
  transparency-style scoring (even if it does not define this exact L4 verbatim).
- You should emulate an FMTI-style approach: focus on whether the model developer explicitly
  characterizes coverage across populations/locales in documentation.

Use the following 0 / 0.5 / 1 rubric for L4-1, based on documentation (ADA):

- 1.0 (High):
  There is explicit, systematic characterization of coverage across demographics or locales.
  Examples include:
    * Language or region distributions (e.g., data or performance by language/region).
    * Clear statements about what populations/locales are well covered vs. underrepresented.
    * Tables, figures, or dedicated sections describing coverage.

- 0.5 (Medium / Partial):
  There are some relevant mentions, but characterization is incomplete or fragmented.
  Examples:
    * The documentation mentions that the model is multilingual or used worldwide
      and vaguely refers to multiple languages, regions, or populations,
      but does NOT provide systematic or structured coverage descriptions.
    * There are isolated comments about decreased performance for some languages/locales
      or general references to "diverse populations" without concrete characterization.

- 0.0 (Low / None):
  There is no meaningful disclosure of coverage across demographics or locales.
  Examples:
    * Only generic phrases like "large and diverse dataset" without specifying populations,
      languages, or regions.
    * No explicit indication of which groups/locales are covered or underrepresented.

You must:
- Treat explicit, structured disclosures as the gold standard for 1.0.
- Use 0.5 only when there is some concrete but incomplete or high-level coverage discussion.
- Use 0.0 when you cannot find any meaningful coverage characterization.

Your tasks:

1. From the FMTI text, infer and summarize the relevant principles that apply to this indicator:
   - How FMTI treats transparency about data representativeness, coverage, or population-level
     descriptions.
   - Summarize these principles in your own words as the "rubric_summary" for L4-1.

2. Read the Gemini-related documents (model card, technical/analysis report) and extract verbatim evidence
   related to:
   - demographic or geographic coverage,
   - language or locale coverage,
   - any discussion of representativeness or underrepresented groups/locales,
   - known gaps in coverage or performance across groups/locales.

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
  "indicator_L4_coverage": {
    "rubric_summary": "Short description of the scoring rules for L4-1 in your own words.",
    "raw_score": 0,
    "raw_scale": "0, 0.5, or 1 (documentation-based coverage characterization)",
    "normalized_score_0_1": 0.0,
    "justification": "3–6 sentences explaining how Gemini 2.0 Pro meets or fails the criteria for L4-1.",
    "evidence": [
      {
        "doc": "Gemini 2.0 Pro Model Card",
        "location": "page/section if available, or 'unknown'",
        "quote": "verbatim English quote"
      }
    ]
  }
}

Constraints:
- DO NOT invent documents or citations.
- ONLY use information contained in the provided texts.
- If you find no relevant coverage characterization, set raw_score = 0 and explain why.
- All JSON keys and values must be in English.
- Your entire response must be valid JSON. Do NOT include any commentary outside the JSON object.
"""

user_content_coverage_gemini = f"""
Below are the documents you can use. They are plain-text extractions from PDFs or web pages.

[DOCUMENT: Gemini 2.0 Pro Model Card / Preview Model Card]
{gemini_model_card_text}

[DOCUMENT: Gemini v2.5 Report]
{gemini_report_text}

[DOCUMENT: Foundation Model Transparency Index v1.1]
{fmt_index_text}

Please now evaluate Gemini 2.0 Pro on the indicator
'L4-1: Coverage across demographics or locales is characterized'
and output ONLY the JSON object.
"""

response_coverage_gemini = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": system_prompt_coverage_gemini},
        {"role": "user", "content": user_content_coverage_gemini},
    ],
    temperature=0.1,
    stream=False,
)

raw_output_coverage_gemini = response_coverage_gemini.choices[0].message.content
print("Raw model output (Gemini L4 coverage):")
print(raw_output_coverage_gemini)

try:
    scores_coverage_gemini = json.loads(raw_output_coverage_gemini)
    print("\nParsed scores (Gemini L4 coverage):")
    print(json.dumps(scores_coverage_gemini, indent=2, ensure_ascii=False))
    with open("gemini2_L4_coverage_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores_coverage_gemini, f, indent=2, ensure_ascii=False)
except json.JSONDecodeError:
    print("\n⚠️ The model output is not valid JSON (Gemini L4 coverage). Please manually check it or do a bit of cleaning before parsing.")
