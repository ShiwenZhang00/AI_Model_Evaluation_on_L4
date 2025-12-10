# Please install OpenAI SDK first: `pip3 install openai`
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


gpt4o_system_card_text = read_txt("gpt4o_system_card.txt", max_chars=150000)
gpt4_tech_report_text  = read_txt("gpt4_technical_report.txt", max_chars=150000)
fmt_index_text = read_txt("The_Foundation_Model_Transparency_Index_v1.1.txt", max_chars=150000)



system_prompt = """
You are an expert evaluator of AI model transparency and data governance.

You will evaluate the GPT-4o model on two specific indicators, as defined in the
\"The AI Ethics Index Category Tree\" (The AI Ethics Index (AIEI) is built on research. The rationale for the AIEI Tree Structure is explained in the accompanying AI Ethics Index Briefing):

- L4-1: Training data sources and licenses are disclosed at an aggregate level.
- L4-2: Third-party datasets include SPDX or clear license metadata.

Important: The FMTI document provided in the context is the primary scoring rubric.
FMTI itself uses a strict binary 0/1 scoring scheme based on explicit disclosure.
For this course, we introduce a slight extension of that scheme:

- 1.0: There is clear, explicit disclosure in the documents that satisfies the indicator.
- 0.5: There is related or suggestive discussion, but no fully explicit disclosure
       that unambiguously satisfies the indicator.
- 0.0: There is no disclosure at all, or the information is so vague that you cannot
       reliably infer that the indicator is satisfied.

You MUST still follow the FMTI logic that explicit disclosure is the gold standard.
Only explicit, unambiguous statements qualify for a 1.0. Partial, indirect, or implied
mentions may justify a 0.5, but never a 1.0.    

Your tasks:

1. From the FMTI text, locate the sections that define the scoring rubric
   (criteria, examples, or thresholds) that are relevant to L4-1 and L4-2.
   - Extract and summarize the relevant rubric for each indicator in your own words.
   - Keep the summary short but precise, and make clear what behavior corresponds
     to explicit disclosure vs. partial/implicit mention vs. no disclosure.

2. Read the provided GPT-4/GPT-4o-related documents (system card, technical report,
   and any other GPT-4/GPT-4o transparency documents in the context), and extract verbatim evidence
   that is relevant to each indicator.
   - For each piece of evidence, record:
     * which document it comes from,
     * any location information you can infer (page/section if mentioned),
     * and a short verbatim quote (do not paraphrase the evidence).

3. Using the FMTI-style rubric plus the 0 / 0.5 / 1 extension, assign a score for each indicator.
   - raw_score MUST be one of: 0, 0.5, or 1.
   - 1.0: The documents contain clear, explicit disclosure that satisfies the indicator.
   - 0.5: There is relevant or partially informative discussion, but disclosure is not fully explicit
          or is ambiguous in a way that falls short of the FMTI explicit-disclosure standard.
   - 0.0: The documents do NOT contain meaningful disclosure for the indicator
          (including cases where information is missing, purely implicit, or too vague).
   - Your justification MUST reference:
     * the FMTI rubric logic (especially the explicit-disclosure threshold), and
     * the specific evidence (or lack of evidence) you used to choose 0 / 0.5 / 1.

4. Output your result strictly as a single JSON object with the following structure:

{
  "model": "GPT-4o",
  "indicator_L4_1": {
    "rubric_summary": "Short description of the FMTI-style scoring rules for L4-1 (including the 0/0.5/1 extension) in your own words.",
    "raw_score": 0,
    "raw_scale": "0, 0.5, or 1 (FMTI-style explicit-disclosure logic with a partial-credit extension)",
    "normalized_score_0_1": 0.0,
    "justification": "3–6 sentences explaining how GPT-4o meets or fails the criteria for L4-1. Explicitly reference the explicit-disclosure threshold and why the score is 0 / 0.5 / 1.",
    "evidence": [
      {
        "doc": "GPT-4o System Card",
        "location": "page/section if available, or 'unknown'",
        "quote": "verbatim English quote"
      }
    ]
  },
  "indicator_L4_2": {
    "rubric_summary": "Short description of the FMTI-style scoring rules for L4-2 (including the 0/0.5/1 extension) in your own words.",
    "raw_score": 0,
    "raw_scale": "0, 0.5, or 1 (FMTI-style explicit-disclosure logic with a partial-credit extension)",
    "normalized_score_0_1": 0.0,
    "justification": "3–6 sentences explaining how GPT-4o meets or fails the criteria for L4-2. Explicitly reference the explicit-disclosure threshold and why the score is 0 / 0.5 / 1.",
    "evidence": [
      {
        "doc": "GPT-4 Technical Report",
        "location": "page/section if available, or 'unknown'",
        "quote": "verbatim English quote"
      }
    ]
  }
}

Normalization rule:
- Because raw_score is in {0, 0.5, 1}, set normalized_score_0_1 to exactly the same value
  as raw_score.

Additional constraints:
- DO NOT invent documents or citations.
- ONLY use information contained in the provided texts.
- If you don't find any relevant disclosure for an indicator, set raw_score = 0 and explain why.
- Keep all text in the JSON keys and values in English.
- Your entire response must be valid JSON. Do NOT include any commentary outside the JSON object.
"""



user_content = f"""
Below are the documents you can use. They are plain-text extractions from PDFs or web pages.

[DOCUMENT: GPT-4o System Card]
{gpt4o_system_card_text}

[DOCUMENT: GPT-4 Technical Report]
{gpt4_tech_report_text}


[DOCUMENT: Foundation Model Transparency Index v1.1]
{fmt_index_text}

Please now evaluate GPT-4o on indicator L4-1 and L4-2, and output ONLY the JSON object.
"""

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ],
    temperature=0.1,
    stream=False,
)

raw_output = response.choices[0].message.content
print("Raw model output:")
print(raw_output)


try:
    scores = json.loads(raw_output)
    print("\nParsed scores:")
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    with open("gpt4o_L4_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
except json.JSONDecodeError:
    print("\n⚠️ The model output is not valid JSON. Please manually check it or do a bit of cleaning before parsing.")