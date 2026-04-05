# Version 4
# Mini SelfCheckGPT-style adaptive confidence routing

import ollama
import re
from llm_model import primary_response  # optional LLM function
from model_3 import ask_model  # optional LLM function

SLM_name = 'phi3:3.8b'  # local SLM model


# -------------------------------
# SINGLE SLM CALL
# -------------------------------
def slm_generate(query):
    prompt = f"""
You MUST follow the format EXACTLY.

You are a SMALL and careful AI model.

Instructions:
- Answer briefly.
- Give a realistic confidence (0 to 1).
- If unsure, say EXACTLY: Sorry, I can't answer that.

STRICT FORMAT:
Answer: <short answer>
Confidence: <number>

IMPORTANT:
- Do NOT generate anything after Confidence.
- Do NOT ask new questions.
- Stop immediately after Confidence line.

Question: {query}
"""

    try:
        response = ollama.chat(
            model=SLM_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        text = response['message']['content'].strip()

        # Keep only up to Confidence line
        # Keep only up to Confidence line, whether or not "Answer:" exists
        match = re.search(r"^(.*?Confidence:\s*[0-9]*\.?[0-9]+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    except Exception as e:
        print("\nSLM Error:", e)
        return {
            "answer": "Sorry, I can't answer that.",
            "confidence": 0.0,
            "raw": str(e)
        }

    return parse_slm_output(text)


# -------------------------------
# MULTI-SLM
# -------------------------------
def multi_slm_generate(query, num_samples):
    results = []

    for i in range(num_samples):
        print(f"\n--- SLM Sample {i+1} ---")
        res = slm_generate(query)
        print(res)
        results.append(res)

    return results


# -------------------------------
# Parse SLM output
# -------------------------------
def parse_slm_output(text):
    answer = ""
    confidence = 0.0

    try:
        answer_match = re.search(r"Answer:\s*(.*?)(?:\n|Confidence:)", text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # fallback: take everything before Confidence if Answer: is missing
            fallback_match = re.search(r"^(.*?)(?:\n|Confidence:)", text, re.DOTALL)
            if fallback_match:
                answer = fallback_match.group(1).strip()
            else:
                answer = text.strip()

        conf_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            confidence = 0.5

        answer = clean_answer_text(answer)

    except Exception as e:
        print("Parsing failed:", e)
        print("Raw:", text)
        return {
            "answer": clean_answer_text(text.strip()),
            "confidence": 0.2,
            "raw": text
        }

    return {
        "answer": answer,
        "confidence": confidence,
        "raw": text
    }


# -------------------------------
# SEMANTIC MATCH
# -------------------------------
def semantic_match(ans1, ans2):
    prompt = f"""
Compare two answers.

Answer 1: {ans1}
Answer 2: {ans2}

Do they mean the same thing?

Reply ONLY with YES or NO.
"""

    try:
        response = ollama.chat(
            model=SLM_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        result = response['message']['content'].strip().lower()
        return "yes" in result

    except Exception:
        return False


# -------------------------------
# WEIGHTED SEMANTIC CONSENSUS
# -------------------------------
def weighted_semantic_consensus(results):
    answers = [r["answer"] for r in results]
    confidences = [r["confidence"] for r in results]

    n = len(answers)
    scores = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                scores[i] += confidences[i]
            else:
                if answers[i].strip().lower() == answers[j].strip().lower():
                    scores[i] += confidences[j]
                elif semantic_match(answers[i], answers[j]):
                    scores[i] += confidences[j]

    best_index = scores.index(max(scores))
    return answers[best_index], scores[best_index], answers, scores


# -------------------------------
# AVERAGE CONFIDENCE
# -------------------------------
def average_confidence(results):
    return sum(r["confidence"] for r in results) / len(results)


# -------------------------------
# SELF-CHECK STYLE CONFIDENCE
# -------------------------------
def selfcheck_confidence(results, final_answer):
    supporters = []
    total = len(results)

    clean_final = clean_answer_text(final_answer)

    for r in results:
        ans = clean_answer_text(r["answer"])

        # Exact cleaned match
        if ans.lower() == clean_final.lower():
            supporters.append(r)

        # Otherwise semantic check on cleaned answers
        elif semantic_match(ans, clean_final):
            supporters.append(r)

    agreement_ratio = len(supporters) / total if total > 0 else 0.0

    if supporters:
        supporter_avg_conf = sum(r["confidence"] for r in supporters) / len(supporters)
    else:
        supporter_avg_conf = 0.0

    final_conf = 0.7 * agreement_ratio + 0.3 * supporter_avg_conf

    return final_conf, agreement_ratio, supporter_avg_conf

# -------------------------------
# LLM Wrapper
# -------------------------------
def llm_generate(query):
    route_confirmation = input("Do you want to route to LLM: ")
    if route_confirmation.lower() != "yes":
        return "[Routing canceled.]"

    print("\nRouting to LLM (OpenRouter)...")
    llm_response = ask_model(query)
    print('\nLLM response:')
    print(llm_response)
    return "[LLM response printed above]"


# -------------------------------
# SATER ROUTER + MINI SELF-CHECKGPT
# -------------------------------
def sater_router(query, threshold=0.75):            #threshold  0.70 -> more willing to trust slm
    # -------------------------------                           0.75 -> balanced
    # STEP 1: INITIAL 2 SAMPLES                                 0.80 -> routes more often
    # -------------------------------
    print("\n=== Initial Sampling (2) ===")
    results = multi_slm_generate(query, num_samples=2)

    answers = [r["answer"] for r in results]

    # -------------------------------
    # Case 1: Refusal in initial stage
    # -------------------------------
    if any("sorry" in r["answer"].lower() for r in results):
        print("\nCase-1: Refusal detected")
        return llm_generate(query)

    # -------------------------------
    # FAST PATH: Exact match
    # -------------------------------
    if answers[0].strip().lower() == answers[1].strip().lower():
        print("\nFast exact agreement between first 2 samples")

        final_answer = answers[0]
        final_conf, agreement_ratio, supporter_avg_conf = selfcheck_confidence(results, final_answer)

        print("\n--- SelfCheck Results (2 samples) ---")
        print("All answers:", answers)
        print("Chosen answer:", final_answer)
        print("Agreement ratio:", agreement_ratio)
        print("Supporter avg confidence:", supporter_avg_conf)
        print("Final confidence:", final_conf)

        if final_conf >= threshold:
            print("\nCase-4: Using SLM (fast path)")
            return final_answer
        else:
            print("\nCase-3: Low SelfCheck confidence")
            return llm_generate(query)

    # -------------------------------
    # SEMANTIC CHECK (2 samples)
    # -------------------------------
    print("\nChecking semantic agreement (2 samples)...")

    if semantic_match(answers[0], answers[1]):
        final_answer = answers[0]
        final_conf, agreement_ratio, supporter_avg_conf = selfcheck_confidence(results, final_answer)

        print("\n--- SelfCheck Results (2 samples, semantic) ---")
        print("All answers:", answers)
        print("Chosen answer:", final_answer)
        print("Agreement ratio:", agreement_ratio)
        print("Supporter avg confidence:", supporter_avg_conf)
        print("Final confidence:", final_conf)

        if final_conf >= threshold:
            print("\nCase-4: Using SLM (semantic agreement)")
            return final_answer
        else:
            print("\nCase-3: Low SelfCheck confidence")
            return llm_generate(query)

    # -------------------------------
    # STEP 2: DISAGREEMENT → ADD 3rd SAMPLE
    # -------------------------------
    print("\nDisagreement detected → Generating 3rd sample...")

    print("\n--- SLM Sample 3 ---")
    third_sample = slm_generate(query)
    print(third_sample)
    results.append(third_sample)

    # -------------------------------
    # Case 2A: Refusal after 3rd sample
    # -------------------------------
    if any("sorry" in r["answer"].lower() for r in results):
        print("\nCase-1: Refusal detected")
        return llm_generate(query)

    # -------------------------------
    # STEP 3: FULL CONSENSUS (3 samples)
    # -------------------------------
    final_answer, best_score, answers, scores = weighted_semantic_consensus(results)
    avg_conf = average_confidence(results)
    final_conf, agreement_ratio, supporter_avg_conf = selfcheck_confidence(results, final_answer)

    print("\n--- Aggregated Results (3 samples) ---")
    print("All answers:", answers)
    print("Scores:", scores)
    print("Chosen answer:", final_answer)
    print("Best score:", best_score)
    print("Average confidence:", avg_conf)
    print("Agreement ratio:", agreement_ratio)
    print("Supporter avg confidence:", supporter_avg_conf)
    print("Final SelfCheck confidence:", final_conf)

    # -------------------------------
    # Case 2: Weak semantic agreement
    # -------------------------------
    if best_score < (0.6 * sum(r["confidence"] for r in results)):
        print("\nCase-2: Weak semantic agreement")
        return llm_generate(query)

    # -------------------------------
    # Case 3: Low SelfCheck confidence
    # -------------------------------
    if final_conf < threshold:
        print("\nCase-3: Low SelfCheck confidence")
        return llm_generate(query)

    # -------------------------------
    # Case 4: Accept SLM
    # -------------------------------
    print("\nCase-4: Using SLM (adaptive SelfCheck consensus)")
    return final_answer


# -------------------------------
# remove noise from the responses
# -------------------------------
def clean_answer_text(ans):
    ans = ans.strip()

    # Remove leaked Confidence part
    ans = re.sub(r"Confidence:\s*[0-9]*\.?[0-9]+", "", ans, flags=re.IGNORECASE)

    # Remove leaked Question part and everything after it
    ans = re.sub(r"Question:.*", "", ans, flags=re.IGNORECASE | re.DOTALL)

    # Remove leading labels like "Capital:"
    ans = re.sub(r"^[A-Za-z\s]+:\s*", "", ans)

    # Collapse extra whitespace
    ans = re.sub(r"\s+", " ", ans).strip()

    return ans
# -------------------------------
# Main
# -------------------------------
def main():
    while True:
        query = input("\nEnter your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        result = sater_router(query)

        if result:
            print("\n=== Final Answer ===")
            print(result)


if __name__ == "__main__":
    main()
