# Version 3
# Multi-sample + Semantic SATER routing

import ollama
import re
from llm_model import primary_response  # LLM function
from model_3 import ask_model # optional LLM function

SLM_name = 'phi3:3.8b'  # local SLM model
num_of_slm_samples = 3  # number of times the same query will be asked to the SLM (samples)


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

        # Remove prompt leakage, and
        # Keep only up to Confidence line
        match = re.search(r"(Answer:.*?Confidence:\s*[0-9]*\.?[0-9]+)", text, re.DOTALL)

        if match:
            text = match.group(1)

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
def multi_slm_generate(query, num_samples=num_of_slm_samples):
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

        conf_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text)
        if conf_match:
            confidence = float(conf_match.group(1))

        # fallback improvement
        if not answer:
            answer = text.strip()

        if confidence == 0.0:
            conf_match = re.search(r"confidence[:\s]*([0-9]*\.?[0-9]+)", text.lower())
            if conf_match:
                confidence = float(conf_match.group(1))
            else:
                confidence = 0.5

    except Exception as e:
        print("Parsing failed:", e)
        print("Raw:", text)
        return {
            "answer": text.strip(),
            "confidence": 0.2,
            "raw": text
        }

    return {
        "answer": answer,
        "confidence": confidence,
        "raw": text
    }


# -------------------------------
# SEMANTIC MATCH (NEW)
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

    except:
        return False


# -------------------------------
# SEMANTIC CONSENSUS (OPTIMIZED)
# -------------------------------
def weighted_semantic_consensus(results):
    answers = [r["answer"] for r in results]
    confidences = [r["confidence"] for r in results]

    n = len(answers)
    scores = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                scores[i] += confidences[i]  # self weight

            else:
                # Exact match first
                if answers[i].strip().lower() == answers[j].strip().lower():
                    scores[i] += confidences[j]

                # Otherwise semantic match
                elif semantic_match(answers[i], answers[j]):
                    scores[i] += confidences[j]

    # pick best answer
    best_index = scores.index(max(scores))

    return answers[best_index], scores[best_index], answers, scores


# -------------------------------
# AVERAGE CONFIDENCE
# -------------------------------
def average_confidence(results):
    return sum(r["confidence"] for r in results) / len(results)


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
# SATER ROUTER (SEMANTIC)
# -------------------------------
def sater_router(query, threshold=0.6):
    # -------------------------------
    # STEP 1: INITIAL 2 SAMPLES
    # -------------------------------
    print("\n=== Initial Sampling (2) ===")
    results = multi_slm_generate(query, num_samples=2)

    answers = [r["answer"] for r in results]
    avg_conf = average_confidence(results)

    # -------------------------------
    # FAST PATH: Exact match
    # -------------------------------
    if answers[0].strip().lower() == answers[1].strip().lower():
        print("\nFast agreement between first 2 samples")

        if avg_conf >= threshold:
            print("\nCase-4: Using SLM (fast path)")
            return answers[0]
        else:
            print("\nCase-3: Low confidence")
            return llm_generate(query)

    # -------------------------------
    # SEMANTIC CHECK (2 samples)
    # -------------------------------
    print("\nChecking semantic agreement (2 samples)...")

    if semantic_match(answers[0], answers[1]):
        if avg_conf >= threshold:
            print("\nCase-4: Using SLM (semantic agreement)")
            return answers[0]
        else:
            print("\nCase-3: Low confidence")
            return llm_generate(query)

    # -------------------------------
    # STEP 2: DISAGREEMENT → ADD 3rd SAMPLE
    # -------------------------------
    print("\nDisagreement detected → Generating 3rd sample...")

    print(f"\n--- SLM Sample 3 ---")
    third_sample = slm_generate(query)
    print(third_sample)

    results.append(third_sample)

    # -------------------------------
    # STEP 3: FULL CONSENSUS (3 samples)
    # -------------------------------
    final_answer, best_score, answers, scores = weighted_semantic_consensus(results)
    avg_conf = average_confidence(results)

    print("\n--- Aggregated Results (3 samples) ---")
    print("All answers:", answers)
    print("Scores:", scores)
    print("Chosen answer:", final_answer)
    print("Best score:", best_score)
    print("Average confidence:", avg_conf)

    # -------------------------------
    # Case 1: Refusal
    # -------------------------------
    if any("sorry" in r["answer"].lower() for r in results):
        print("\nCase-1: Refusal detected")
        return llm_generate(query)

    # -------------------------------
    # Case 2: Weak semantic agreement
    # -------------------------------
    if best_score < (0.6 * sum(r["confidence"] for r in results)):
        print("\nCase-2: Weak semantic agreement")
        return llm_generate(query)

    # -------------------------------
    # Case 3: Low confidence
    # -------------------------------
    if avg_conf < threshold:
        print("\nCase-3: Low average confidence")
        return llm_generate(query)

    # -------------------------------
    # Case 4: Accept SLM
    # -------------------------------
    print("\nCase-4: Using SLM (adaptive consensus)")
    return final_answer


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
