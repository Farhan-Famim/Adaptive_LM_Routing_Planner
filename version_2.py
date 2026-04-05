# Version 2
# Multi-sample + Semantic SATER routing

import ollama
import re
from llm_model import primary_response  # LLM function

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
def multi_slm_generate(query, num_samples=2):
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
def semantic_consensus(results):
    answers = [r["answer"] for r in results]

    base = answers[0]
    votes = 1

    for ans in answers[1:]:
        # ✅ Step 1: Exact match (FAST + RELIABLE)
        if base.strip().lower() == ans.strip().lower():
            votes += 1

        # ✅ Step 2: Otherwise use semantic check
        elif semantic_match(base, ans):
            votes += 1

    return base, votes, answers


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
    primary_response(query)
    return "[LLM response printed above]"


# -------------------------------
# SATER ROUTER (SEMANTIC)
# -------------------------------
def sater_router(query, threshold=0.6):
    # -------------------------------
    # MULTI-SLM
    # -------------------------------
    results = multi_slm_generate(query)

    final_answer, votes, answers = semantic_consensus(results)
    avg_conf = average_confidence(results)

    print("\n--- Aggregated Results ---")
    print("All answers:", answers)
    print("Chosen answer:", final_answer)
    print("Semantic agreement:", votes)
    print("Average confidence:", avg_conf)

    # -------------------------------
    # Case 1: Refusal
    # -------------------------------
    if any("sorry" in r["answer"].lower() for r in results):
        print("\nCase-1: Refusal detected")
        return llm_generate(query)

    # -------------------------------
    # Case 2: Semantic disagreement
    # -------------------------------
    if votes < len(results):
        print("\nCase-2: Semantic disagreement")
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
    print("\nCase-4: Using SLM semantic agreement")
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
