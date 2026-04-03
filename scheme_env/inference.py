import os
import sys
import json
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode('utf-8')
    req  = urllib.request.Request(
        ENV_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def env_reset(task: int) -> dict:
    return _post("/reset", {"seed": task})

def env_step(action_type: str, value: str = "") -> dict:
    return _post("/step", {"action": {"action_type": action_type, "value": value}})

def safe(text: str, limit: int = 999) -> str:
    return text.encode('ascii', 'ignore').decode()[:limit]

def grader_from_reward(reward: float) -> float:
    if reward == 10.0: return 1.0
    if reward == 5.0:  return 1.0
    if reward == 3.0:  return 0.5
    return 0.0

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Indian Government Welfare Officer AI agent.
Your job is to evaluate applicants and enroll them in the correct welfare scheme.

Available actions (respond ONLY with valid JSON, nothing else):
1. {"action_type": "ask_question", "value": "<field_name>"}
   - field_name must be one of: age, income, occupation, has_aadhaar
2. {"action_type": "request_document", "value": "<doc_name>"}
3. {"action_type": "approve_scheme", "value": "<scheme_name>"}
   - scheme_name options: PMKVY, MGNREGS, PMAY
4. {"action_type": "reject_applicant", "value": "<reason>"}
   - Use ONLY when applicant is clearly ineligible for ALL schemes
5. {"action_type": "escalate", "value": ""}

Scheme eligibility rules:
- PMKVY: Age 18-35, skilled worker (mason, carpenter etc), income < 10000
- MGNREGS: Age 18-60, rural labourer (farm_labourer etc), has_aadhaar = True
- PMAY: Age 21-55, any occupation, income < 6000, has_aadhaar = True
- REJECT if: age < 18, income > 100000, or no scheme matches profile

Strategy:
- If missing_data is empty, make a final decision immediately
- If missing_data has items, ask for them one by one before deciding
- If age < 18 or income > 100000, use reject_applicant immediately
- Never ask for info already in known_profile
- Always pick the MOST optimal scheme

Respond ONLY with a JSON object. No explanation. No markdown. Just JSON."""


def get_agent_action(observation: dict, history: list):
    obs_text = (
        f"Current observation:\n"
        f"- known_profile: {observation.get('known_profile', {})}\n"
        f"- missing_data: {observation.get('missing_data', [])}\n"
        f"- notification: {safe(observation.get('notification', ''))}\n"
        f"- is_terminated: {observation.get('is_terminated', False)}\n"
        f"\nWhat is your next action? Respond with JSON only."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": obs_text})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=100,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse: {raw!r}")
        action = {"action_type": "escalate", "value": ""}

    return action, raw


# ── Run one episode ────────────────────────────────────────────────────────────
def run_episode(task: int) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK {task} / 3")
    print(f"{'='*60}")

    result       = env_reset(task)
    obs          = result.get("observation", result)
    grader_score = 0.0

    print(f"  Profile   : {obs.get('known_profile', {})}")
    print(f"  Missing   : {obs.get('missing_data', [])}")
    print(f"  Notif     : {safe(obs.get('notification', ''), 80)}")

    history = []
    step    = 0

    while step < 12:
        step += 1

        if obs.get("is_terminated", False):
            grader_score = obs.get("metadata", {}).get("grader_score", 0.0)
            break

        action, raw_response = get_agent_action(obs, history)
        action_type = action.get("action_type", "escalate")
        value       = action.get("value", "")

        print(f"  Step {step:02d}: {action_type}({value!r})", end=" -> ")

        history.append({"role": "assistant", "content": raw_response})

        step_result = env_step(action_type, value)
        obs         = step_result.get("observation", step_result)
        reward      = step_result.get("reward", 0.0)
        done        = step_result.get("done", False)

        print(f"reward={reward}, done={done}")
        print(f"           {safe(obs.get('notification', ''), 70)}")

        history.append({
            "role": "user",
            "content": f"reward={reward}, notification={safe(obs.get('notification', ''))}"
        })

        if done:
            # Try metadata first, fallback to reward-based scoring
            grader_score = obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                grader_score = grader_from_reward(reward)
            break

        time.sleep(0.5)

    print(f"\n  GRADER SCORE: {grader_score:.1f} / 1.0")
    return grader_score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SCHEME ENV -- INFERENCE EVALUATION")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Env   : {ENV_URL}")
    print("="*60)

    scores = {}
    for task in [1, 2, 3]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"\n  [ERROR] Task {task} failed: {e}")
            scores[task] = 0.0
        time.sleep(1)

    print("\n" + "="*60)
    print("  FINAL GRADER SCORES")
    print("="*60)
    print(f"  Task 1 (Scheme Discovery)    : {scores[1]:.1f} / 1.0")
    print(f"  Task 2 (Missing Data)        : {scores[2]:.1f} / 1.0")
    print(f"  Task 3 (Conflict Resolution) : {scores[3]:.1f} / 1.0")
    print(f"  Average                      : {sum(scores.values())/3:.2f} / 1.0")
    print("="*60)


if __name__ == "__main__":
    main()