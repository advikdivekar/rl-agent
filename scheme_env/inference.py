import os
import sys
import json
import time
import urllib.request
import re

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "scheme_env"


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ENV_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def env_reset(task: int) -> dict:
    return _post("/reset", {"seed": task})


def env_step(action_type: str, value: str) -> dict:
    return _post("/step", {"action": {"action_type": action_type, "value": value}})


def safe(text: str, limit: int = 999) -> str:
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode()[:limit]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = """You are an Indian Government Welfare Officer AI agent.
Evaluate applicants and enroll them in the correct welfare scheme.

Available actions — respond ONLY with valid JSON, nothing else:
1. {"action_type": "ask_question", "value": "<field_name>"}
   field_name must be one of: age, income, occupation, has_aadhaar
2. {"action_type": "request_document", "value": "<doc_name>"}
3. {"action_type": "approve_scheme", "value": "<scheme_name>"}
   scheme options: PMKVY, MGNREGS, PMAY
4. {"action_type": "reject_applicant", "value": "<reason>"}
5. {"action_type": "escalate", "value": ""}

Scheme eligibility rules:
- PMKVY: age 18-35, skilled worker (mason/carpenter), income < 10000
- MGNREGS: age 18-60, rural labourer (farm_labourer), has_aadhaar = True
- PMAY: age 21-55, any occupation, income < 6000, has_aadhaar = True
- REJECT if: age < 18, income > 100000, or no scheme matches

Rules:
- If missing_data is NOT empty, pick EXACTLY ONE field from the list and use ask_question. Do not try to ask for everything at once.
- If missing_data is empty, make a final decision immediately (approve_scheme or reject_applicant).
- If age < 18 or income > 100000, use reject_applicant immediately.
- Never ask for info already in known_profile.
- Always pick the MOST optimal scheme.

Respond ONLY with a JSON object. No explanation. No markdown. Just JSON."""


def get_agent_action(observation: dict, history: list):
    obs_text = (
        f"known_profile: {observation.get('known_profile', {})}\n"
        f"missing_data: {observation.get('missing_data', [])}\n"
        f"notification: {safe(observation.get('notification', ''))}\n"
        f"is_terminated: {observation.get('is_terminated', False)}\n"
        f"What is your next action? Respond with JSON only."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-8:]
    messages.append({"role": "user", "content": obs_text})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=100,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        action = {"action_type": "escalate", "value": ""}

    return action, raw


def run_episode(task: int) -> float:
    task_names = {
        1: "scheme_discovery",
        2: "missing_data",
        3: "conflict_resolution",
    }
    task_name = task_names[task]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    result = env_reset(task)
    obs = result.get("observation", result)
    
    grader_score = 0.0
    rewards = []
    history = []
    step = 0

    print(f"\n{'='*60}", flush=True)
    print(f"  TASK {task}/3 — {task_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Profile : {obs.get('known_profile', {})}", flush=True)
    print(f"  Missing : {obs.get('missing_data', [])}", flush=True)
    print(f"  Notif   : {safe(obs.get('notification', ''), 90)}", flush=True)

    while step < 12:
        step += 1

        if obs.get("is_terminated", False):
            grader_score = obs.get("metadata", {}).get("grader_score", 0.0)
            break

        action, raw_response = get_agent_action(obs, history)
        action_type = action.get("action_type", "escalate")
        value = action.get("value", "") or ""

        history.append({"role": "assistant", "content": raw_response})

        step_result = env_step(action_type, value)
        
        obs = step_result.get("observation", step_result)
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        notification = safe(obs.get("notification", ""))

        rewards.append(reward)

        action_str = f"{action_type}({value!r})"
        log_step(step=step, action=action_str, reward=reward, done=done, error="")

        print(f"  Step {step:02d}: {action_str} -> reward={reward}, done={done}", flush=True)
        print(f"           {notification[:80]}", flush=True)

        history.append({
            "role": "user",
            "content": f"reward={reward}, notification={notification}"
        })

        if done:
            grader_score = obs.get("metadata", {}).get("grader_score", None)
            if grader_score is None:
                if reward == 10.0:
                    grader_score = 1.0
                elif reward == 5.0:
                    grader_score = 1.0
                elif reward == 3.0:
                    grader_score = 0.5
                else:
                    grader_score = 0.0
            break

        time.sleep(0.3)

    success = grader_score >= 1.0
    log_end(success=success, steps=step, score=grader_score, rewards=rewards)

    print(f"\n  GRADER SCORE: {grader_score:.1f} / 1.0", flush=True)
    return grader_score


def main():
    print("\n" + "=" * 60, flush=True)
    print("  SCHEME ENV — INFERENCE EVALUATION", flush=True)
    print(f"  Model : {MODEL_NAME}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print("=" * 60, flush=True)

    scores = {}
    for task in [1, 2, 3]:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"\n  [ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.0
        time.sleep(1)

    print("\n" + "=" * 60, flush=True)
    print("  FINAL GRADER SCORES", flush=True)
    print("=" * 60, flush=True)
    print(f"  Task 1 (Scheme Discovery)    : {scores[1]:.1f} / 1.0", flush=True)
    print(f"  Task 2 (Missing Data)        : {scores[2]:.1f} / 1.0", flush=True)
    print(f"  Task 3 (Conflict Resolution) : {scores[3]:.1f} / 1.0", flush=True)
    print(f"  Average                      : {sum(scores.values())/3:.2f} / 1.0", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()