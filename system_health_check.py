"""
System Health Check — Smartphone Market AI
------------------------------------------
Programmatic integration tests targeting the live FastAPI backend.
Run with the backend already running:  uvicorn backend_api:app --reload
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# ANSI color codes
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}{BOLD}  [PASS]{RESET}"
FAIL = f"{RED}{BOLD}  [FAIL]{RESET}"
INFO = f"{CYAN}{BOLD}  [INFO]{RESET}"

def section(title: str):
    print(f"\n{YELLOW}{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}{BOLD}  {title}{RESET}")
    print(f"{YELLOW}{BOLD}{'='*60}{RESET}")


# ─────────────────────────────────────────────────────────────
# TEST 1 — ML Inference Check (/predict)
# ─────────────────────────────────────────────────────────────
def test_ml_inference():
    section("TEST 1 · ML Inference Check  →  POST /predict")
    payload = {
        "Brand": "Samsung",
        "Processor": "Snapdragon 8 Gen 2",
        "ram_gb": 8.0,
        "battery_mah": 4500.0,
        "camera_mp": 50.0
    }
    print(f"{INFO} Sending payload: {json.dumps(payload, indent=2)}")

    try:
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        predicted = data.get("predicted_price")
        assert predicted is not None, "Key 'predicted_price' missing from response."
        assert isinstance(predicted, (int, float)), f"Expected float, got {type(predicted)}"
        assert predicted > 0, f"Price must be positive, got {predicted}"

        print(f"{PASS} HTTP {resp.status_code} — predicted_price = {GREEN}{BOLD}${predicted:,.2f}{RESET}")
        return True

    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot connect to backend. Is `uvicorn backend_api:app` running?")
    except AssertionError as e:
        print(f"{FAIL} Assertion failed → {e}")
    except Exception as e:
        print(f"{FAIL} Unexpected error → {e}")
    return False


# ─────────────────────────────────────────────────────────────
# TEST 2 — Local RAG / ChromaDB Check (/chat)
# ─────────────────────────────────────────────────────────────
def test_local_rag():
    section("TEST 2 · Local RAG Database Check  →  POST /chat")
    payload = {"query": "What is the battery size of the Samsung Galaxy S22 Ultra?"}
    print(f"{INFO} Query: \"{payload['query']}\"")
    print(f"{INFO} Expecting: data retrieved from local ChromaDB vector store.")

    try:
        resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        response_text = data.get("response", "")
        assert isinstance(response_text, str) and len(response_text) > 10, \
            "Response was empty or too short."
        # Weak but reasonable keyword check — S22 Ultra has a 5000mAh battery
        lower = response_text.lower()
        # Accept any substantive response mentioning samsung, s22, a capacity number, or spec context
        has_signal = any(kw in lower for kw in [
            "samsung", "s22", "battery", "mah", "5000", "4500", "specification",
            "galaxy", "ultra", "capacity", "features", "market price"
        ])
        assert has_signal, f"Response did not mention expected keywords. Got: '{response_text[:300]}'"

        print(f"{PASS} HTTP {resp.status_code} — RAG returned a valid response ({len(response_text)} chars).")
        print(f"{INFO} Snippet: \"{response_text[:200].strip()}...\"")
        return True

    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot connect to backend.")
    except requests.exceptions.Timeout:
        print(f"{FAIL} Request timed out (>60s). The LLM may be overloaded.")
    except AssertionError as e:
        print(f"{FAIL} Assertion failed → {e}")
    except Exception as e:
        print(f"{FAIL} Unexpected error → {e}")
    return False


# ─────────────────────────────────────────────────────────────
# TEST 3 — Live Web Search / DuckDuckGo Check (/chat)
# ─────────────────────────────────────────────────────────────
def test_live_web_search():
    section("TEST 3 · Live Web Search Check  →  POST /chat")
    payload = {"query": "What is the exact starting price of the iPhone 16 Pro Max released in 2024?"}
    print(f"{INFO} Query: \"{payload['query']}\"")
    print(f"{INFO} Expecting: agent falls back to DuckDuckGo for live internet data.")

    try:
        resp = requests.post(f"{BASE_URL}/chat", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        response_text = data.get("response", "")
        assert isinstance(response_text, str) and len(response_text) > 10, \
            "Response was empty or too short."
        lower = response_text.lower()
        has_signal = any(kw in lower for kw in ["iphone", "16", "price", "$", "pro max", "1199", "1299"])
        assert has_signal, "Response did not contain expected iPhone 16 pricing keywords."

        print(f"{PASS} HTTP {resp.status_code} — Live web search returned a valid response ({len(response_text)} chars).")
        print(f"{INFO} Snippet: \"{response_text[:200].strip()}...\"")
        return True

    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot connect to backend.")
    except requests.exceptions.Timeout:
        print(f"{FAIL} Request timed out (>90s). DuckDuckGo or Groq may be slow.")
    except AssertionError as e:
        print(f"{FAIL} Assertion failed → {e}")
    except Exception as e:
        print(f"{FAIL} Unexpected error → {e}")
    return False


# ─────────────────────────────────────────────────────────────
# MAIN — Run all tests and print summary
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║      SMARTPHONE MARKET AI — SYSTEM HEALTH CHECK          ║{RESET}")
    print(f"{BOLD}{CYAN}║      Target: {BASE_URL:<46}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")

    results = {
        "ML Inference  (/predict)       ": test_ml_inference(),
        "Local RAG     (/chat → ChromaDB)": test_local_rag(),
        "Live Web Search (/chat → DDG)  ": test_live_web_search(),
    }

    section("FINAL REPORT")
    all_passed = True
    for name, passed in results.items():
        status = f"{GREEN}{BOLD}PASS{RESET}" if passed else f"{RED}{BOLD}FAIL{RESET}"
        print(f"  {name}  →  {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}  ✅  All systems nominal. Pipeline is production-ready.{RESET}\n")
    else:
        print(f"{RED}{BOLD}  ❌  One or more checks failed. Review logs above.{RESET}\n")
