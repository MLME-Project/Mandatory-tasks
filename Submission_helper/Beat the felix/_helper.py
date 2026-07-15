import time
import requests

'''
You move functions or classes to seperate scripts and
import them into the main script. This is a good practice to 
keep your code organized and maintainable. In this 
case, the BioreactorClient class is defined in a separate 
script and can be imported into other scripts as needed.
'''


BASE_URL = "https://mlme26biosim.org"



class BioreactorClient:
    """Cookie-based session wrapper around the lab's REST API."""

    def __init__(self, base_url: str = BASE_URL):
        self.s = requests.Session()
        self.base = base_url.rstrip("/")

    def login(self, user: str, password: str) -> None:
        r = self.s.post(
            f"{self.base}/api/login",
            json={"user": user, "password": password},
            timeout=15,
        )
        r.raise_for_status()

    def _csrf(self) -> str:
        token = self.s.cookies.get("mlme26_csrf")
        if not token:
            raise RuntimeError("no CSRF cookie set — call login() first")
        return token

    def run(self, scale: str, T: float, pH: float,
            F1: float, F2: float, F3: float) -> dict:
        """Submit one experiment and return the server's response dict.

        The response contains at least "Y" (the measured yield) and
        "cost_eur" (what the experiment cost). Network/rate-limit hiccups are
        retried with exponential backoff so a single trial isn't lost.
        """
        payload = {
            "scale": scale,
            "recipe": {"T": T, "pH": pH, "F1": F1, "F2": F2, "F3": F3},
        }
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                r = self.s.post(
                    f"{self.base}/api/run", json=payload,
                    headers={"X-CSRF-Token": self._csrf()},
                    timeout=60,
                )
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e
                wait = 2.5 ** attempt
                print(f"  [network error: {type(e).__name__}, "
                      f"sleeping {wait:.1f}s]", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 429:
                wait = 2.5 ** attempt
                print(f"  [rate-limited, sleeping {wait:.1f}s]", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 402:
                raise RuntimeError(
                    f"group budget exhausted: {r.json().get('detail')}"
                )
            if r.status_code >= 500:
                wait = 2.5 ** attempt
                print(f"  [server {r.status_code}, sleeping {wait:.1f}s]",
                      flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(
            f"too many failed attempts (last error: {last_err!r})"
        )

    def history(self) -> list[dict]:
        r = self.s.get(f"{self.base}/api/history", timeout=30)
        r.raise_for_status()
        return r.json()["items"]

