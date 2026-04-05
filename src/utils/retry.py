import random
import time


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "timed out" in str(e) \
                                    or "Too Many Requests" in str(e) or "Forbidden for url" in str(e) \
                                    or "internal" in str(e).lower():
                    num_retries += 1

                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None

                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    time.sleep(delay)
                else:
                    print(str(e))
                    return None

    return wrapper
