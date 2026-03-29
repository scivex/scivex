"""Progress bar utilities — works with or without tqdm."""
import sys
import time


class ProgressBar:
    """Simple text-based progress bar (no dependencies).

    Args:
        total: Total number of steps.
        desc: Description label.
        width: Bar width in characters.
    """

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self._start = time.perf_counter()

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by n steps."""
        self.current = min(self.current + n, self.total)
        self._render()

    def _render(self) -> None:
        frac = self.current / self.total if self.total > 0 else 1.0
        filled = int(self.width * frac)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._start
        prefix = f"{self.desc}: " if self.desc else ""
        sys.stderr.write(
            f"\r{prefix}[{bar}] {self.current}/{self.total} ({elapsed:.1f}s)"
        )
        if self.current >= self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def close(self) -> None:
        """Finish the progress bar."""
        if self.current < self.total:
            self.current = self.total
            self._render()


def track(iterable, desc: str = "", total: int = None):
    """Iterate with a progress bar.

    Auto-detects tqdm if installed, otherwise uses the built-in ProgressBar.

    Args:
        iterable: Any iterable to wrap.
        desc: Description label.
        total: Total count (inferred from iterable if possible).

    Yields:
        Items from the iterable.
    """
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, desc=desc, total=total)
        return
    except ImportError:
        pass

    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            # Can't determine length; iterate without progress
            yield from iterable
            return

    bar = ProgressBar(total, desc=desc)
    for item in iterable:
        yield item
        bar.update()
    bar.close()
