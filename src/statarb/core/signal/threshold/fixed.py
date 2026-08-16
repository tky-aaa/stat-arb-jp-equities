class FixedThreshold:
    def __init__(
        self,
        entry_threshold: float,
    ) -> None:
        if entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive.")

        self.entry_threshold = entry_threshold

    def optimize(self) -> float:
        return self.entry_threshold
