"""News article corpus processor."""


class CorpusProcessor:
    """Process and prepare news articles for indexing."""

    def __init__(self, corpus_dir: str = "./data/corpus"):
        self.corpus_dir = corpus_dir

    def process_cnn_dailymail(self):
        """Process CNN/DailyMail articles."""
        raise NotImplementedError

    def process_ccnews(self):
        """Process CC-News articles."""
        raise NotImplementedError
