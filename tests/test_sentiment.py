import unittest

from smartsignal.sentiment import score_headline


class SentimentTests(unittest.TestCase):
    def test_scores_positive_financial_language(self) -> None:
        self.assertGreater(score_headline("Company beats estimates and raises outlook"), 0)

    def test_scores_negative_financial_language(self) -> None:
        self.assertLess(score_headline("Company misses estimates after weak demand"), 0)

    def test_handles_negation(self) -> None:
        self.assertLess(score_headline("Results were not strong"), 0)


if __name__ == "__main__":
    unittest.main()

