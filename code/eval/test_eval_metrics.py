"""
Unit Tests for Evaluation Metrics

Tests ROUGE-L and A-ESR implementations with hard-coded toy sequences
to verify correctness.

Run tests:
    python -m pytest test_eval_metrics.py -v
    # or
    python test_eval_metrics.py
"""

import unittest
import sys
import os

# Add parent directory to path to import metrics module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import RougeL, ExtractionSuccessRate


class TestRougeL(unittest.TestCase):
    """Test cases for ROUGE-L metric."""

    def test_exact_match(self):
        """Test ROUGE-L with exact match (should be 1.0)."""
        candidate = "the quick brown fox jumps over the lazy dog"
        reference = "the quick brown fox jumps over the lazy dog"

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['precision'], 1.0, places=4,
                               msg="Exact match should have precision 1.0")
        self.assertAlmostEqual(result['recall'], 1.0, places=4,
                               msg="Exact match should have recall 1.0")
        self.assertAlmostEqual(result['f1'], 1.0, places=4,
                               msg="Exact match should have F1 1.0")

    def test_no_overlap(self):
        """Test ROUGE-L with no overlap (should be 0.0)."""
        candidate = "apple banana cherry"
        reference = "dog elephant frog"

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['precision'], 0.0, places=4,
                               msg="No overlap should have precision 0.0")
        self.assertAlmostEqual(result['recall'], 0.0, places=4,
                               msg="No overlap should have recall 0.0")
        self.assertAlmostEqual(result['f1'], 0.0, places=4,
                               msg="No overlap should have F1 0.0")

    def test_partial_overlap(self):
        """Test ROUGE-L with partial overlap."""
        candidate = "the quick brown fox"
        reference = "the quick red fox"

        result = RougeL.compute(candidate, reference)

        # LCS: "the quick fox" (3 tokens)
        # Precision: 3/4 = 0.75
        # Recall: 3/4 = 0.75
        # F1: 0.75
        self.assertAlmostEqual(result['precision'], 0.75, places=4,
                               msg="Partial overlap precision should be 0.75")
        self.assertAlmostEqual(result['recall'], 0.75, places=4,
                               msg="Partial overlap recall should be 0.75")
        self.assertAlmostEqual(result['f1'], 0.75, places=4,
                               msg="Partial overlap F1 should be 0.75")

    def test_subset_candidate(self):
        """Test when candidate is subset of reference."""
        candidate = "quick brown fox"
        reference = "the quick brown fox jumps"

        result = RougeL.compute(candidate, reference)

        # LCS: "quick brown fox" (3 tokens)
        # Precision: 3/3 = 1.0
        # Recall: 3/5 = 0.6
        # F1: 2 * 1.0 * 0.6 / (1.0 + 0.6) = 0.75
        self.assertAlmostEqual(result['precision'], 1.0, places=4,
                               msg="Subset candidate should have precision 1.0")
        self.assertAlmostEqual(result['recall'], 0.6, places=4,
                               msg="Subset candidate should have recall 0.6")
        self.assertAlmostEqual(result['f1'], 0.75, places=4,
                               msg="Subset candidate F1 should be 0.75")

    def test_subset_reference(self):
        """Test when reference is subset of candidate."""
        candidate = "the quick brown fox jumps"
        reference = "quick brown fox"

        result = RougeL.compute(candidate, reference)

        # LCS: "quick brown fox" (3 tokens)
        # Precision: 3/5 = 0.6
        # Recall: 3/3 = 1.0
        # F1: 2 * 0.6 * 1.0 / (0.6 + 1.0) = 0.75
        self.assertAlmostEqual(result['precision'], 0.6, places=4,
                               msg="Subset reference should have precision 0.6")
        self.assertAlmostEqual(result['recall'], 1.0, places=4,
                               msg="Subset reference should have recall 1.0")
        self.assertAlmostEqual(result['f1'], 0.75, places=4,
                               msg="Subset reference F1 should be 0.75")

    def test_case_insensitive(self):
        """Test that ROUGE-L is case-insensitive."""
        candidate = "The Quick Brown Fox"
        reference = "the quick brown fox"

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['f1'], 1.0, places=4,
                               msg="ROUGE-L should be case-insensitive")

    def test_empty_candidate(self):
        """Test with empty candidate."""
        candidate = ""
        reference = "the quick brown fox"

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['precision'], 0.0, places=4)
        self.assertAlmostEqual(result['recall'], 0.0, places=4)
        self.assertAlmostEqual(result['f1'], 0.0, places=4)

    def test_empty_reference(self):
        """Test with empty reference."""
        candidate = "the quick brown fox"
        reference = ""

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['precision'], 0.0, places=4)
        self.assertAlmostEqual(result['recall'], 0.0, places=4)
        self.assertAlmostEqual(result['f1'], 0.0, places=4)

    def test_both_empty(self):
        """Test with both empty."""
        candidate = ""
        reference = ""

        result = RougeL.compute(candidate, reference)

        self.assertAlmostEqual(result['precision'], 0.0, places=4)
        self.assertAlmostEqual(result['recall'], 0.0, places=4)
        self.assertAlmostEqual(result['f1'], 0.0, places=4)

    def test_reordered_tokens(self):
        """Test with reordered tokens (LCS should find longest match)."""
        candidate = "fox brown quick the"
        reference = "the quick brown fox"

        result = RougeL.compute(candidate, reference)

        # LCS: "fox" (1 token) - only one token in order
        # Precision: 1/4 = 0.25
        # Recall: 1/4 = 0.25
        # F1: 0.25
        self.assertAlmostEqual(result['precision'], 0.25, places=4,
                               msg="Reordered tokens precision should be 0.25")
        self.assertAlmostEqual(result['recall'], 0.25, places=4,
                               msg="Reordered tokens recall should be 0.25")
        self.assertAlmostEqual(result['f1'], 0.25, places=4,
                               msg="Reordered tokens F1 should be 0.25")

    def test_medical_example(self):
        """Test with realistic medical text example."""
        candidate = "Patient presents with chest pain radiating to left arm"
        reference = "Patient presents with severe chest pain radiating to the left arm"

        result = RougeL.compute(candidate, reference)

        # LCS: "Patient presents with chest pain radiating to left arm" (9 tokens)
        # Candidate tokens: 9, Reference tokens: 11
        # Precision: 9/9 = 1.0
        # Recall: 9/11 ≈ 0.818
        # F1: 2 * 1.0 * 0.818 / (1.0 + 0.818) ≈ 0.9
        self.assertAlmostEqual(result['precision'], 1.0, places=2)
        self.assertGreater(result['recall'], 0.80)
        self.assertLess(result['recall'], 0.85)
        self.assertGreater(result['f1'], 0.88)
        self.assertLess(result['f1'], 0.92)


class TestExtractionSuccessRate(unittest.TestCase):
    """Test cases for A-ESR metric."""

    def test_all_above_threshold(self):
        """Test when all scores are above threshold."""
        rouge_scores = [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        self.assertAlmostEqual(result['a_esr'], 1.0, places=4,
                               msg="All scores above threshold should give A-ESR 1.0")
        self.assertEqual(result['success_count'], 6,
                         msg="All 6 samples should be successful")
        self.assertEqual(result['total_count'], 6)

    def test_all_below_threshold(self):
        """Test when all scores are below threshold."""
        rouge_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        self.assertAlmostEqual(result['a_esr'], 0.0, places=4,
                               msg="All scores below threshold should give A-ESR 0.0")
        self.assertEqual(result['success_count'], 0,
                         msg="No samples should be successful")
        self.assertEqual(result['total_count'], 5)

    def test_mixed_scores(self):
        """Test with mixed scores."""
        rouge_scores = [0.85, 0.90, 0.95, 0.88, 0.92, 0.87, 0.93, 0.89, 0.91, 0.94]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        # Scores >= 0.9: 0.90, 0.95, 0.92, 0.93, 0.91, 0.94 (6 out of 10)
        expected_a_esr = 6 / 10
        self.assertAlmostEqual(result['a_esr'], expected_a_esr, places=4,
                               msg="A-ESR should be 0.6 for 6/10 success")
        self.assertEqual(result['success_count'], 6)
        self.assertEqual(result['total_count'], 10)

    def test_threshold_1_0(self):
        """Test A-ESR with threshold 1.0 (exact match)."""
        rouge_scores = [0.95, 0.99, 1.0, 0.98, 1.0, 0.97]
        threshold = 1.0

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        # Only scores == 1.0: 1.0, 1.0 (2 out of 6)
        expected_a_esr = 2 / 6
        self.assertAlmostEqual(result['a_esr'], expected_a_esr, places=4,
                               msg="A-ESR(1.0) should count only exact matches")
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['total_count'], 6)

    def test_threshold_0_9(self):
        """Test A-ESR with threshold 0.9."""
        rouge_scores = [0.89, 0.90, 0.91, 0.88, 0.92]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        # Scores >= 0.9: 0.90, 0.91, 0.92 (3 out of 5)
        expected_a_esr = 3 / 5
        self.assertAlmostEqual(result['a_esr'], expected_a_esr, places=4)
        self.assertEqual(result['success_count'], 3)
        self.assertEqual(result['total_count'], 5)

    def test_empty_scores(self):
        """Test with empty score list."""
        rouge_scores = []
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        self.assertAlmostEqual(result['a_esr'], 0.0, places=4)
        self.assertEqual(result['success_count'], 0)
        self.assertEqual(result['total_count'], 0)

    def test_single_score_above(self):
        """Test with single score above threshold."""
        rouge_scores = [0.95]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        self.assertAlmostEqual(result['a_esr'], 1.0, places=4)
        self.assertEqual(result['success_count'], 1)
        self.assertEqual(result['total_count'], 1)

    def test_single_score_below(self):
        """Test with single score below threshold."""
        rouge_scores = [0.85]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        self.assertAlmostEqual(result['a_esr'], 0.0, places=4)
        self.assertEqual(result['success_count'], 0)
        self.assertEqual(result['total_count'], 1)

    def test_boundary_value(self):
        """Test with score exactly at threshold."""
        rouge_scores = [0.9]
        threshold = 0.9

        result = ExtractionSuccessRate.compute(rouge_scores, threshold)

        # Score == threshold should be counted as success
        self.assertAlmostEqual(result['a_esr'], 1.0, places=4,
                               msg="Score at threshold should be counted as success")
        self.assertEqual(result['success_count'], 1)
        self.assertEqual(result['total_count'], 1)

    def test_multiple_thresholds(self):
        """Test same scores with different thresholds."""
        rouge_scores = [0.7, 0.8, 0.9, 0.95, 1.0]

        # Threshold 0.7
        result_07 = ExtractionSuccessRate.compute(rouge_scores, 0.7)
        self.assertEqual(result_07['success_count'], 5)  # All pass

        # Threshold 0.8
        result_08 = ExtractionSuccessRate.compute(rouge_scores, 0.8)
        self.assertEqual(result_08['success_count'], 4)  # 0.8, 0.9, 0.95, 1.0

        # Threshold 0.9
        result_09 = ExtractionSuccessRate.compute(rouge_scores, 0.9)
        self.assertEqual(result_09['success_count'], 3)  # 0.9, 0.95, 1.0

        # Threshold 1.0
        result_10 = ExtractionSuccessRate.compute(rouge_scores, 1.0)
        self.assertEqual(result_10['success_count'], 1)  # Only 1.0


class TestIntegration(unittest.TestCase):
    """Integration tests combining ROUGE-L and A-ESR."""

    def test_extraction_success_pipeline(self):
        """Test complete evaluation pipeline."""
        # Simulated extraction results
        test_cases = [
            {
                'generated': 'The patient presents with acute chest pain',
                'reference': 'The patient presents with acute chest pain'
            },
            {
                'generated': 'The patient presents with chest pain',
                'reference': 'The patient presents with acute chest pain radiating to left arm'
            },
            {
                'generated': 'Patient has fever and cough',
                'reference': 'The patient presents with acute chest pain'
            }
        ]

        # Compute ROUGE-L scores
        rouge_scores = []
        for case in test_cases:
            result = RougeL.compute(case['generated'], case['reference'])
            rouge_scores.append(result['f1'])

        # Verify individual scores
        self.assertAlmostEqual(rouge_scores[0], 1.0, places=4,
                               msg="Exact match should have F1 1.0")
        self.assertGreater(rouge_scores[1], 0.5,
                           msg="Partial match should have F1 > 0.5")
        self.assertLess(rouge_scores[2], 0.3,
                        msg="No match should have F1 < 0.3")

        # Compute A-ESR(0.9)
        a_esr_09 = ExtractionSuccessRate.compute(rouge_scores, 0.9)
        # Only first case should pass
        self.assertEqual(a_esr_09['success_count'], 1,
                         msg="Only exact match should pass A-ESR(0.9)")

        # Compute A-ESR(1.0)
        a_esr_10 = ExtractionSuccessRate.compute(rouge_scores, 1.0)
        # Only first case should pass
        self.assertEqual(a_esr_10['success_count'], 1,
                         msg="Only exact match should pass A-ESR(1.0)")


def run_tests():
    """Run all tests and print results."""
    print("=" * 80)
    print("UNIT TESTS FOR EVALUATION METRICS")
    print("=" * 80)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRougeL))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractionSuccessRate))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print()
        print("✓ ALL TESTS PASSED!")
        print()
        print("Correctness proof: All evaluation metrics are working as expected.")
        print("- ROUGE-L correctly computes LCS-based similarity")
        print("- A-ESR correctly computes extraction success rates")
        print("- Edge cases handled properly (empty strings, exact matches, etc.)")
    else:
        print()
        print("✗ SOME TESTS FAILED")
        print("Please review the failures above.")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
