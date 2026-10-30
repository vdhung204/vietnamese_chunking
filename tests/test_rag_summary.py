import unittest

from evaluation.summarize_rag import summarize


class SummaryTests(unittest.TestCase):
    def test_paired_cluster_intervals_and_duplicate_sensitivity(self):
        questions = [dict(id=str(i), article_id=str(i // 2), domain='test',
                          question='repeat' if i < 2 else str(i)) for i in range(4)]
        def records(hit):
            return [dict(question_id=str(i), hit_at_3=hit, hit_at_5=hit,
                         reciprocal_rank_at_5=float(hit), failure_reason=None if hit else 'ranking_miss')
                    for i in range(4)]
        report = dict(results=[dict(method='first', details=records(False)),
                               dict(method='second', details=records(True))])
        result = summarize(report, questions, repetitions=20)
        self.assertEqual(result['source_titles'], 2)
        self.assertEqual(result['metrics']['recall_at_5']['delta'], 1)
        self.assertEqual(result['metrics']['recall_at_5']['delta_ci95'], [1, 1])
        self.assertEqual(result['paired_hit_counts_at_5']['only_second'], 4)
        self.assertEqual(result['deduplicated_sensitivity']['questions'], 3)
        report['results'][1]['details'] = records(False)
        result = summarize(report, questions, repetitions=20)
        self.assertEqual(result['metrics']['recall_at_5']['delta_ci95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
