import unittest

from chunking import make_chunks, spans


class ChunkTests(unittest.TestCase):
    def paper(self, text):
        return dict(paper_id=60, full_text=text, year=2018, has_experiments=True,
                    precursor_ids=[143], central_metals=['[Al]'])

    def test_offsets_coverage_and_progress(self):
        text = ('Al2O3 증착 조건.\n\n' * 200) + 'end'
        ranges = list(spans(text, 100, 20))
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], len(text))
        for (start, end), (next_start, next_end) in zip(ranges, ranges[1:]):
            self.assertLess(start, next_start)
            self.assertEqual(end - next_start, 20)
            self.assertGreater(next_end, end)
        rows = list(make_chunks(self.paper(text), 100, 20))
        for row in rows:
            self.assertEqual(row['text'], text[row['char_start']:row['char_end']])
            self.assertEqual(row['page_start'], 0)
            self.assertLessEqual(len(row['text'].encode('utf-8')), 65535)

    def test_page_boundary_and_stable_ids(self):
        paper = self.paper('abc\fdef')
        rows = list(make_chunks(paper, 4, 0))
        self.assertEqual([r['chunk_id'] for r in rows], ['60-0000', '60-0001'])
        self.assertEqual((rows[0]['page_start'], rows[0]['page_end']), (1, 1))
        self.assertEqual((rows[1]['page_start'], rows[1]['page_end']), (2, 2))
        self.assertEqual(rows, list(make_chunks(paper, 4, 0)))

    def test_empty_and_invalid(self):
        self.assertEqual(list(spans('')), [])
        for size, overlap in [(10, 10), (10, -1), (0, 0), (15001, 10)]:
            with self.assertRaises(ValueError):
                list(spans('abc', size, overlap))

    def test_sections_sorted_without_crossing_boundaries(self):
        paper = self.paper('Intro\n\nResults')
        paper['sections'] = [dict(seq=2, text='Results'), dict(seq=1, text='Intro')]
        rows = list(make_chunks(paper, 100, 20))
        self.assertEqual([r['text'] for r in rows], ['Intro', 'Results'])
        self.assertEqual([r['chunk_id'] for r in rows], ['60-0000', '60-0001'])
        self.assertEqual([r['char_start'] for r in rows], [0, 7])

    def test_unmatched_section_preserves_text_with_unknown_offset(self):
        paper = self.paper('original\ftext')
        paper['sections'] = [dict(seq=0, text='cleaned text')]
        row = list(make_chunks(paper, 100, 0))[0]
        self.assertEqual(row['text'], 'cleaned text')
        self.assertEqual((row['char_start'], row['char_end']), (-1, -1))
        self.assertEqual((row['page_start'], row['page_end']), (0, 0))

    def test_empty_sections_fall_back_to_full_text(self):
        paper = self.paper('original')
        paper['sections'] = [dict(seq=0, text='  ')]
        self.assertEqual(list(make_chunks(paper, 100, 0))[0]['text'], 'original')

    def test_repeated_section_text_uses_later_offset(self):
        paper = self.paper('same same')
        paper['sections'] = [dict(seq=0, text='same'), dict(seq=1, text='same')]
        rows = list(make_chunks(paper, 100, 0))
        self.assertEqual([r['char_start'] for r in rows], [0, 5])

    def test_capacity_not_silently_truncated(self):
        paper = self.paper('data')
        paper['precursor_ids'] = list(range(17))
        with self.assertRaises(ValueError):
            list(make_chunks(paper, 100, 0))


if __name__ == '__main__':
    unittest.main()
