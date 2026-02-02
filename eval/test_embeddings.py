
import unittest
from retrieval.embeddings import hash_embed, batch_cosine

class TestEmbeddings(unittest.TestCase):
    def test_dimensions(self):
        vec = hash_embed("hello world")
        self.assertEqual(len(vec), 4096)
        
    def test_determinism(self):
        v1 = hash_embed("foo bar")
        v2 = hash_embed("foo bar")
        self.assertEqual(v1, v2)
        
    def test_similarity(self):
        # "foo bar" should be similar to "bar foo" (bigrams might lower it slightly but bag of words is high)
        v1 = hash_embed("foo bar")
        v2 = hash_embed("bar foo")
        # cosine should be high
        sim = batch_cosine(v1, [v2])[0]
        self.assertGreater(sim, 0.8)
        
    def test_dissimilarity(self):
        v1 = hash_embed("apple banana")
        v2 = hash_embed("car truck")
        sim = batch_cosine(v1, [v2])[0]
        self.assertLess(sim, 0.3)
        
    def test_bigram_effect(self):
        # "ab cd" vs "ab cd" -> 1.0
        # "ab cd" vs "cd ab" -> should be less than 1.0 because bigrams mismatch ("ab_cd" vs "cd_ab")
        v1 = hash_embed("ab cd")
        v2 = hash_embed("cd ab")
        sim = batch_cosine(v1, [v2])[0]
        self.assertLess(sim, 0.99)
        self.assertGreater(sim, 0.5)

if __name__ == '__main__':
    unittest.main()
