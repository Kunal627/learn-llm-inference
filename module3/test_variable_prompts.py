import unittest

import torch

from tiny_llm import last_valid_logits, pad_prompts


class VariablePromptTests(unittest.TestCase):
    def test_padding_and_last_valid_logits(self):
        input_ids, mask, lengths = pad_prompts([[11, 12, 13], [21], [31, 32]], 0, "cpu", 4)
        self.assertEqual(input_ids.tolist(), [[11, 12, 13, 0], [21, 0, 0, 0], [31, 32, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        logits = torch.arange(24).reshape(3, 4, 2)
        torch.testing.assert_close(last_valid_logits(logits, lengths), logits[[0, 1, 2], [2, 0, 1]])

    def test_invalid_prompts(self):
        for prompts, max_length in [([], None), ([[]], None), ([[1, 2]], 1)]:
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                pad_prompts(prompts, 0, "cpu", max_length)


if __name__ == "__main__":
    unittest.main()