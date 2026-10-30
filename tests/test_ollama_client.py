import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rag.llm_client import OllamaClient


class OllamaTests(unittest.TestCase):
    @patch('rag.llm_client.urlopen')
    def test_request_and_response(self, open_url):
        client = OllamaClient()
        open_url.assert_not_called()
        response = MagicMock()
        response.read.return_value = json.dumps({'response': 'Answer', 'done': True}).encode()
        open_url.return_value.__enter__.return_value = response
        self.assertEqual(client.generate('Question'), 'Answer')
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:11434/api/generate')
        self.assertEqual(json.loads(request.data), dict(model='qwen2.5:3b', prompt='Question',
            stream=False, options=dict(temperature=0., num_predict=512)))

    @patch('rag.llm_client.urlopen', side_effect=URLError('offline'))
    def test_offline(self, open_url):
        with self.assertRaises(RuntimeError):
            OllamaClient().generate('Question')

    @patch('rag.llm_client.urlopen')
    def test_malformed_response(self, open_url):
        open_url.return_value.__enter__.return_value.read.return_value = b'{}'
        with self.assertRaises(RuntimeError):
            OllamaClient().generate('Question')
