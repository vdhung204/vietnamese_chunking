"""Provider-independent generation interface; no credentials needed for retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 512


class LLMClient(ABC):
    def __init__(self, config: GenerationConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate using self.config; provider adapters implement this method."""


class OllamaClient(LLMClient):
    """Local Ollama HTTP adapter. Construction never contacts or starts Ollama."""

    def __init__(self, config=None, base_url='http://localhost:11434', timeout=180):
        super().__init__(config or GenerationConfig('qwen2.5:3b'))
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = dict(model=self.config.model, prompt=prompt, stream=False,
                       options=dict(temperature=self.config.temperature,
                                    num_predict=self.config.max_tokens))
        request = Request(self.base_url + '/api/generate',
                          data=json.dumps(payload).encode('utf-8'),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:1000]
            raise RuntimeError(f'Ollama HTTP {exc.code}: {detail}') from exc
        except (TimeoutError, URLError) as exc:
            raise RuntimeError(
                f'Không kết nối được hoặc Ollama hết thời gian chờ tại {self.base_url}. '
                'Hãy kiểm tra Ollama đang chạy và model đã được pull.') from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError('Ollama trả về dữ liệu JSON không hợp lệ.') from exc
        if not isinstance(result, dict) or not isinstance(result.get('response'), str):
            raise RuntimeError(f'Ollama không trả về câu trả lời hợp lệ: {result}')
        if result.get('error'):
            raise RuntimeError(f"Ollama: {result['error']}")
        return result['response']
