"""Source-specific evidence coverage, scored only AFTER retrieval."""

import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def contains_evidence(chunk, question):
    evidence = normalize(question['gold_evidence'])
    return bool(evidence) and chunk['document_id'] == question['document_id'] and (
        evidence in normalize(chunk['text']))


def validate_benchmark(documents, questions):
    if not documents or not questions:
        raise ValueError("Benchmark requires documents and questions")
    sources = {}
    for doc in documents:
        if not isinstance(doc['id'], str) or not doc['id'] or doc['id'] in sources or not doc['text'].strip():
            raise ValueError("Invalid or duplicate document")
        sources[doc['id']] = doc['text']
    seen = set()
    for question in questions:
        if not question['id'] or question['id'] in seen or not question['question'].strip():
            raise ValueError("Invalid or duplicate question")
        seen.add(question['id'])
        source = sources.get(question['document_id'])
        if source is None or not contains_evidence(
            dict(document_id=question['document_id'], text=source), question
        ):
            raise ValueError(f"Evidence absent from labeled source: {question['id']}")


def evaluate_retrieval(questions, rankings, chunks):
    if not questions or len(questions) != len(rankings):
        raise ValueError("Nonempty questions and aligned rankings required")
    details = []
    for question, ranked in zip(questions, rankings):
        first_rank = next((i for i, chunk in enumerate(ranked[:5], 1)
                           if contains_evidence(chunk, question)), None)
        preserved = any(contains_evidence(chunk, question) for chunk in chunks)
        details.append(dict(
            question_id=question['id'], question=question['question'],
            document_id=question['document_id'], gold_evidence=question['gold_evidence'],
            hit_at_3=first_rank is not None and first_rank <= 3,
            hit_at_5=first_rank is not None,
            reciprocal_rank_at_5=1 / first_rank if first_rank else 0.,
            evidence_preserved=preserved,
            failure_reason=None if first_rank else ('ranking_miss' if preserved else 'evidence_split'),
            context_words=sum(len(chunk['text'].split()) for chunk in ranked),
            retrieved=ranked))
    count = len(details)
    return dict(
        recall_at_3=sum(d['hit_at_3'] for d in details) / count,
        recall_at_5=sum(d['hit_at_5'] for d in details) / count,
        mrr_at_5=sum(d['reciprocal_rank_at_5'] for d in details) / count,
        evidence_preservation_rate=sum(d['evidence_preserved'] for d in details) / count,
        mean_context_words=sum(d['context_words'] for d in details) / count,
        details=details)
