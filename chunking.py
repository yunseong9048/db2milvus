"""논문 본문을 원문 위치를 보존하며 나누는 모듈. DB 접속이나 임베딩은 처리하지 않는다.

호출 흐름: migrate.py → make_chunks() → spans()
"""

import bisect
import logging

LOG = logging.getLogger(__name__)


def section_sources(paper):
    """(분할할 본문, full_text 기준 시작 위치)를 섹션 순서대로 반환한다.

    원문과 정확히 대응하지 않는 섹션의 위치는 -1이다. 원문을 변형하거나
    sections를 합쳐 가짜 full_text 위치를 만들지 않는다.
    """
    full_text = paper['full_text'] or ''
    sections = sorted(paper.get('sections') or [], key=lambda row: row['seq'])
    sections = [row for row in sections if (row.get('text') or '').strip()]
    if not sections:
        yield full_text, 0
        return
    cursor = 0
    for section in sections:
        text = section['text']
        # 순서대로 찾으면 같은 문장이 여러 섹션에 있어도 앞부분을 재사용하지 않는다.
        start = full_text.find(text, cursor)
        if start >= 0:
            cursor = start + len(text)
        else:
            LOG.warning('paper_id=%s section_seq=%s: full_text 위치 미상',
                        paper['paper_id'], section['seq'])
        yield text, start


def spans(text, size=1800, overlap=200):
    """본문에서 청크별 (시작 위치, 끝 위치)를 차례로 반환한다.

    size는 최대 문자 수, overlap은 이웃 청크 사이에 겹칠 문자 수다.
    끝 위치의 문자는 포함하지 않으므로 text[start:end]로 본문을 얻는다.
    """
    # 한글 등은 한 문자에 여러 바이트를 쓰므로 Milvus의 바이트 제한도 고려한다.
    if not 0 <= overlap < size or size > 15000:
        raise ValueError('Require 0 <= overlap < chunk size <= 15000')
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 너무 짧게 자르지 않도록 청크 뒤쪽에서 경계를 찾는다.
            floor = start + max(overlap + 1, size // 2)
            # 문단 → 줄바꿈 → 마침표 뒤 공백 → 공백 순으로 자를 위치를 찾는다.
            for separator in ('\n\n', '\n', '. ', ' '):
                cut = text.rfind(separator, floor, end)
                if cut >= floor:
                    end = cut + len(separator)
                    break
        yield start, end  # 원문은 수정하지 않고 위치만 전달한다.
        if end == len(text):
            break
        # 앞 청크의 끝부분을 다음 청크에도 넣어 문맥이 끊기는 것을 줄인다.
        start = end - overlap


def make_chunks(paper, size, overlap):
    """논문 한 행(dict)을 받아 Milvus에 넣을 청크 dict를 차례로 반환한다.

    paper에는 full_text, paper_id, year, has_experiments,
    precursor_ids, central_metals가 필요하다. sections가 있으면 seq 순서로
    각 행의 text를 따로 청킹한다. 섹션 사이에는 겹침을 적용하지 않는다.
    embedding과 저장 시각은 migrate.py에서 나중에 추가한다.
    """
    text = paper['full_text'] or ''
    # 스키마 제한을 넘는 메타데이터는 임의로 잘라 버리지 않고 오류로 알린다.
    if len(paper['precursor_ids']) > 16 or len(paper['central_metals']) > 16:
        raise ValueError(f"paper {paper['paper_id']}: ARRAY capacity 16 exceeded")
    if any(len(m.encode('utf-8')) > 8 for m in paper['central_metals']):
        raise ValueError(f"paper {paper['paper_id']}: central_metal exceeds 8 bytes")
    # 원문의 페이지 구분 문자(\f)가 있는 경우에만 페이지 번호를 계산한다.
    # paper_sections.para_pages만으로는 full_text의 정확한 문자 위치를 알 수 없다.
    breaks = [i for i, char in enumerate(text) if char == '\f']
    for seq, (body, start, end) in enumerate(section_spans(paper, size, overlap)):
        # 예: 논문 60의 첫 청크는 60-0000. 재실행해도 같은 ID를 만든다.
        chunk_id = f"{paper['paper_id']}-{seq:04d}"
        if len(chunk_id.encode()) > 32:
            raise ValueError('chunk_id exceeds VARCHAR(32)')
        # 논문/전구체 정보를 청크마다 복사해 Milvus에서 조인 없이 필터링한다.
        # bisect는 현재 문자 앞의 페이지 경계 수를 센다. 경계가 없으면 0(미상).
        yield dict(chunk_id=chunk_id, paper_id=paper['paper_id'], chunk_seq=seq,
                   text=body, char_start=start, char_end=end,
                   page_start=bisect.bisect_left(breaks, start) + 1 if breaks and start >= 0 else 0,
                   page_end=bisect.bisect_left(breaks, end - 1) + 1 if breaks and end >= 0 else 0,
                   year=paper['year'], has_experiments=paper['has_experiments'],
                   precursor_ids=paper['precursor_ids'], central_metals=paper['central_metals'])


def section_spans(paper, size, overlap):
    """섹션별 청크 본문과 원문 위치를 반환한다. 미상 위치는 양쪽 모두 -1."""
    for text, base in section_sources(paper):
        for start, end in spans(text, size, overlap):
            yield text[start:end], base + start if base >= 0 else -1, base + end if base >= 0 else -1


