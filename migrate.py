"""PostgreSQL 논문 데이터를 청킹하고 임베딩하여 Milvus에 적재한다.

실행 순서: 설정 읽기 → DB 조회 → 청킹 → 임베딩 → 적재 → 이전 청크 정리.
청킹 알고리즘은 같은 폴더의 chunking.py에 있다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from chunking import make_chunks, spans

LOG = logging.getLogger(__name__)

# 논문당 한 행을 읽고, 연결된 실험에서 전구체 ID와 중심 금속 목록을 모은다.
# DISTINCT로 중복을 제거하고 정렬한다. 연도가 NULL이면 0을 사용한다.
# %s는 execute()가 전달받은 조회 조건을 안전하게 바인딩하는 자리다.
SQL = """
SELECT p.paper_id, p.full_text, COALESCE(p.year, 0) AS year,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
           'seq', s.seq, 'section', s.section, 'text', s.text,
           'para_pages', s.para_pages) ORDER BY s.seq)
           FROM literature.paper_sections s WHERE s.paper_id = p.paper_id),
           '[]'::jsonb) AS sections,
       p.has_experiments,
       ARRAY(SELECT DISTINCT e.precursor_id FROM literature.experiments e
             WHERE e.paper_id=p.paper_id AND e.precursor_id IS NOT NULL
             ORDER BY e.precursor_id) AS precursor_ids,
       ARRAY(SELECT DISTINCT pr.central_metal
             FROM literature.experiments e
             JOIN literature.precursors pr USING (precursor_id)
             WHERE e.paper_id=p.paper_id AND pr.central_metal IS NOT NULL
             ORDER BY pr.central_metal) AS central_metals
FROM literature.papers p
WHERE p.paper_id > %s AND (%s::bigint IS NULL OR p.paper_id = %s)
ORDER BY p.paper_id
"""


def schema_for(client, dim, bm25):
    """PPT 12·13페이지의 필드 정의를 만든다. 실제 컬렉션 생성은 prepare() 담당.

    dim: 모델의 임베딩 차원. bm25: 키워드 검색용 희소 벡터 사용 여부.
    """
    from pymilvus import DataType as D, Function, FunctionType
    # ID는 직접 생성하고, 스키마에 정의하지 않은 필드 입력은 허용하지 않는다.
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field('chunk_id', D.VARCHAR, max_length=32, is_primary=True)
    for name in ('paper_id', 'char_start', 'char_end', 'created_at', 'updated_at'):
        schema.add_field(name, D.INT64)
    for name in ('chunk_seq', 'page_start', 'page_end', 'year'):
        schema.add_field(name, D.INT32)
    schema.add_field('text', D.VARCHAR, max_length=65535, enable_analyzer=bm25)
    schema.add_field('has_experiments', D.BOOL)
    schema.add_field('precursor_ids', D.ARRAY, element_type=D.INT64, max_capacity=16)
    schema.add_field('central_metals', D.ARRAY, element_type=D.VARCHAR,
                     max_capacity=16, max_length=8)
    schema.add_field('embedding', D.FLOAT_VECTOR, dim=dim)
    if bm25:
        # 본문을 입력하면 Milvus의 BM25 함수가 sparse 벡터를 자동 생성한다.
        schema.add_field('sparse', D.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(name='text_bm25', input_field_names=['text'],
                                    output_field_names=['sparse'], function_type=FunctionType.BM25))
    return schema


def prepare(client, name, dim, bm25, signature):
    """기존 컬렉션의 호환성을 확인하거나 새 컬렉션과 검색 인덱스를 만든다."""
    expected = schema_for(client, dim, bm25)
    if client.has_collection(name):
        # 필드 타입, 길이 제한, 벡터 차원 등이 현재 코드와 같은지 확인한다.
        info = client.describe_collection(name)
        actual = {f['name']: f for f in info['fields']}
        for field in expected.to_dict()['fields']:
            other = actual.get(field['name'])
            if other is None or other['type'] != field['type']:
                raise ValueError(f"Existing collection schema mismatch: {field['name']}")
            for key, value in field.get('params', {}).items():
                if str(other.get('params', {}).get(key)) != str(value):
                    raise ValueError(f"Existing collection parameter mismatch: {field['name']}.{key}")
        if set(actual) != {f['name'] for f in expected.to_dict()['fields']}:
            raise ValueError('Existing collection has unexpected fields; use a new collection')
        # 차원이 같아도 모델이 다를 수 있으므로 저장된 실행 설정도 비교한다.
        if info.get('description') != signature:
            raise ValueError('Embedding/chunk configuration differs; use a new collection name')
    else:
        # HNSW/COSINE: 본문 의미가 비슷한 벡터를 검색하기 위한 인덱스.
        indexes = client.prepare_index_params()
        indexes.add_index('embedding', index_type='HNSW', metric_type='COSINE',
                          params={'M': 16, 'efConstruction': 200})
        # INVERTED: 논문 ID, 연도, 전구체 등의 조건으로 청크를 필터링한다.
        for field in ('paper_id', 'year', 'has_experiments', 'precursor_ids', 'central_metals'):
            indexes.add_index(field, index_type='INVERTED')
        if bm25:
            indexes.add_index('sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25')
        client.create_collection(name, schema=expected, index_params=indexes,
                                 consistency_level='Strong', description=signature)
    client.load_collection(name)  # 기존 청크 조회와 검색을 위해 컬렉션을 로드한다.


def old_chunks(client, collection, paper_id):
    """논문의 기존 {청크 ID: 최초 생성 시각}을 읽는다.

    생성 시각 보존과 재청킹 후 남는 이전 청크 삭제에 사용한다.
    """
    result = {}
    iterator = client.query_iterator(collection_name=collection,
                                     filter=f'paper_id == {int(paper_id)}',
                                     output_fields=['chunk_id', 'created_at'], batch_size=1000,
                                     consistency_level='Strong')
    try:
        # 결과를 여러 묶음으로 읽어 한 번의 조회 건수 제한을 피한다.
        while batch := iterator.next():
            result.update((row['chunk_id'], row['created_at']) for row in batch)
    finally:
        iterator.close()
    return result


def main():
    """명령행 옵션과 .env 설정을 읽어 전체 이관을 실행하는 시작 함수."""
    # 1. 접속 정보와 모델 설정을 migrate.py 옆의 .env에서 읽는다.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name('.env'))
    parser = argparse.ArgumentParser(description=__doc__)
    # dry-run은 DB 조회·청킹만 수행한다. output은 확인용 JSONL 저장 경로다.
    parser.add_argument('--dry-run', action='store_true', help='No model download or Milvus writes')
    parser.add_argument('--output', type=Path, help='Dry-run JSONL destination')
    # 특정 논문만 처리하거나 지정한 ID 이후의 논문부터 처리한다.
    parser.add_argument('--paper-id', type=int)
    parser.add_argument('--after-paper-id', type=int, default=0)
    parser.add_argument('--limit', type=int)
    # chunk-size와 overlap은 문자 수, batch-size는 한 번에 처리할 청크 개수다.
    parser.add_argument('--chunk-size', type=int, default=1800)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--bm25', action='store_true')
    args = parser.parse_args()
    # 실제 DB에 접속하기 전에 청킹 크기와 겹침 값이 유효한지 확인한다.
    list(spans('', args.chunk_size, args.overlap))
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    if args.output and not args.dry_run:
        parser.error('--output requires --dry-run')
    dsn = os.environ.get('POSTGRES_DSN')
    if not dsn:
        parser.error('Set POSTGRES_DSN in .env')
    import psycopg
    from psycopg.rows import dict_row
    client = None
    output = args.output.open('w', encoding='utf-8') if args.output else None
    try:
        if not args.dry_run:
            # 2. 임베딩 모델과 Milvus 연결을 준비한다. dry-run에서는 생략한다.
            from sentence_transformers import SentenceTransformer
            from pymilvus import MilvusClient
            model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
            revision = os.getenv('EMBEDDING_REVISION') or None
            model = SentenceTransformer(model_name, revision=revision,
                                        device=os.getenv('EMBEDDING_DEVICE', 'cpu'))
            collection = os.getenv('MILVUS_COLLECTION', 'paper_chunks')
            client = MilvusClient(uri=os.getenv('MILVUS_URI', 'http://localhost:19530'),
                                  token=os.getenv('MILVUS_TOKEN', ''))
            # 실행 설정을 컬렉션 설명에 기록해 서로 다른 모델/청킹 설정 혼용을 막는다.
            signature = json.dumps(dict(model=model_name, revision=revision,
                                        chunk_size=args.chunk_size, overlap=args.overlap,
                                        version=2, bm25=args.bm25), sort_keys=True)
            prepare(client, collection, model.get_sentence_embedding_dimension(), args.bm25, signature)
        total = 0
        # 3. 원본 DB는 읽기 전용으로 사용하고, 실행 중 일관된 스냅샷을 읽는다.
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            # 서버 커서를 사용해 모든 논문 본문을 한꺼번에 메모리에 올리지 않는다.
            with conn.cursor(name='paper_stream') as cur:
                query = SQL + (' LIMIT %s' if args.limit else '')
                params = [args.after_paper_id, args.paper_id, args.paper_id]
                if args.limit:
                    params.append(args.limit)
                cur.execute(query, params)
                for paper in cur:
                    # 4. 논문 한 편을 청크로 나눈다. 구현은 chunking.py 참조.
                    chunks = list(make_chunks(paper, args.chunk_size, args.overlap))
                    if args.dry_run:
                        # 벡터 없이 본문, 원문 위치, 메타데이터만 파일에 저장한다.
                        if output:
                            for row in chunks:
                                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    else:
                        # 기존 ID와 생성 시각을 읽어 재실행 시 최초 생성 시각을 보존한다.
                        previous = old_chunks(client, collection, paper['paper_id'])
                        now = time.time_ns() // 1_000_000
                        for offset in range(0, len(chunks), args.batch_size):
                            # 메모리 사용량과 요청 크기를 줄이기 위해 묶음 단위로 처리한다.
                            batch = chunks[offset:offset + args.batch_size]
                            texts = [row['text'] for row in batch]
                            # 모델 한도를 넘는 본문이 조용히 잘려 임베딩되는 것을 막는다.
                            if any(len(model.tokenizer.encode(t)) > model.max_seq_length for t in texts):
                                raise ValueError('Chunk exceeds model token limit; reduce --chunk-size')
                            # 5. 본문을 숫자 벡터로 변환하고 COSINE 검색에 맞게 정규화한다.
                            vectors = model.encode(texts, batch_size=args.batch_size,
                                                   normalize_embeddings=True).tolist()
                            for row, vector in zip(batch, vectors):
                                # 새 ID면 생성 시각을 넣고 기존 ID면 갱신 시각만 바꾼다.
                                row.update(embedding=vector, updated_at=now,
                                           created_at=previous.get(row['chunk_id'], now))
                            # 6. 동일 chunk_id가 있으면 갱신하고 없으면 새로 넣는다.
                            client.upsert(collection_name=collection, data=batch)
                        # 7. 모든 새 청크 적재 성공 후, 재청킹으로 사라진 이전 ID를 삭제한다.
                        # 본문이 비었다면 새 청크가 없으므로 기존 청크 전체가 삭제된다.
                        stale = sorted(set(previous) - {row['chunk_id'] for row in chunks})
                        for offset in range(0, len(stale), args.batch_size):
                            client.delete(collection_name=collection, ids=stale[offset:offset + args.batch_size])
                    total += len(chunks)
                    LOG.info('paper_id=%s chunks=%s total=%s', paper['paper_id'], len(chunks), total)
        if client:
            # 마지막으로 서버에 flush를 요청하고 완료 로그를 남긴다.
            client.flush(collection_name=collection)
        LOG.info('Completed: %s chunks%s', total, ' (dry-run)' if args.dry_run else '')
    finally:
        # 정상 종료 또는 오류 발생 시 열린 파일과 Milvus 연결을 정리한다.
        if output:
            output.close()
        if client:
            client.close()


if __name__ == '__main__':
    # python migrate.py로 실행하면 시작한다. 다른 파일에서 import할 때는 실행하지 않는다.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    main()
