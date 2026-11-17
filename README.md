# PostgreSQL → Milvus 논문 청크 적재

`db_schema_261112_v0.3.pptx` 12·13페이지의 `paper_chunks` 스키마와
`deposition_ald_dumps/deposition_ald_dumps/deposition_ald_all.sql` 테이블 정의에 맞춘 초기 구현입니다.
SQL 파일을 실행하거나 PostgreSQL 데이터를 변경하지 않습니다. 덤프와 동일한 구조의 실행 중인 PostgreSQL에서 읽습니다.

코드는 두 파일로 구성됩니다.
- `chunking.py`: 청크 경계 계산(`spans`), 본문 분할·문자 위치·페이지·메타데이터 생성(`make_chunks`). DB나 임베딩 라이브러리 없이 사용할 수 있습니다.
- `migrate.py`: PostgreSQL 조회, 임베딩 생성, Milvus 스키마 및 적재 처리. `chunking.py`를 import해서 실행합니다.

## 준비 및 실행 (PowerShell)

Python 3.11 이상과 실행 중인 Milvus 2.6 서버가 필요합니다. Windows에서는 원격 서버 또는 Docker/WSL의 Milvus 서버를 사용하세요.

```powershell
cd C:\Users\11790\Desktop\milvus
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# .env의 POSTGRES_DSN, MILVUS_URI, MILVUS_TOKEN 등을 실제 값으로 수정
.\.venv\Scripts\python.exe -m unittest -v

# PostgreSQL에서 3개 논문 조회 및 청킹만 확인 (모델 다운로드 / Milvus 접속 없음)
.\.venv\Scripts\python.exe migrate.py --dry-run --limit 3 --output preview.jsonl

# 한 논문으로 실제 적재 확인
.\.venv\Scripts\python.exe migrate.py --paper-id 60

# 전체 적재
.\.venv\Scripts\python.exe migrate.py

# BM25도 사용하려면 처음부터 아래 옵션으로 생성 및 실행
.\.venv\Scripts\python.exe migrate.py --bm25
```

접속 문자열의 비밀번호에 `@`, `:` 등 특수문자가 있으면 URL 인코딩해야 합니다.
기본 임베딩 모델은 `BAAI/bge-m3`이며 첫 실행에 모델 다운로드가 발생합니다.
차원은 모델에서 자동 조회합니다. GPU 사용 시 `EMBEDDING_DEVICE=cuda`로 설정합니다.
재현성이 필요하면 `EMBEDDING_REVISION`에 모델 커밋을 지정하세요.

## 데이터 매핑

| Milvus 필드 | PostgreSQL / 생성 규칙 |
|---|---|
| chunk_id | `<paper_id>-<0부터 시작하는 chunk_seq:04d>` |
| paper_id, year, has_experiments | literature.papers, year NULL은 0 |
| text | paper_sections를 seq 순서로 개별 청킹, 유효 섹션이 없으면 full_text 사용 |
| char_start, char_end | full_text의 Python 문자 위치, 끝 위치 제외. 정확히 대응하지 않으면 둘 다 -1 |
| page_start, page_end | full_text의 form-feed(`\f`) 경계로 계산, 경계 정보 없으면 0(미상) |
| precursor_ids | experiments.paper_id로 연결한 precursor_id 중복 제거 |
| central_metals | experiments → precursors 연결 후 central_metal 중복 제거 |
| embedding | 청크 본문을 정규화한 밀집 벡터 |
| sparse | --bm25 사용 시 Milvus BM25 함수가 생성 |
| created_at, updated_at | Unix 밀리초, 기존 ID의 created_at 보존 |

기본 청크 크기는 1,800자, 겹침은 200자입니다. 각 섹션 안에서만 문단·문장 경계를 우선해 나누며 서로 다른 섹션 사이에는 겹침을 적용하지 않습니다. 청크 번호는 논문 전체에서 계속 증가합니다. 본문을 정규화하거나 공백을 제거하지 않습니다.
모델 토큰 한도를 초과하는 청크는 조용히 잘라 임베딩하지 않고 오류를 냅니다. 필요하면 `--chunk-size 1000 --overlap 150`처럼 줄이세요.
`--dry-run`은 임베딩을 수행하지 않으므로 토큰 한도 검사는 실제 적재 시 수행합니다.

문서의 페이지 번호는 1부터지만 덤프에는 정확한 PDF 페이지별 원문 위치가 없습니다.
`paper_sections.para_pages`만으로는 `papers.full_text`의 각 글자 페이지를 확정할 수 없어 사용하지 않습니다.
정확한 페이지 인용이 필요하면 PDF 추출 단계에서 페이지 경계를 저장해야 합니다.
full_text가 비어 있어도 유효한 섹션이 있으면 섹션을 청킹합니다. 둘 다 없으면 청크를 만들지 않습니다.
섹션 본문을 full_text에서 순서대로 정확히 찾을 수 없는 경우 본문은 보존하고 문자 위치는 -1, 페이지는 0으로 기록하며 경고를 남깁니다.
이번 섹션 청킹 방식은 설정 버전 2입니다. 기존 버전 1 컬렉션과 혼용하지 않도록 `.env`의 `MILVUS_COLLECTION=paper_chunks_v2`처럼 새 이름으로 실행하세요.
Milvus 필드는 기존 스키마를 유지하므로 섹션명 필터는 아직 추가하지 않았습니다.

## 재실행 및 운영 범위

- 같은 ID를 upsert하고 해당 논문의 모든 새 청크 적재가 성공한 뒤 남는 이전 ID를 삭제합니다. 빈 본문으로 변경된 논문의 이전 청크도 삭제됩니다.
- 논문 단위 트랜잭션은 아니므로 실패 중간에는 일부 새 청크가 보일 수 있습니다. 같은 옵션으로 재실행하면 수렴합니다.
- 같은 컬렉션에 동시 적재를 실행하지 마세요. created_at 조회와 청크 정리가 다른 실행과 경쟁할 수 있습니다.
- 기존 컬렉션의 필드·차원·설정 정보를 확인합니다. 모델/청킹/BM25 설정이 다르면 중단하므로 새 `MILVUS_COLLECTION` 이름을 사용하세요.
- PostgreSQL은 읽기 전용 반복 읽기 트랜잭션과 서버 커서로 순회합니다. 대규모 실행에서는 긴 스냅샷 유지에 유의하세요.
- `--after-paper-id 60`으로 60보다 큰 논문부터 실행할 수 있습니다. 실패 논문은 완료 로그를 기준으로 포함시켜 재시도하세요.
- PostgreSQL에서 논문 행 자체가 삭제된 경우 Milvus의 해당 논문 자동 삭제는 수행하지 않습니다. 이 코드는 전체 삭제 동기화/CDC가 아닌 초기 이관 및 재적재용입니다.
- ARRAY 16개 또는 중심 금속 UTF-8 8바이트 제한을 넘으면 오류를 냅니다. 데이터를 누락시키며 잘라 넣지 않습니다.
- 기존 논문의 ID는 유지되므로 청킹 결과가 변경되어도 created_at은 ID 기준으로 보존됩니다.
- 이 환경에는 실행 가능한 Python 및 DB 연결 설정이 없어 Python 테스트와 실제 적재는 아직 실행하지 못했습니다.

API 참고: [Milvus BM25](https://milvus.io/docs/v2.6.x/full-text-search.md),
[Milvus query](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/query.md),
[SentenceTransformer encode](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
