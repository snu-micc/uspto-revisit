"""
LLaMA-Adapter(accessory) 파인튜닝 모델로 ORD JSON 추출 배치 추론.

원본 저장소(qai222/LLM_organic_synthesis, qai222/LLaMA2-Accessory)의
workplace_ord_finetune/infer_batch.py 를 참고해 이 프로젝트의 데이터/경로에 맞게 작성.

핵심 - 기존 ai_compare.py 와 다른 점:
  1. transformers.AutoModelForCausalLM 대신 accessory.model.meta.MetaModel 사용
     (bias-tuning + prefix-adapter 를 네이티브로 지원하는 실제 아키텍처)
  2. 프롬프트가 원본 학습 포맷과 정확히 일치 (title 없음, few-shot 없음):
     "Below is a description of an organic reaction. ... ### Procedure:\n{}\n\n### ORD JSON:\n"
  3. max_seq_len=2048 (원본 학습 설정과 동일, 기존 4096 아님)
"""
import os, sys, json, time
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, "<server_home>/llama_benchmark/LLaMA2-Accessory")

import torch
import pandas as pd
from accessory.model.meta import MetaModel

# ── 설정 ──────────────────────────────────────────────────────────────────
PRETRAINED = "<server_home>/llama_benchmark/finetuned_model/accessory_pretrained"
TOKENIZER = "<server_home>/llama_benchmark/finetuned_model/tokenizer.model"
EXCEL_PATH = "Figure3_final_anaylsis_final.xlsx"
OUTPUT_JSON = "ai_results_accessory_max_seq_2048_reppen.json"
MAX_SEQ_LEN = 2048
MAX_GEN_LEN = 1024
REPETITION_PENALTY = 1.2  # 1.1 로는 일부 반복-루프 케이스가 안 풀려서 1.2 로 상향 (실측 검증됨)

# 원본 학습에 실제로 쓰인 프롬프트 템플릿 (ord/data/data_llm.py 의 DEFAULT_PROMPT_TEMPLATE)
DEFAULT_PROMPT_TEMPLATE = ("Below is a description of an organic reaction. Extract information "
                          "from it to an ORD JSON record.\n\n### Procedure:\n{}\n\n### ORD JSON:\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── 0. Input 데이터 로드 (title 제외, paragraph 만 사용) ─────────────────────
print("\n[0] Input 데이터 로드 중...")
raw_input = pd.read_excel(EXCEL_PATH, sheet_name="Table2_raw_final_v2")
Ai_input = raw_input[raw_input["Issue"].astype(str) != "['rare_temp']"][["title", "paragraph"]]
print(f"Ai_input shape: {Ai_input.shape}")

# ── 이어하기 (title 은 결과 식별용으로만 저장, 모델 입력엔 안 씀) ────────────
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8-sig") as f:
        results = json.load(f)
    print(f"기존 결과 {len(results)}개 로드 → 이어서 진행")
else:
    results = []
    print("새로 시작")

done_keys = {(r["title"], r["paragraph"]) for r in results}


def save_results(results, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── 1. 모델 로드 (accessory MetaModel, bias+prefix 네이티브 지원) ───────────
print("\n[1] MetaModel 로드 중...")
model = MetaModel.from_pretrained(
    pretrained_path=PRETRAINED,
    llama_type="llama_adapter",
    llama_config=[f"{PRETRAINED}/config.json"],
    tokenizer_path=TOKENIZER,
    with_visual=False,
    max_seq_len=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    device=device,
)
model.eval()
print("모델 준비 완료")


def generate(procedure_text: str) -> str:
    prompt = DEFAULT_PROMPT_TEMPLATE.format(procedure_text)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model.generate([prompt], images=None, max_gen_len=MAX_GEN_LEN,
                             temperature=0.0, top_p=1.0,
                             repetition_penalty=REPETITION_PENALTY)
    return out[0].strip()


# ── 2. 벤치마크 실행 ──────────────────────────────────────────────────────
total = len(Ai_input)
print(f"\n[2] 벤치마크 시작: 전체 {total}개 / 이미 완료 {len(done_keys)}개")
print("=" * 70)

for idx, row in Ai_input.iterrows():
    title = row["title"]
    paragraph = row["paragraph"]

    if (title, paragraph) in done_keys:
        continue

    print(f"\n[{len(results)+1}/{total}] {str(title)[:50]}")
    print("-" * 70)

    t0 = time.time()
    answer = generate(paragraph)
    dt = time.time() - t0

    if not answer:
        print(f"  [WARN] 빈 출력 발생 | {dt:.1f}s")
    print(f"[Accessory] ({dt:.1f}s) {answer[:200]}")

    results.append({"title": title, "paragraph": paragraph, "prediction": answer})
    save_results(results, OUTPUT_JSON)

print("\n" + "=" * 70)
print(f"벤치마크 완료! 총 {len(results)}개 저장 → {OUTPUT_JSON}")
