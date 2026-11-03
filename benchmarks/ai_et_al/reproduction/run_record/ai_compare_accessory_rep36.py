"""
ai_compare_accessory.py(ai_results_accessory_max_seq_2048_reppen.json 생성 스크립트)를
교체 문단 36건 + 재현 확인 문단 6건(llama_input.json)에 그대로 적용.

모델/프롬프트/디코딩 설정은 ai_compare_accessory.py 와 동일:
  - MetaModel(llama_adapter), bf16, max_seq_len=2048
  - DEFAULT_PROMPT_TEMPLATE, paragraph 만 입력 (title 미사용)
  - greedy (temperature=0.0, top_p=1.0), max_gen_len=1024, batch size 1

기존 reppen 결과 400건은 두 단계로 만들어졌다:
  1) repetition_penalty 기능 추가 전 실행(job 11001, 2026-07-29) = rp 1.0
  2) 1) 에서 JSON 파싱이 안 된 레코드만 rp 1.2 로 재생성(job 11198, 2026-07-30)
그래서 rp 1.0 / 1.2 두 패스를 모두 돌리고, 같은 규칙으로 합친 파일도 만든다.
출력은 고치지 않고 원문 그대로 저장한다.
"""
import os, sys, json, time, traceback
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, "<server_home>/llama_benchmark/LLaMA2-Accessory")

import torch
from accessory.model.meta import MetaModel

# ── 설정 (ai_compare_accessory.py 와 동일) ────────────────────────────────
PRETRAINED = "<server_home>/llama_benchmark/finetuned_model/accessory_pretrained"
TOKENIZER = "<server_home>/llama_benchmark/finetuned_model/tokenizer.model"
MAX_SEQ_LEN = 2048
MAX_GEN_LEN = 1024
DEFAULT_PROMPT_TEMPLATE = ("Below is a description of an organic reaction. Extract information "
                          "from it to an ORD JSON record.\n\n### Procedure:\n{}\n\n### ORD JSON:\n")

RUN_DIR = "<server_home>/llama_benchmark/rep36"
INPUT_JSON = f"{RUN_DIR}/input/llama_input.json"
OUT_DIR = f"{RUN_DIR}/output"
PASSES = [1.0, 1.2]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
os.makedirs(OUT_DIR, exist_ok=True)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)
print(f"입력 {len(records)}건")


def save_results(results, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


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


def generate(procedure_text: str, repetition_penalty: float) -> str:
    prompt = DEFAULT_PROMPT_TEMPLATE.format(procedure_text)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model.generate([prompt], images=None, max_gen_len=MAX_GEN_LEN,
                             temperature=0.0, top_p=1.0,
                             repetition_penalty=repetition_penalty)
    return out[0].strip()


pass_results = {}
for rp in PASSES:
    out_path = f"{OUT_DIR}/llama_output_rp{rp}.json"
    results = []
    print(f"\n[2] repetition_penalty={rp} 시작 ({len(records)}건)")
    print("=" * 70)
    for i, rec in enumerate(records, start=1):
        print(f"\n[rp={rp} {i}/{len(records)}] {rec['idx']} {str(rec['title'])[:50]}")
        t0 = time.time()
        error = None
        try:
            answer = generate(rec["paragraph"], rp)
        except Exception:
            answer = ""
            error = traceback.format_exc()
            torch.cuda.empty_cache()
        dt = time.time() - t0
        if error:
            print(f"  [ERROR] {dt:.1f}s\n{error}")
        elif not answer:
            print(f"  [WARN] 빈 출력 발생 | {dt:.1f}s")
        print(f"[Accessory] ({dt:.1f}s) valid_json={is_valid_json(answer)} {answer[:200]}")
        results.append({"idx": rec["idx"], "title": rec["title"], "paragraph": rec["paragraph"],
                        "prediction": answer, "error": error})
        save_results(results, out_path)
    pass_results[rp] = results
    print(f"\nrp={rp} 완료 → {out_path}")

# ── 기존 reppen 파일과 같은 규칙으로 합치기: rp1.0 결과, JSON 무효면 rp1.2 결과 ──
combined = []
for r10, r12 in zip(pass_results[1.0], pass_results[1.2]):
    use = r10 if is_valid_json(r10["prediction"]) else r12
    combined.append(dict(use))
save_results(combined, f"{OUT_DIR}/llama_output.json")
with open(f"{OUT_DIR}/llama_output_source.csv", "w", encoding="utf-8-sig") as f:
    f.write("idx,rp1.0_valid_json,rp1.2_valid_json,used_repetition_penalty\n")
    for r10, r12 in zip(pass_results[1.0], pass_results[1.2]):
        v10, v12 = is_valid_json(r10["prediction"]), is_valid_json(r12["prediction"])
        f.write(f"{r10['idx']},{v10},{v12},{1.0 if v10 else 1.2}\n")

print("\n" + "=" * 70)
print(f"완료! {OUT_DIR}/llama_output.json ({len(combined)}건)")
