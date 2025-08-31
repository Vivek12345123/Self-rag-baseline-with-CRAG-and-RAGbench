from vllm import LLM, SamplingParams
import json, time, os, sys, logging, re, string, itertools, random, atexit, datetime
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
import torch.distributed as dist

from datasets import load_dataset
from datasets.utils.file_utils import DownloadConfig

# ======================= Logging =======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("selfrag_eval")

# =================== Download Config ===================
# (Use only supported args for your datasets version)
DC = DownloadConfig(max_retries=5)

# =================== Retry Wrapper =====================
def load_dataset_retry(*args, retries=5, base_sleep=2.0, jitter=0.75, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            sleep = (base_sleep ** (attempt - 1)) + random.uniform(0.0, jitter)
            logger.warning(
                f"load_dataset failed (attempt {attempt}/{retries}): {e}. Retrying in {sleep:.1f}s"
            )
            time.sleep(sleep)

# =================== Optional Metrics ==================
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    print("Warning: rouge_score not available. pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    print("Warning: bert_score not available. pip install bert_score")
    BERTSCORE_AVAILABLE = False

# ====================== Model ==========================
class SelfRAGModel:
    def __init__(self, model_path: str = "selfrag/selfrag_llama2_7b",
                 download_dir: str = "/gscratch/h2lab/akari/model_cache",
                 dtype: str = "half"):
        self.model = LLM(model_path, download_dir=download_dir, dtype=dtype)
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=200, skip_special_tokens=False
        )

    def format_prompt(self, input_text, paragraph=None):
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
        if paragraph:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>\n"
        return prompt

    def extract_utility_score(self, text: str) -> int:
        for i in range(5, 0, -1):
            if f"[Utility:{i}]" in text:
                return i
        return 0

    def extract_relevance(self, text: str) -> bool:
        return "[Relevant]" in text

    def extract_support(self, text: str) -> str:
        if "[Fully supported]" in text: return "fully_supported"
        if "[Partially supported]" in text: return "partially_supported"
        if "[No support / Contradictory]" in text: return "no_support"
        return "unknown"

    def uses_retrieval(self, text: str) -> bool:
        # NOTE: we will also track retrieval via our own boolean when we *supply* context
        return "[Retrieve]" in text

# ================= Answer Extraction ==================
def extract_answer_from_response(response: str) -> str:
    """
    Extract the clean answer from a Self-RAG response by removing special tokens
    and formatting.
    """
    # Remove utility score
    response = re.sub(r'\[Utility:\d\]', '', response)
    
    # Remove relevance marker
    response = re.sub(r'\[Relevant\]', '', response)
    
    # Remove support level markers
    response = re.sub(r'\[Fully supported\]', '', response)
    response = re.sub(r'\[Partially supported\]', '', response)
    response = re.sub(r'\[No support / Contradictory\]', '', response)
    
    # Remove retrieval markers
    response = re.sub(r'\[Retrieve\]', '', response)
    response = re.sub(r'\[Retrieval\]<paragraph>.*?</paragraph>', '', response, flags=re.DOTALL)
    
    # Clean up any additional whitespace
    response = ' '.join(response.split())
    
    return response.strip()

# ==================== Evaluator ========================
class SelfRAGEvaluator:
    def __init__(self):
        self.rouge_scorer = (
            rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            if ROUGE_AVAILABLE else None
        )

    def normalize_answer(self, s: str) -> str:
        def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t, flags=re.I)
        def white_space_fix(t): return " ".join(t.split())
        def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
        return white_space_fix(remove_articles(remove_punc((s or "").lower())))

    def exact_match_score(self, pred, gt) -> float:
        return float(self.normalize_answer(pred) == self.normalize_answer(gt))

    def f1_score(self, pred, gt) -> float:
        p = self.normalize_answer(pred).split()
        g = self.normalize_answer(gt).split()
        if not p and not g: return 1.0
        if not p or not g: return 0.0
        common = Counter(p) & Counter(g)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = num_same / len(p)
        recall = num_same / len(g)
        return 2 * precision * recall / (precision + recall)

    def evaluate_multiple_answers(self, prediction, ground_truths):
        if not ground_truths: return {'em': 0.0, 'f1': 0.0}
        best_em = 0.0; best_f1 = 0.0
        
        # Extract clean answer from prediction
        clean_prediction = extract_answer_from_response(prediction)
        
        # For debugging: if scores are still zero, log the comparison
        if os.environ.get("SR_DEBUG_MATCHING", "0") == "1":
            logger.debug(f"Comparing cleaned answer: '{clean_prediction}' with ground truths: {ground_truths}")
        
        for gt in ground_truths:
            if not (gt and str(gt).strip()): continue
            em_score = self.exact_match_score(clean_prediction, gt)
            f1_score = self.f1_score(clean_prediction, gt)
            best_em = max(best_em, em_score)
            best_f1 = max(best_f1, f1_score)
            
            # Extra debug logging if enabled
            if os.environ.get("SR_DEBUG_MATCHING", "0") == "1" and (em_score > 0 or f1_score > 0.5):
                logger.debug(f"Match found - EM: {em_score}, F1: {f1_score}")
                logger.debug(f"Normalized pred: '{self.normalize_answer(clean_prediction)}'")
                logger.debug(f"Normalized GT: '{self.normalize_answer(gt)}'")
                
        return {'em': best_em, 'f1': best_f1}

evaluator = SelfRAGEvaluator()

# ================== Safe Generation ====================
def safe_generate(model: SelfRAGModel, prompt: str):
    out = model.model.generate([prompt], model.sampling_params)[0]
    if not getattr(out, "outputs", None):
        return "", 0
    first = out.outputs[0]
    return (first.text or ""), len(first.token_ids or [])

# ============== Tiny TF-IDF Retriever ==================
class MiniTfidfRetriever:
    """
    Minimal TF-IDF + cosine retriever (no external deps).
    Build on a list of strings. Tokenizer = lowercase, \w+.
    """
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.N = len(docs)
        self.token_re = re.compile(r"\w+")
        # build vocabulary and idf
        df = Counter()
        self.doc_tokens = []
        for text in docs:
            toks = self._tokenize(text)
            self.doc_tokens.append(toks)
            df.update(set(toks))
        self.idf = {t: (np.log((1 + self.N) / (1 + c)) + 1.0) for t, c in df.items()}
        # doc vectors
        self.doc_vecs = []
        self.doc_norms = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            vec = {t: (tf[t] * self.idf.get(t, 0.0)) for t in tf}
            norm = np.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.doc_vecs.append(vec)
            self.doc_norms.append(norm)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [m.group(0).lower() for m in self.token_re.finditer(text)]

    def _vec(self, text: str):
        toks = self._tokenize(text)
        tf = Counter(toks)
        vec = {t: (tf[t] * self.idf.get(t, 0.0)) for t in tf}
        norm = np.sqrt(sum(v*v for v in vec.values())) or 1.0
        return vec, norm

    def search(self, query: str, k: int = 3) -> List[tuple[int, float]]:
        qv, qn = self._vec(query or "")
        if not qv:
            return []
        scores = []
        for i, dv in enumerate(self.doc_vecs):
            # dot product over smaller dict
            if len(qv) < len(dv):
                dot = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            else:
                dot = sum(w * qv.get(t, 0.0) for t, w in dv.items())
            sim = dot / (self.doc_norms[i] * qn)
            scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# ================= Helper utils ========================
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    if not text: return []
    parts = _SENT_SPLIT.split(text.strip())
    # keep non-empty short-ish sentences
    return [p.strip() for p in parts if p and len(p.strip()) > 0]

def cap_list(xs: List[str], cap: int) -> List[str]:
    if cap and len(xs) > cap:
        return xs[:cap]
    return xs

def join_snippets(snips: List[str], max_chars: int = 1500) -> str:
    out, total = [], 0
    for s in snips:
        s2 = s.strip()
        if not s2: continue
        if total + len(s2) + 1 > max_chars: break
        out.append(s2); total += len(s2) + 1
    return "\n".join(out)

# ===================== Benchmarks ======================

def run_nq_rag_benchmark(model, sample_size: int = 200, streaming=False):
    """
    Natural Questions (google-research-datasets/natural_questions default).
    Build a global corpus from any available 'document' tokens when present,
    otherwise fall back to no retrieval for those items.
    """
    logger.info(f"Running NQ (RAG) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("google-research-datasets/natural_questions", "default",
                                         split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("google-research-datasets/natural_questions", "default",
                                    split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Error loading NQ: {e}", exc_info=True)
        return []

    # Build corpus from document tokens where available
    corpus = []
    for item in ds:
        document = item.get("document") or {}
        tokens = document.get("tokens") or []
        if tokens and isinstance(tokens, list):
            text = " ".join([t.get("token","") if isinstance(t, dict) else str(t) for t in tokens[:800]])
            if text.strip():
                corpus.append(text.strip())
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    corpus = cap_list(list(dict.fromkeys(corpus)), corpus_cap)

    retriever = MiniTfidfRetriever(corpus) if corpus else None
    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))

    results = []
    for i, item in enumerate(ds):
        try:
            q = item.get("question", "")
            if isinstance(q, dict):
                question = q.get("text", "") or ""
            else:
                question = q or ""

            # ground truths (if present in this config)
            gts = item.get("answers") or []
            gts = [a for a in gts if a]

            paragraph = None
            if retriever and question:
                hits = retriever.search(question, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            prompt = model.format_prompt(question, paragraph)
            t0 = time.time()
            resp, tok = safe_generate(model, prompt)
            dt = time.time() - t0

            scores = evaluator.evaluate_multiple_answers(resp, gts) if gts else {'em':0.0,'f1':0.0}

            results.append({
                'dataset':'nq','question':question,'response':resp,
                'ground_truth_answers':gts,'exact_match':scores['em'],'f1_score':scores['f1'],
                'inference_time':dt,'tokens_generated':tok,
                'utility_score':model.extract_utility_score(resp),
                'is_relevant':model.extract_relevance(resp),
                'support_level':model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })
            if (i+1)%10==0: logger.info(f"NQ (RAG) processed {i+1}/{len(ds)}")
        except Exception as e:
            logger.error(f"NQ item {i} error: {e}", exc_info=True)

    logger.info(f"NQ (RAG) completed with {len(results)} samples")
    return results


def run_trivia_qa_rag_benchmark(model, sample_size: int = 200, streaming=False):
    """
    TriviaQA (rc): build a global corpus from 'context' strings and retrieve for each question.
    """
    logger.info(f"Running TriviaQA(rc) (RAG) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("mandarjoshi/trivia_qa", "rc", split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Error loading TriviaQA: {e}", exc_info=True)
        return []

    # Build global corpus
    contexts = [ (item.get("context") or "").strip() for item in ds ]
    contexts = [c for c in contexts if c]
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    corpus = cap_list(list(dict.fromkeys(contexts)), corpus_cap)
    retriever = MiniTfidfRetriever(corpus) if corpus else None
    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))

    results = []
    for i, item in enumerate(ds):
        try:
            question = (item.get("question") or "").strip()
            ans = item.get("answer", {}) or {}
            gts = []
            if ans.get("value"): gts.append(ans["value"])
            gts += [a for a in (ans.get("aliases") or []) if a]

            paragraph = None
            if retriever and question:
                hits = retriever.search(question, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            prompt = model.format_prompt(question, paragraph)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time()-t0
            scores = evaluator.evaluate_multiple_answers(resp, gts) if gts else {'em':0.0,'f1':0.0}

            results.append({
                'dataset':'trivia_qa','question':question,'response':resp,
                'ground_truth_answers':gts,'exact_match':scores['em'],'f1_score':scores['f1'],
                'inference_time':dt,'tokens_generated':tok,
                'utility_score':model.extract_utility_score(resp),
                'is_relevant':model.extract_relevance(resp),
                'support_level':model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })
            if (i+1)%10==0: logger.info(f"TriviaQA (RAG) processed {i+1}/{len(ds)}")
        except Exception as e:
            logger.error(f"TriviaQA item {i} error: {e}", exc_info=True)

    logger.info(f"TriviaQA (RAG) completed with {len(results)} samples")
    return results


def run_hotpot_qa_rag_benchmark(model, sample_size: int = 200, streaming=False):
    """
    HotpotQA distractor: build a global corpus of paragraphs (title: sentences...) and retrieve.
    """
    logger.info(f"Running HotpotQA(distractor) (RAG) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("hotpotqa/hotpot_qa", "distractor", split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Error loading HotpotQA: {e}", exc_info=True)
        return []

    # Build corpus of paragraphs (title + sentences)
    paragraphs = []
    for item in ds:
        for pair in item.get("context", []):
            if isinstance(pair,(list,tuple)) and len(pair)==2:
                title, sentences = pair
                if sentences:
                    paragraphs.append(f"{title}: {' '.join(sentences)}")
    paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    corpus = cap_list(list(dict.fromkeys(paragraphs)), corpus_cap)
    retriever = MiniTfidfRetriever(corpus) if corpus else None
    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))

    results=[]
    for i, item in enumerate(ds):
        try:
            question = (item.get("question") or "").strip()
            gold = (item.get("answer") or "").strip()

            paragraph = None
            if retriever and question:
                hits = retriever.search(question, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            prompt = model.format_prompt(question, paragraph)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time()-t0
            scores = evaluator.evaluate_multiple_answers(resp, [gold]) if gold else {'em':0.0,'f1':0.0}

            results.append({
                'dataset':'hotpot_qa','question':question,'response':resp,'ground_truth_answer':gold,
                'exact_match':scores['em'],'f1_score':scores['f1'],
                'inference_time':dt,'tokens_generated':tok,
                'utility_score':model.extract_utility_score(resp),
                'is_relevant':model.extract_relevance(resp),
                'support_level':model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })
            if (i+1)%10==0: logger.info(f"HotpotQA (RAG) processed {i+1}/{len(ds)}")
        except Exception as e:
            logger.error(f"Hotpot item {i} error: {e}", exc_info=True)

    logger.info(f"HotpotQA (RAG) completed with {len(results)} samples")
    return results


def run_squad_v2_rag_benchmark(model, sample_size: int = 200, streaming=False):
    """
    SQuAD v2: per-item retrieval over that item's context sentences (selective grounding).
    """
    logger.info(f"Running SQuAD v2 (RAG) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("rajpurkar/squad_v2", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("rajpurkar/squad_v2", split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Error loading SQuAD v2: {e}", exc_info=True)
        return []

    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))

    results=[]
    for i, item in enumerate(ds):
        try:
            question = (item.get("question") or "").strip()
            context = (item.get("context") or "").strip()
            answers = item.get("answers", {}) or {}
            gts = [a for a in (answers.get("text") or []) if a]
            is_impossible = (len(gts) == 0)

            paragraph = None
            if context and question:
                # per-item retriever over sentences of this context
                sents = split_sentences(context)
                if sents:
                    retr = MiniTfidfRetriever(sents)
                    hits = retr.search(question, k=top_k)
                    if hits:
                        snips = [sents[idx] for idx, _ in hits]
                        paragraph = join_snippets(snips)

            prompt = model.format_prompt(question, paragraph if paragraph else context)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time()-t0

            if not is_impossible and gts:
                scores = evaluator.evaluate_multiple_answers(resp, gts)
            else:
                # simple impossible detection
                no_ans = ["no answer","cannot answer","not provided","unknown","unanswerable"]
                detected = any(ind in (resp.lower() if resp else "") for ind in no_ans)
                scores = {'em': 1.0 if detected else 0.0, 'f1': 1.0 if detected else 0.0}

            results.append({
                'dataset':'squad_v2','question':question,'response':resp,
                'ground_truth_answers':gts,'is_impossible':is_impossible,
                'exact_match':scores['em'],'f1_score':scores['f1'],
                'inference_time':dt,'tokens_generated':tok,
                'utility_score':model.extract_utility_score(resp),
                'is_relevant':model.extract_relevance(resp),
                'support_level':model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })
            if (i+1)%10==0: logger.info(f"SQuAD v2 (RAG) processed {i+1}/{len(ds)}")
        except Exception as e:
            logger.error(f"SQuAD item {i} error: {e}", exc_info=True)

    logger.info(f"SQuAD v2 (RAG) completed with {len(results)} samples")
    return results


def run_fever_rag_benchmark(model, sample_size: int = 200, streaming: bool = False):
    """
    FEVER with actual retrieval: build a global corpus from available evidence texts.
    NOTE: using a HF mirror that *includes* textual evidence. Adjust if your mirror changes.
    """
    logger.info(f"Running FEVER (RAG) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("mwong/fever-evidence-related", split="valid", streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("mwong/fever-evidence-related", split="valid", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Failed to load FEVER: {e}", exc_info=True)
        return []

    def collect_texts(evidence):
        texts = []
        def collect(x):
            if isinstance(x, str) and x.strip():
                texts.append(x.strip())
            elif isinstance(x, dict):
                t = (x.get("text") or x.get("evidence_text") or "").strip()
                if t: texts.append(t)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    collect(y)
        collect(evidence)
        return texts

    corpus = []
    for item in ds:
        corpus.extend(collect_texts(item.get("evidence")))
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    corpus = cap_list(list(dict.fromkeys([c for c in corpus if c])), corpus_cap)
    retriever = MiniTfidfRetriever(corpus) if corpus else None
    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))

    results = []
    for i, item in enumerate(ds):
        try:
            claim = (item.get("claim") or "").strip()
            label = (item.get("label") or "").strip()

            paragraph = None
            if retriever and claim:
                hits = retriever.search(claim, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            prompt = model.format_prompt(claim, paragraph)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time()-t0

            scores = evaluator.evaluate_multiple_answers(resp, [label]) if label else {'em':0.0,'f1':0.0}

            results.append({
                'dataset': 'fever','claim': claim,'response': resp,'label': label,
                'exact_match': scores['em'],'f1_score': scores['f1'],
                'inference_time': dt,'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })
            if (i+1)%10==0: logger.info(f"FEVER (RAG) processed {i+1}/{len(ds)}")
        except Exception as e:
            logger.error(f"FEVER item {i} error: {e}", exc_info=True)

    logger.info(f"FEVER (RAG) completed with {len(results)} samples")
    return results

def run_ragtruth_rag_benchmark(model, sample_size: int = 200, streaming: bool = False, split: Optional[str] = None):
    """
    RAGTruth (wandb/RAGTruth-processed) with actual retrieval.
    - Builds a global TF-IDF index over evidence/contexts found in the dataset.
    - Retrieves top-k snippets per query and injects them into the [Retrieval] paragraph.
    - If a gold answer/reference string exists, evaluates EM/F1; otherwise EM/F1=0 by design.
    """
    ds_id = "wandb/RAGTruth-processed"
    # Try sensible split fallbacks if the caller didn't specify one
    split_candidates = [split] if split else ["validation", "val", "dev", "test"]
    ds = None
    err_last = None

    logger.info(f"Running RAGTruth (RAG) sample_size={sample_size} (streaming={streaming})")

    # -------- dataset load with fallback over splits --------
    for sp in split_candidates:
        if sp is None:
            continue
        try:
            if streaming:
                ds_iter = load_dataset_retry(ds_id, split=sp, streaming=True, download_config=DC)
                ds = list(itertools.islice(ds_iter, sample_size))
            else:
                ds = load_dataset_retry(ds_id, split=sp, download_config=DC)
                if sample_size < len(ds):
                    ds = ds.select(range(sample_size))
            logger.info(f"Loaded {ds_id} split='{sp}' with {len(ds)} rows")
            break
        except Exception as e:
            err_last = e
            logger.warning(f"Load failed for split '{sp}': {e}")

    if ds is None:
        logger.error(f"Failed to load {ds_id} on splits {split_candidates}: {err_last}", exc_info=True)
        return []

    # -------- helpers to be robust to schema variations --------
    TEXT_KEYS_QUERY = ["claim", "question", "query", "prompt", "instruction", "input", "task"]
    TEXT_KEYS_GOLD  = ["answer", "answers", "reference", "reference_answer",
                       "ground_truth", "ground_truth_answer", "target", "output", "label"]
    # Evidence/context style fields we can index
    EVIDENCE_KEYS   = ["evidence", "evidences", "evidence_text", "evidence_texts",
                       "contexts", "context", "passages", "documents", "docs",
                       "supporting_facts", "references"]

    def first_nonempty(item, keys):
        for k in keys:
            if k in item and item[k] is not None:
                v = item[k]
                if isinstance(v, str) and v.strip():
                    return v.strip()
                # prefer single string if list
                if isinstance(v, list):
                    for vv in v:
                        if isinstance(vv, str) and vv.strip():
                            return vv.strip()
                # sometimes nested dict has 'text'
                if isinstance(v, dict):
                    t = v.get("text") or v.get("value")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
        return ""

    def collect_texts(obj):
        """Collect textual snippets from strings / dicts / lists recursively."""
        acc = []
        def rec(x):
            if x is None: return
            if isinstance(x, str):
                s = x.strip()
                if s: acc.append(s)
            elif isinstance(x, dict):
                # common text-ish keys
                for key in ("text", "snippet", "passage", "content", "sentence", "evidence_text", "value"):
                    val = x.get(key)
                    if isinstance(val, str) and val.strip():
                        acc.append(val.strip())
                # also crawl nested
                for vv in x.values():
                    rec(vv)
            elif isinstance(x, (list, tuple)):
                for y in x: rec(y)
        rec(obj)
        return acc

    # -------- build a global retrieval corpus --------
    corpus = []
    for item in ds:
        for key in EVIDENCE_KEYS:
            if key in item and item[key] is not None:
                corpus.extend(collect_texts(item[key]))

    # fallback: if we found nothing, also pull any long "context" from the record
    if not corpus:
        for item in ds:
            corpus.extend(collect_texts(item))

    # dedupe + cap size for memory
    corpus = [c for c in corpus if isinstance(c, str) and c.strip()]
    corpus = list(dict.fromkeys(corpus))
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    if len(corpus) > corpus_cap:
        corpus = corpus[:corpus_cap]

    if not corpus:
        logger.warning("RAGTruth: No textual evidence/contexts discovered; retrieval will be a no-op.")
        retriever = None
    else:
        logger.info(f"RAGTruth: Building TF-IDF index on {len(corpus)} snippets...")
        retriever = MiniTfidfRetriever(corpus)

    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))
    results = []

    # -------- evaluation loop --------
    for i, item in enumerate(ds):
        try:
            query = first_nonempty(item, TEXT_KEYS_QUERY)
            # gold answer/reference if any (some RAGTruth variants are hallucination labels only)
            golds = []
            gold_candidate = first_nonempty(item, TEXT_KEYS_GOLD)
            if gold_candidate:
                # if the field is a list of golds:
                if isinstance(item.get("answers"), list):
                    golds = [g for g in item["answers"] if isinstance(g, str) and g.strip()]
                else:
                    golds = [gold_candidate]

            # retrieve
            paragraph = None
            if retriever and query:
                hits = retriever.search(query, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            # prompt + generate
            prompt = model.format_prompt(query or "", paragraph)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time() - t0

            # score (only if we have textual golds)
            scores = evaluator.evaluate_multiple_answers(resp, golds) if golds else {'em': 0.0, 'f1': 0.0}

            results.append({
                'dataset': 'ragtruth',
                'query': query,
                'response': resp,
                'ground_truth_answers': golds,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })

            if (i + 1) % 10 == 0:
                logger.info(f"RAGTruth (RAG) processed {i+1}/{len(ds) if not streaming else sample_size}")

        except Exception as e:
            logger.error(f"RAGTruth item {i} error: {e}", exc_info=True)

    logger.info(f"RAGTruth (RAG) completed with {len(results)} samples")
    return results

def run_msmarco_rag_benchmark(
    model,
    sample_size: int = 200,
    streaming: bool = False,
    config: str = "v2.1",
    split: str = "validation"
):
    """
    MS MARCO (RAG):
    - Build a global TF-IDF corpus from dataset passages.
    - Retrieve top-k snippets per query and supply as [Retrieval] paragraph.
    - Score EM/F1 vs answers + wellFormedAnswers (when provided).
    """
    ds_id = "microsoft/ms_marco"
    logger.info(f"Running MS MARCO (RAG) {ds_id}:{config} split={split} sample_size={sample_size} (streaming={streaming})")

    # -------- Load dataset --------
    try:
        if streaming:
            ds_iter = load_dataset_retry(ds_id, config, split=split, streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry(ds_id, config, split=split, download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Error loading {ds_id}:{config} {split}: {e}", exc_info=True)
        return []

    # -------- Helpers to extract passages robustly --------
    def collect_passage_texts(passages_field):
        """
        Normalize various MS MARCO passage representations to a flat list of strings.
        Common forms:
          - dict with key 'passage_text': list[str]
          - list[dict] with keys like 'passage_text' / 'text' / 'passage'
          - list[str]
        """
        out = []
        if passages_field is None:
            return out

        if isinstance(passages_field, dict):
            # typical: {'passage_text': [...], ...}
            for key in ("passage_text", "text", "passage", "snippet"):
                vals = passages_field.get(key)
                if isinstance(vals, list):
                    out.extend([v.strip() for v in vals if isinstance(v, str) and v.strip()])
                elif isinstance(vals, str) and vals.strip():
                    out.append(vals.strip())

        elif isinstance(passages_field, list):
            for elt in passages_field:
                if isinstance(elt, str) and elt.strip():
                    out.append(elt.strip())
                elif isinstance(elt, dict):
                    for key in ("passage_text", "text", "passage", "snippet", "content"):
                        v = elt.get(key)
                        if isinstance(v, str) and v.strip():
                            out.append(v.strip())
        else:
            # single string?
            if isinstance(passages_field, str) and passages_field.strip():
                out.append(passages_field.strip())

        return out

    # -------- Build global corpus --------
    corpus = []
    for item in ds:
        p = item.get("passages")
        corpus.extend(collect_passage_texts(p))

        # Occasionally ms_marco variants expose 'contexts' or 'documents'
        if not p:
            for alt_key in ("contexts", "documents", "context"):
                if alt_key in item and item[alt_key] is not None:
                    corpus.extend(collect_passage_texts(item[alt_key]))

    # Deduplicate + cap
    corpus = [c for c in corpus if isinstance(c, str) and c.strip()]
    corpus = list(dict.fromkeys(corpus))
    corpus_cap = int(os.environ.get("SR_CORPUS_CAP", "50000"))
    if len(corpus) > corpus_cap:
        corpus = corpus[:corpus_cap]

    if not corpus:
        logger.warning("MS MARCO: No passages found to build corpus; retrieval will be a no-op.")
        retriever = None
    else:
        logger.info(f"MS MARCO: Building TF-IDF index on {len(corpus)} snippets...")
        retriever = MiniTfidfRetriever(corpus)

    top_k = int(os.environ.get("SR_RETRIEVE_K", "3"))
    results = []

    # -------- Evaluation loop --------
    for i, item in enumerate(ds):
        try:
            query = (item.get("query") or "").strip()

            # Gold answers
            answers = item.get("answers") or []
            wf = item.get("wellFormedAnswers") or []
            gts = []
            if isinstance(answers, list):
                gts.extend([a for a in answers if isinstance(a, str) and a.strip()])
            elif isinstance(answers, str) and answers.strip():
                gts.append(answers.strip())
            if isinstance(wf, list):
                gts.extend([a for a in wf if isinstance(a, str) and a.strip()])
            elif isinstance(wf, str) and wf.strip():
                gts.append(wf.strip())

            # Retrieve
            paragraph = None
            if retriever and query:
                hits = retriever.search(query, k=top_k)
                if hits:
                    snips = [corpus[idx] for idx, _ in hits]
                    paragraph = join_snippets(snips)

            # Prompt + generate
            prompt = model.format_prompt(query, paragraph)
            t0 = time.time(); resp, tok = safe_generate(model, prompt); dt = time.time() - t0

            # Score
            scores = evaluator.evaluate_multiple_answers(resp, gts) if gts else {'em': 0.0, 'f1': 0.0}

            results.append({
                'dataset': 'msmarco',
                'query': query,
                'response': resp,
                'ground_truth_answers': gts,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': bool(paragraph),
                'retrieved_k': top_k if paragraph else 0
            })

            if (i + 1) % 10 == 0:
                logger.info(f"MS MARCO (RAG) processed {i+1}/{len(ds) if not streaming else sample_size}")

        except Exception as e:
            logger.error(f"MS MARCO item {i} error: {e}", exc_info=True)

    logger.info(f"MS MARCO (RAG) completed with {len(results)} samples")
    return results



# ============== Aggregation & I/O ======================
def compute_aggregate_metrics(results):
    if not results: return {}
    metrics = ['exact_match','f1_score','utility_score']
    aggregated={}
    for m in metrics:
        vals=[r.get(m,0.0) for r in results if m in r]
        if vals:
            aggregated[m] = {
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'count': len(vals), 'min': float(np.min(vals)), 'max': float(np.max(vals))
            }
    for b in ['is_relevant','uses_retrieval']:
        vals=[float(r.get(b,False)) for r in results if b in r]
        if vals:
            aggregated[b] = {'mean': float(np.mean(vals)), 'count': len(vals)}
    support = Counter([r.get('support_level','unknown') for r in results])
    aggregated['support_distribution'] = dict(support)
    return aggregated

def save_results_to_json(results, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}", exc_info=True)

# ========================= Main ========================
def main():
    print("="*70)
    print("SELF-RAG EVALUATION (RAG-enabled TF-IDF retrieval)")
    print("="*70)

    logger.info("Initializing Self-RAG model...")
    try:
        model = SelfRAGModel(
            model_path="selfrag/selfrag_llama2_7b",
            download_dir="/gscratch/h2lab/akari/model_cache",
            dtype="half"
        )
        logger.info("✅ Model init OK")
    except Exception as e:
        logger.error(f"❌ Model init failed: {e}", exc_info=True)
        return

    sample_size = int(os.environ.get("SR_SAMPLE_SIZE", "200"))
    streaming = os.environ.get("SR_STREAMING", "0") == "1"

    results={}
    benchmarks = [
        ("Natural Questions (RAG)", run_nq_rag_benchmark),
        ("TriviaQA (RAG)",          run_trivia_qa_rag_benchmark),
        ("HotpotQA (RAG)",          run_hotpot_qa_rag_benchmark),
        ("SQuAD v2 (RAG)",          run_squad_v2_rag_benchmark),
        ("FEVER (RAG)",             run_fever_rag_benchmark),
        ("RAGTruth (RAG)",          run_ragtruth_rag_benchmark),
        ("MS MARCO (RAG)",          run_msmarco_rag_benchmark),
    ]

    logger.info(f"Running {len(benchmarks)} benchmarks; sample_size={sample_size}; streaming={streaming}")
    for name, func in benchmarks:
        print(f"\n{'='*60}\n🚀 RUNNING: {name}\n{'='*60}")
        try:
            t0=time.time()
            bench_results = func(model, sample_size=sample_size, streaming=streaming)
            t1=time.time()
            key = name.lower().replace(" ","_").replace("(","").replace(")","")
            if bench_results:
                aggregated = compute_aggregate_metrics(bench_results)
                results[key] = {
                    'individual_results': bench_results,
                    'aggregated_metrics': aggregated,
                    'total_samples': len(bench_results),
                    'execution_time': t1 - t0
                }
                logger.info(f"✅ {name}: {len(bench_results)} samples in {t1-t0:.2f}s")
            else:
                results[key] = {
                    'individual_results': [],
                    'aggregated_metrics': {},
                    'total_samples': 0,
                    'execution_time': t1 - t0,
                    'status': 'failed'
                }
                logger.warning(f"⚠️ {name} produced no results")
            save_results_to_json(results, f"selfrag_results_partial_{int(time.time())}.json")
        except Exception as e:
            logger.error(f"❌ Error running {name}: {e}", exc_info=True)
            key = name.lower().replace(" ","_").replace("(","").replace(")","")
            results[key] = {
                'individual_results': [],
                'aggregated_metrics': {},
                'total_samples': 0,
                'execution_time': 0,
                'status': 'error',
                'error_message': str(e)
            }

    final = f"selfrag_evaluation_final_{int(time.time())}.json"
    save_results_to_json(results, final)

    # Console summary
    print("\n" + "="*80)
    print("🏆 SELF-RAG EVALUATION COMPLETE - FINAL SUMMARY")
    print("="*80)
    succ = sum(1 for v in results.values() if v.get('total_samples',0)>0)
    total = sum(v.get('total_samples',0) for v in results.values())
    for k,v in results.items():
        name = k.upper().replace("_"," ")
        if v.get('total_samples',0)>0:
            ag=v['aggregated_metrics']
            print(f"\n📈 {name}: n={v['total_samples']}  time={v.get('execution_time',0):.2f}s")
            if 'exact_match' in ag:
                em=ag['exact_match']; print(f"   EM: {em['mean']:.3f} ± {em['std']:.3f} (n={em['count']})")
            if 'f1_score' in ag:
                f1=ag['f1_score']; print(f"   F1: {f1['mean']:.3f} ± {f1['std']:.3f} (n={f1['count']})")
            if 'utility_score' in ag:
                u=ag['utility_score']; print(f"   Utility: {u['mean']:.3f} ± {u['std']:.3f}")
        else:
            print(f"\n❌ {name}: {v.get('status','no-data')}")

    print("\n" + "="*80)
    print(f"📊 OVERALL: {succ}/{len(benchmarks)} benchmarks produced results; total samples: {total}")
    print(f"🗂  Results saved to: {final}")
    print("="*80)
    return results

# ============== Distributed Cleanup ====================
def _dist_cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier(timeout=datetime.timedelta(seconds=5))
            except Exception:
                pass
            dist.destroy_process_group()
    except Exception:
        pass

atexit.register(_dist_cleanup)

def run_all():
    return main()

# ====================== Entrypoint ======================
if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")

    print("🔥 SELF-RAG EVALUATION SYSTEM (RAG-enabled)")
    print("="*70)
    print("🔍 Pre-flight checks...")

    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"✅ GPU: {name} ({mem:.1f} GB), {n} visible")
        else:
            print("⚠️ No GPU detected")
    except Exception:
        print("⚠️ PyTorch not available for GPU check")

    for pkg in ['vllm','datasets','transformers','torch']:
        try:
            __import__(pkg); print(f"✅ {pkg} available")
        except Exception:
            print(f"❌ {pkg} missing")

    print("\n🚀 Starting evaluation...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal: {e}", exc_info=True)
        print("❌ Evaluation failed. See logs.")

    try:
        results = run_all()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    finally:
        _dist_cleanup()
