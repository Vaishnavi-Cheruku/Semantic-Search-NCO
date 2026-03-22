import os, re, json, torch, faiss, pandas as pd, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------
# PATHS — Using raw strings to avoid escape sequence issues
# -------------------------------------------------------------------
OUT_DIR = "C:/VAISHNAVI/College/SEM-7/Project/Semantic_search(Final)/outputs"
FINETUNED_SBERT = "C:/VAISHNAVI/College/SEM-7/Project/Semantic_search(Final)/outputs/sbert_nco_finetuned"
FAISS_INDEX_PATH = "C:/VAISHNAVI/College/SEM-7/Project/Semantic_search(Final)/outputs/nco_faiss.index"
INDEX_DF_PATH = "C:/VAISHNAVI/College/SEM-7/Project/Semantic_search(Final)/outputs/index_df_canonical.csv"
MLP_PATH = "C:/VAISHNAVI/College/SEM-7/Project/Semantic_search(Final)/outputs/indicbert_mlp_mapper.pt"



# -------------------------------------------------------------------
# VERIFY FILES EXIST
# -------------------------------------------------------------------
print("\n=== File Verification ===")
print(f"FAISS Index: {FAISS_INDEX_PATH} - {'✓' if os.path.exists(FAISS_INDEX_PATH) else '✗'}")
print(f"Index DF: {INDEX_DF_PATH} - {'✓' if os.path.exists(INDEX_DF_PATH) else '✗'}")
print(f"MLP Mapper: {MLP_PATH} - {'✓' if os.path.exists(MLP_PATH) else '✗'}")
print(f"Fine-tuned SBERT: {FINETUNED_SBERT} - {'✓' if os.path.exists(FINETUNED_SBERT) else '✗'}")

# -------------------------------------------------------------------
# FASTAPI APP + CORS
# -------------------------------------------------------------------
app = FastAPI(title="NCO Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# LOAD SBERT (FINE-TUNED)
# -------------------------------------------------------------------
try:
    if os.path.exists(FINETUNED_SBERT):
        print("\n✓ Loading FINE-TUNED SBERT:", FINETUNED_SBERT)
        sbert = SentenceTransformer(FINETUNED_SBERT)
    else:
        print("\n⚠ Finetuned SBERT not found! Loading base model.")
        sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    sbert.max_seq_length = 256
    print("✓ SBERT loaded successfully")
except Exception as e:
    print(f"❌ SBERT loading failed: {e}")
    raise

# -------------------------------------------------------------------
# LOAD FAISS INDEX
# -------------------------------------------------------------------
try:
    assert os.path.exists(FAISS_INDEX_PATH), f"FAISS index missing at: {FAISS_INDEX_PATH}"
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✓ FAISS loaded. Vectors: {faiss_index.ntotal}")
except Exception as e:
    print(f"❌ FAISS loading failed: {e}")
    raise

# -------------------------------------------------------------------
# LOAD INDEX DF
# -------------------------------------------------------------------
try:
    assert os.path.exists(INDEX_DF_PATH), f"index_df missing at: {INDEX_DF_PATH}"
    index_df = pd.read_csv(INDEX_DF_PATH, dtype=str).fillna("")
    print(f"✓ index_df loaded. Rows: {len(index_df)}")
except Exception as e:
    print(f"❌ Index DF loading failed: {e}")
    raise

# -------------------------------------------------------------------
# LOAD INDICBERT + MLP MAPPER
# -------------------------------------------------------------------
USE_INDIC = os.path.exists(MLP_PATH)

if USE_INDIC:
    try:
        print("\n✓ Loading IndicBERT + MLP mapper...")
        ckpt = torch.load(MLP_PATH, map_location="cpu")

        inp_dim = ckpt["inp_dim"]
        out_dim = ckpt["out_dim"]
        indic_model_name = ckpt.get("indic_model", "ai4bharat/IndicBERTv2-MLM-only")

        indic_tokenizer = AutoTokenizer.from_pretrained(indic_model_name)
        indic_model = AutoModel.from_pretrained(indic_model_name)
        indic_model.eval()

        class MapperMLP(torch.nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(inp, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, out)
                )
            def forward(self, x):
                y = self.net(x)
                return y / (torch.norm(y, dim=1, keepdim=True) + 1e-9)

        mlp = MapperMLP(inp_dim, out_dim)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        print("✓ IndicBERT + MLP loaded successfully")
    except Exception as e:
        print(f"❌ IndicBERT loading failed: {e}")
        USE_INDIC = False
        indic_tokenizer = None
        indic_model = None
        mlp = None
else:
    indic_tokenizer = None
    indic_model = None
    mlp = None
    print("⚠ No Indic MLP found → Indic pipeline disabled.")

# -------------------------------------------------------------------
# MULTI-SCRIPT DETECTION
# -------------------------------------------------------------------
indic_scripts_re = re.compile(
    r'['
    r'\u0900-\u097F'  # Devanagari
    r'\u0980-\u09FF'  # Bengali
    r'\u0A00-\u0A7F'  # Gurmukhi
    r'\u0A80-\u0AFF'  # Gujarati
    r'\u0B00-\u0B7F'  # Oriya
    r'\u0B80-\u0BFF'  # Tamil
    r'\u0C00-\u0C7F'  # Telugu
    r'\u0C80-\u0CFF'  # Kannada
    r'\u0D00-\u0D7F'  # Malayalam
    r'\u0600-\u06FF'  # Urdu
    r']'
)

# -------------------------------------------------------------------
# ENCODING FUNCTIONS
# -------------------------------------------------------------------

def encode_sbert(q):
    return sbert.encode([q], normalize_embeddings=True)[0]

def encode_indic(q):
    enc = indic_tokenizer([q], truncation=True, padding=True,
                          max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = indic_model(**enc)
        mask = enc["attention_mask"]
        hidden = out.last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        mapped = mlp(pooled).cpu().numpy()[0]
        return mapped

# -------------------------------------------------------------------
# FAISS SEARCH
# -------------------------------------------------------------------
def faiss_search(vec, k=5):
    D, I = faiss_index.search(np.array([vec]).astype("float32"), k)
    out = []
    for score, idx in zip(D[0], I[0]):
        rec = index_df.iloc[idx]
        out.append({
            "score": float(score),
            "NCO_Code": rec["NCO_Code"],
            "Title": rec["Title"],
            "Description": (rec.get("Description", "") or "")[:300]
        })
    return out

# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sbert_loaded": sbert is not None,
        "faiss_vectors": faiss_index.ntotal,
        "indic_enabled": USE_INDIC
    }

@app.get("/search")
def search(query: str, k: int = 5):
    if not query or not query.strip():
        raise HTTPException(400, "Query is required and cannot be empty")
    
    if k < 1 or k > 20:
        raise HTTPException(400, "k must be between 1 and 20")

    query = query.strip()

    try:
        vec = encode_sbert(query)

        results = faiss_search(vec, k)

        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")

print("\n=== API Ready ===")
print("Run with: uvicorn main:app --reload")
print("Access at: http://localhost:8000")
print("Docs at: http://localhost:8000/docs")