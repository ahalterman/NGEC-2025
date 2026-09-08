"""
Document features for the reference event classifiers.

Both the training script (`setup/train_classifiers/codebook_llm/`) and the
serving classifier (`plover_sklearn.py`) build features through this module, so
that what a model is trained on and what it sees during inference should match.

We represent each document as a concatenation of two feature vectors:

1. **Chunked sentence embeddings.** Sentence-transformer models truncate at a
   releatively low number of tokens and silently drop everything after the context
   window. We chunk each document, embed the chunks separately, and mean pool the embeddings.
2. **TF-IDF word features.** Many event types are easily predicted from a small number
   of keywords. To get those benefits, we also use a tf-idf rescaled bag-of-word representation
   for each doc.
"""

import numpy as np
import scipy.sparse as sp

# Word windows sized to stay under the encoder's 384 word-piece limit. News prose
# runs roughly 1.3 word pieces per word, so 220 words is about 290 pieces, which is
# comfortably inside the limit without wasting too much.
CHUNK_WORDS = 220
CHUNK_OVERLAP = 30


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a document into overlapping word windows.

    The overlap helps avoid the situation where an event mentioned in a sentence that straddles a chunk
    boundary would otherwise be split across two windows.

    Returns at least one chunk, so a short document is handled the same way as a
    long one.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def encode_documents(encoder, texts, batch_size: int = 64,
                     show_progress: bool = False,
                     chunk_words: int = CHUNK_WORDS,
                     overlap: int = CHUNK_OVERLAP) -> np.ndarray:
    """
    Encode documents as the mean of their chunk embeddings.

    All chunks from all documents are encoded in one batched call.

    Parameters
    ----------
    encoder : SentenceTransformer
    texts : list of str

    Returns
    -------
    numpy.ndarray, shape (len(texts), embedding_dim), L2-normalized.
    """
    all_chunks = []
    owners = []  # which document each chunk came from
    for i, text in enumerate(texts):
        chunks = chunk_text(text, chunk_words, overlap)
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))

    chunk_emb = encoder.encode(all_chunks, batch_size=batch_size,
                               show_progress_bar=show_progress)
    chunk_emb = np.asarray(chunk_emb, dtype=np.float32)

    dim = chunk_emb.shape[1]
    summed = np.zeros((len(texts), dim), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.float32)
    np.add.at(summed, owners, chunk_emb)
    np.add.at(counts, owners, 1.0)

    doc_emb = summed / counts[:, None]
    return _l2_normalize(doc_emb)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # A document can in principle produce a zero vector; don't divide by zero.
    norms[norms == 0] = 1.0
    return x / norms


def combine_features(doc_emb: np.ndarray, tfidf) -> sp.csr_matrix:
    """
    Join embeddings and TF-IDF word features into one sparse matrix.

    Parameters
    ----------
    doc_emb : numpy.ndarray, shape (n_docs, embedding_dim)
        Mean-pooled chunk embeddings from `encode_documents`.
    tfidf : scipy.sparse matrix, shape (n_docs, vocab_size)
        Output of a fitted TfidfVectorizer's `transform`.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_docs, embedding_dim + vocab_size).
    The dense embedding block comes first, so a model's coefficients can be split
    back into the two blocks by slicing at `embedding_dim` -- which is how the
    selected word list is recovered for inspection.

    Each block is L2-normalized separately first. Without that, whichever block
    happens to have the larger norm would dominate, and the single L1 penalty
    would fall almost entirely on the other one.
    """
    # TfidfVectorizer already L2-normalizes rows by default, but normalize again
    # rather than depend on how the vectorizer was configured.
    tfidf = sp.csr_matrix(tfidf)
    norms = sp.linalg.norm(tfidf, axis=1)
    norms[norms == 0] = 1.0
    tfidf = sp.diags(1.0 / norms) @ tfidf

    return sp.hstack([sp.csr_matrix(_l2_normalize(doc_emb)), tfidf],
                     format="csr")
