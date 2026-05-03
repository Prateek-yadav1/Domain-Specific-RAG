from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        # store original docs
        self.documents = documents
        
        # tokenize
        self.tokenized_docs = [doc.page_content.split() for doc in documents]
        
        # build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_top_k(self, query, k=3):
        tokenized_query = query.split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # get top indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [self.documents[i] for i in top_indices]