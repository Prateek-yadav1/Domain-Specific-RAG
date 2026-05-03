from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score


def compute_rouge(reference, prediction):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    return scores['rougeL'].fmeasure


def compute_bleu(reference, prediction):

    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()

    return sentence_bleu(reference_tokens, prediction_tokens)


def compute_bertscore(reference, prediction):

    P, R, F1 = score([prediction], [reference], lang="en", verbose=False)

    return F1.item()


def compute_accuracy(reference, prediction):

    return int(reference.strip().lower() == prediction.strip().lower())