import utils.utils as utils
from collections import Counter
import itertools
import json


def create_vocab(iterable, top_k=None):
    all_tokens = itertools.chain.from_iterable(iterable)
    print(all_tokens)
    counter = Counter(all_tokens)
    if top_k is not None:
        words = [word for word, count in counter.most_common(top_k)]
    else:
        words = counter.keys()
    tokens = sorted(words, key=lambda x: (counter[x], x), reverse=True)
    vocab = {word: index for index, word in enumerate(tokens)}
    return vocab


if __name__ == "__main__":

    with open("config.json", "r") as conf:
        config = json.loads(conf.read())
    with open(config.get("question_train"), "r") as f:
        questions = json.load(f)
    with open(config.get("answer_train"), "r") as f:
        answers = json.load(f)
    questions = utils.question_loading(questions)
    answers = utils.answer_loading(answers)
    question_vocab = create_vocab(questions)
    answer_vocab = create_vocab(answers, top_k=config.get("top_k_answers"))
    vocab = {"question_vocab": question_vocab, "answer_vocab": answer_vocab}
    with open(config.get("vocab_path"), "w") as f:
        json.dump(vocab, f)
    print("Vocab created")
