"""
Author: yitong 2969413251@qq.com
Date: 2023-05-20 10:26:44
"""
import toolz, re
from collections import Counter


def wordpunct_tokenize(text):
    _pattern = r"\w+|[^\w\s]+"
    _regexp = re.compile(_pattern, flags=re.UNICODE | re.MULTILINE | re.DOTALL)
    return _regexp.findall(text)


class BPETokenizer:
    def __init__(
        self,
        vocab_size=1000,
        lowercase=True,
        basic_tokenizer=wordpunct_tokenize,
        unk="<UNK>",
        sep="<SEP>",
        pad="<PAD>",
        cls="<CLS>",
        mask="<MASK>",
        user_specials=None,
    ):
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.basic_tokenizer = basic_tokenizer
        self.unk, self.sep, self.pad, self.cls, self.mask = unk, sep, pad, cls, mask
        self.special = [unk, sep, pad, cls, mask]
        self.special.extend(user_specials if user_specials else [])

    def load(self, vocab_fn=None, vocab=None):
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = [line.strip() for line in open(vocab_fn, "r").readlines()]
        self.vocab_size = len(self.vocab)

    def fit(self, corpus: list, max_steps=10000, out_fn="vocab.txt"):
        """
        train the tokenizer, return the trained vocabulary
        """

        # count the initial dict

        if self.lowercase:
            corpus = [s.lower() for s in corpus]

        word_corpus = Counter(
            [
                tuple(data) + ("</w>",)
                for data in toolz.concat(map(self.basic_tokenizer, corpus))
            ]
        )
        vocab = self._count_vocab(word_corpus)

        # merge high-frequency binary in the vocab step by step
        for i in range(max_steps):
            word_corpus, bi_cnt = self._fit_step(word_corpus)
            vocab = self._count_vocab(word_corpus)
            if len(vocab) >= self.vocab_size or bi_cnt < 0:
                break

        # insert some speical words
        for s in self.special:
            if s not in vocab:
                vocab.insert(0, (s, 99999))

        # save the vocab
        with open(out_fn, "w") as f:
            f.write("\n".join([w for w, _ in vocab]))
        # sorted vocab?
        self.vocab = [token for token, _ in vocab]
        return vocab

    def _count_vocab(self, word_corpus):
        _r = Counter(
            [
                data
                for data in toolz.concat(
                    [word * cnt for word, cnt in word_corpus.items()]
                )
            ]
        )
        _r = sorted(_r.items(), key=lambda x: -x[1])
        return _r

    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()

        # step as 1, slide_window as 2,
        # count the frequency of binary-tuple by sliding every words
        for token, count in word_corpus.items():
            if len(token) < 2:
                continue
            for bigram in toolz.sliding_window(ngram, token):
                bigram_counter[bigram] += count

        # select the most frequent binary
        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=bigram_counter.get)
        else:
            return word_corpus, -1

        bi_cnt = bigram_counter.get(max_bigram)

        # replace binary words with a token consisting of bi_cnt
        words_tokens = list(word_corpus.keys())
        for tokens in words_tokens:
            _new_token = tuple(
                " ".join(tokens)
                .replace(" ".join(max_bigram), "".join(max_bigram))
                .split(" ")
            )
            if _new_token != tokens:
                word_corpus[_new_token] = word_corpus[tokens]
                word_corpus.pop(tokens)
        return word_corpus, bi_cnt

    def tokenize(self, text: str, add_pre=None, add_mid=None, add_post="</w>"):
        """
        transfer text into tokens
        """
        all_tokens = []
        if self.lowercase:
            text = text.lower()

        for token in self.basic_tokenizer(text):
            new_token = []
            token = list(token)
            token = [add_pre] + token if add_pre else token
            token = token + [add_post] if add_post else token
            start, end = 0, len(token)

            # find the maximum length of the sub_token
            while start < end:
                sub_token = "".join(token[start:end])

                if start > 0 and add_mid:
                    sub_token = add_mid + sub_token

                if sub_token in self.vocab:
                    new_token.append(sub_token)
                    start = end
                    end = len(token)
                elif end - start == 1:
                    new_token.append(self.unk)
                    start = end
                    end = len(token)
                else:
                    end -= 1
            all_tokens.append(new_token)
        return all_tokens

    def _token2id(self, token):
        if token in self.vocab:
            return self.vocab.index(token)
        return self.vocab.index(self.unk)

    def _id2token(self, id):
        return self.vocab[id]

    def encode(self, text: str):
        """
        transfer text to token_ids
        """
        token_list = self.tokenize(text)
        ids_list = [
            list(map(lambda x: self._token2id(x), tokens)) for tokens in token_list
        ]
        return ids_list

    def decode(self, token_ids):
        """
        transfer token_ids to text
        """
        sentences = []
        for ids in token_ids:
            sentence = list(map(lambda x: self._id2token(x), ids))
            sentence = ''.join(sentence).replace('</w>', ' ')
            sentences.append(sentence)
        return sentences

class WordPieceTokenizer(BPETokenizer):
    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()
        unigram_counter = Counter()

        # step as 1, slide_ window size as 2
        # slide on each word to count the the frquency of binaries
        for token, count in word_corpus.items(): 
            for c in token:
                unigram_counter[c] += count
            if len(token) < 2:
                for bigram in toolz.sliding_window(ngram, token):
                    bigram_counter[bigram] += count
        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=lambda x: bigram_counter.get(x) / (unigram_counter.get(x[0]) * unigram_counter.get(x[1])))
        else:
            return word_corpus, -1
        bi_cnt = max(bigram_counter.values())

        words_tokens = list(word_corpus.keys())
        for tokens in words_tokens:
            _new_tokens = tuple(' '.join(tokens).replace(' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if _new_tokens != tokens:
                word_corpus[_new_tokens] = word_corpus[tokens]
                word_corpus.pop(tokens)
        return word_corpus, bi_cnt

def bpe_example():
    corpus = '''
            Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
            Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
            Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
            Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
            Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
        '''
    corpus = [s.strip() for s in corpus.strip().split('\n')]
    bpe = BPETokenizer(vocab_size=60)
    bpe.fit(corpus)

    print(bpe.vocab)
    print(bpe.tokenize('Object raspberrypi functools dict kwargs'))
    print(bpe.encode('Object raspberrypi functools dict kwargs'))
    print(bpe.decode(bpe.encode('Object raspberrypi functools dict kwargs')))

def wp_sample():
    corpus = '''
            Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
            Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
            Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
            Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
            Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
        '''
    corpus = [s.strip() for s in corpus.strip().split('\n')]
    bpe = WordPieceTokenizer(vocab_size=60)
    bpe.fit(corpus)

    print(bpe.vocab)
    print(bpe.tokenize('Object raspberrypi functools dict kwargs'))
    print(bpe.encode('Object raspberrypi functools dict kwargs'))
    print(bpe.decode(bpe.encode('Object raspberrypi functools dict kwargs')))

if __name__ == '__main__':
    bpe_example()   
    wp_sample()
