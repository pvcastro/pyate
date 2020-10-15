from pathlib import Path

from gensim.utils import any2utf8, any2unicode
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.word2vec import PathLineSentences, LineSentence
from gensim.models.phrases import Phrases, Phraser

import spacy
from spacy.util import minibatch

from split_datalawyer import SentenceSplit

from joblib import Parallel, delayed
from functools import partial
import multiprocessing, csv, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

nlp = spacy.load('pt', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
sentence_split = SentenceSplit()

documents_path = Path('/media/discoD/Data_Lawyer/Jurimetria/Corpora/Models/5B/documents')
unigram_sentences_path = documents_path / 'unigram_sentences.csv'
bigram_model_path = documents_path / 'bigram_model_all'
bigram_sentences_path = documents_path / 'bigram_sentences.csv'
trigram_model_path = documents_path / 'trigram_model_all'
trigram_sentences_path = documents_path / 'trigram_sentences.csv'
quadgram_model_path = documents_path / 'quadgram_model_all'
quadgram_sentences_path = documents_path / 'quadgram_sentences.csv'


def get_custom_stop_words():
    spacy_stop_words = set(nlp.Defaults.stop_words)
    custom_stop_words = set(
        ['a', 'agora', 'ainda', 'alem', 'algum', 'alguma', 'algumas', 'alguns', 'alguém', 'além', 'ambas', 'ambos',
         'ampla', \
         'amplas', 'amplo', 'amplos', 'and', 'ante', 'antes', 'ao', 'aonde', 'aos', 'apos', 'após', 'aquela',
         'aquelas', \
         'aquele', 'aqueles', 'aquilo', 'as', 'assim', 'através', 'até', 'cada', 'coisa', 'coisas', 'com', 'como',
         'contra', \
         'contudo', 'cuja', 'cujas', 'cujo', 'cujos', 'côm', 'da', 'daquele', 'daqueles', 'das', 'data', 'de',
         'dela', 'delas', \
         'dele', 'deles', 'demais', 'depois', 'desde', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas',
         'deste', \
         'destes', 'deve', 'devem', 'devendo', 'dever', 'deveria', 'deveriam', 'deverá', 'deverão', 'devia',
         'deviam', \
         'dispoe', 'dispoem', 'dispõe', 'dispõem', 'disse', 'disso', 'disto', 'dito', 'diversa', 'diversas',
         'diversos', 'diz', \
         'dizem', 'do', 'dos', 'durante', 'dà', 'dàs', 'dá', 'dás', 'dê', 'e', 'ela', 'elas', 'ele', 'eles', 'em',
         'enquanto', \
         'entao', 'entre', 'então', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'estamos', 'estas',
         'estava', \
         'estavam', 'este', 'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver',
         'estivera', \
         'estiveram', 'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéramos', 'estivéssemos', 'estou',
         'està', \
         'estàs', 'está', 'estás', 'estávamos', 'estão', 'eu', 'fazendo', 'fazer', 'feita', 'feitas', 'feito',
         'feitos', 'foi', \
         'fomos', 'for', 'fora', 'foram', 'forem', 'formos', 'fosse', 'fossem', 'fui', 'fôramos', 'fôssemos',
         'grande', \
         'grandes', 'ha', 'haja', 'hajam', 'hajamos', 'havemos', 'havia', 'hei', 'houve', 'houvemos', 'houver',
         'houvera', \
         'houveram', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houvermos', 'houverá',
         'houverão', \
         'houveríamos', 'houvesse', 'houvessem', 'houvéramos', 'houvéssemos', 'há', 'hão', 'isso', 'isto', 'já',
         'la', 'lhe', \
         'lhes', 'lo', 'logo', 'lá', 'mais', 'mas', 'me', 'mediante', 'menos', 'mesma', 'mesmas', 'mesmo', 'mesmos',
         'meu', 'meus', \
         'minha', 'minhas', 'muita', 'muitas', 'muito', 'muitos', 'nº', 'na', 'nas', 'nem', 'nenhum', 'nessa',
         'nessas',
         'nesse', \
         'nesta', 'nestas', 'neste', 'ninguém', 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa',
         'nunca', 'ná', \
         'nás', 'não', 'nós', 'o', 'or', 'os', 'ou', 'outra', 'outras', 'outro', 'outros', 'para', 'pela', 'pelas',
         'pelo', 'pelos', \
         'pequena', 'pequenas', 'pequeno', 'pequenos', 'per', 'perante', 'pode', 'podendo', 'poder', 'poderia',
         'poderiam', \
         'podia', 'podiam', 'pois', 'por', 'porque', 'porquê', 'portanto', 'porém', 'posso', 'pouca', 'poucas',
         'pouco', 'poucos', \
         'primeiro', 'primeiros', 'proprio', 'própria', 'próprias', 'próprio', 'próprios', 'pôde', 'quais', 'qual',
         'qualquer', \
         'quando', 'quanto', 'quantos', 'quaís', 'que', 'quem', 'quer', 'quê', 'se', 'seja', 'sejam', 'sejamos',
         'sem', 'sempre', \
         'sendo', 'ser', 'serei', 'seremos', 'seria', 'seriam', 'será', 'serão', 'seríamos', 'seu', 'seus', 'si',
         'sido', 'sob', \
         'sobre', 'somos', 'sou', 'sua', 'suas', 'são', 'só', 'tal', 'talvez', 'tambem', 'também', 'tampouco', 'te',
         'tem', 'temos', \
         'tendo', 'tenha', 'tenham', 'tenhamos', 'tenho', 'ter', 'terei', 'teremos', 'teria', 'teriam', 'terá',
         'terão', 'teríamos', \
         'teu', 'teus', 'teve', 'ti', 'tido', 'tinha', 'tinham', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram',
         'tiverem', \
         'tivermos', 'tivesse', 'tivessem', 'tivéramos', 'tivéssemos', 'toda', 'todas', 'todavia', 'todo', 'todos',
         'tu', 'tua', \
         'tuas', 'tudo', 'tém', 'têm', 'tínhamos', 'um', 'uma', 'umas', 'uns', 'vendo', 'ver', 'vez', 'vindo', 'vir',
         'você', \
         'vocês', 'vos', 'vós', 'à', 'às', 'á', 'ás', 'ão', 'è', 'é', 'éramos', 'êm', 'ò', 'ó', 'õ', 'última',
         'últimas', 'último', \
         'últimos'])
    custom_stop_words.update(spacy_stop_words)
    return [any2utf8(stop_word) for stop_word in custom_stop_words]


def should_discard(token):
    return token.is_punct or token.is_space


def get_relevant_tokens(sentence):
    doc = nlp(sentence)
    return [token.lower_ for token in doc if not should_discard(token)]


class DataLawyerCorpus(object):

    def __init__(self, documents_path, year, court, debug=False):
        self.documents_path = documents_path
        self.year = year
        self.court = court
        self.debug = debug
        self.doc_count = 0

    def __iter__(self):
        for year_directory in self.documents_path.iterdir():
            if year_directory.is_dir():
                if not self.year or (self.year and year_directory.name == self.year):
                    for court_directory in year_directory.iterdir():
                        if not self.court or (self.court and court_directory.name == self.court):
                            for document in court_directory.iterdir():
                                if document.is_file():
                                    self.doc_count += 1
                                    if self.doc_count % 100 == 0 and self.debug:
                                        print('Iterated {} documents so far'.format(self.doc_count))
                                    contents = read_document(document).strip()
                                    if len(contents) > 0:
                                        yield contents


def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space


def read_document(document):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """

    with document.open(mode='r', encoding='utf-8') as file:
        return file.read()


def get_csv_writer(corpus_out_path):
    out_file = corpus_out_path.open(mode='a', encoding='utf-8')
    return csv.writer(out_file, quoting=csv.QUOTE_MINIMAL)


def tokenize_sentence_corpus(corpus_out_path, batch_id, documents):
    csv_writer = get_csv_writer(corpus_out_path)
    for document in documents:
        rows = []
        try:
            sentences = sentence_split.get_sentences(document, split_by_semicolon=False)
            if sentences and len(sentences) > 0:
                tokenized_sentences = [get_relevant_tokens(sentence) for sentence in sentences]
                if tokenized_sentences and len(tokenized_sentences) > 0:
                    for tokenized_sentence in tokenized_sentences:
                        if len(tokenized_sentence) > 0:
                            rows.append([' '.join(tokenized_sentence)])
                    csv_writer.writerows(rows)
        except KeyError as ke:
            print(document)
            raise ke


def process_texts(documents_path, year, court, corpus_out_path, batch_size=100, n_jobs=multiprocessing.cpu_count(),
                  debug=False):
    print("Processing texts...")

    if unigram_sentences_path.exists():
        unigram_sentences_path.unlink()

    partitions = minibatch(DataLawyerCorpus(documents_path, year, court, debug), size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    do = delayed(partial(tokenize_sentence_corpus, corpus_out_path))
    tasks = (do(i, batch) for i, batch in enumerate(partitions))

    executor(tasks)


# process_texts(documents_path, year='2020', court='01', corpus_out_path=unigram_sentences_path, batch_size=8, n_jobs=2,
#               debug=True)

stop_words = get_custom_stop_words()

pruned_words, counters, total_words = Phrases.learn_vocab(sentences=LineSentence(unigram_sentences_path),
                                                          max_vocab_size=800000000,
                                                          common_terms=stop_words,
                                                          progress_per=100)

counters = sorted(counters.items(), key=lambda key_value: key_value[1], reverse=True)

count = 0
for key, value in counters:
    count += 1
    print(any2unicode(key), value)
print(count)

bigram_model = Phrases(LineSentence(unigram_sentences_path), max_vocab_size=800000000, progress_per=100, threshold=0.5,
                       min_count=100, common_terms=stop_words, scoring='npmi')
for sentence in LineSentence(unigram_sentences_path):
    bigram_sentence = u' '.join(bigram_model[sentence])
    print(bigram_sentence + '\n')
