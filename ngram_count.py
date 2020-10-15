from gensim.models.phrases import Phrases
from gensim.utils import any2unicode, any2utf8

import spacy

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

nlp = spacy.load('pt', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


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
    return [token.text for token in doc if not should_discard(token)]
    # for sentence in doc.sents:
    #     return [token.text for token in sentence if not should_discard(token)]


stop_words = get_custom_stop_words()

sentences = ['mencionadas insurgências restam de há muito superadas.',
             'nesse sentido:',
             'tst - recurso de revista rr 1473005620085030137 (tst).',
             'data de publicação: 14/08/2015',
             'ementa: i - agravo de instrumento em recurso de revista da reclamada.',
             'justiça gratuita.',
             '"demonstrada divergência jurisprudencial específica, impõe-se o provimento do agravo de instrumento para determinar o processamento do recurso de revista da reclamada."',
             'agravo de instrumento provido.',
             'ii - recurso de revista da reclamada 1 - sindicato.',
             'substituição processual.']

# tokenized_sentences = [[word for word in sentence.split()] for sentence in sentences]
tokenized_sentences = [get_relevant_tokens(sentence) for sentence in sentences]

pruned_words, counters, total_words = Phrases.learn_vocab(sentences=tokenized_sentences,
                                                          max_vocab_size=800000000,
                                                          common_terms=stop_words,
                                                          progress_per=1)

counters = sorted(counters.items(), key=lambda key_value: key_value[1], reverse=True)

count = 0
for key, value in counters:
    count += 1
    print(any2unicode(key), value)
print(count)

bigram_model = Phrases(tokenized_sentences, max_vocab_size=800000000, progress_per=1, threshold=0.5, min_count=2,
                       common_terms=stop_words, scoring='npmi')
for sentence in tokenized_sentences:
    bigram_sentence = u' '.join(bigram_model[sentence])
    print(bigram_sentence + '\n')
