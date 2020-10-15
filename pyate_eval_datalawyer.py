import spacy

spacy.load('pt_core_news_sm')

from pathlib import Path
from pyate import combo_basic, term_extractor, cvalues
from gensim.models.word2vec import PathLineSentences

documents_path = Path('/media/discoD/Data_Lawyer/Jurimetria/Corpora/Models/5B/documents')

wiki_path = Path('/media/discoD/Corpora/Wikipedia/text')


class WikipediaCorpus(object):

    def __init__(self, wiki_path):
        self.wiki_path = wiki_path

    def __iter__(self):
        for sub_directory in self.wiki_path.iterdir():
            if sub_directory.is_dir():
                for sentences in PathLineSentences(str(sub_directory)):
                    yield sentences

    def __len__(self):
        doc_count = 0
        for sub_directory in self.wiki_path.iterdir():
            if sub_directory.is_dir():
                for _ in PathLineSentences(str(sub_directory)):
                    doc_count += 1
        return doc_count


class DataLawyerCorpus(object):

    def __init__(self, documents_path):
        self.documents_path = documents_path

    def __iter__(self):
        for year_directory in self.documents_path.iterdir():
            for court_directory in year_directory.iterdir():
                for document in court_directory.iterdir():
                    if document.is_file():
                        with document.open(mode='r', encoding='utf-8') as file:
                            yield file.read()

    def __len__(self):
        doc_count = 0
        for year_directory in self.documents_path.iterdir():
            for court_directory in year_directory.iterdir():
                for document in court_directory.iterdir():
                    if document.is_file():
                        doc_count += 1
        return doc_count


verbose = True
# print('Combo Basic:')
# print(combo_basic(string, verbose=verbose).sort_values(ascending=False))
# print('')
print('Term Extractor:')
print(term_extractor(technical_corpus=DataLawyerCorpus(documents_path),
                     general_corpus=WikipediaCorpus(wiki_path),
                     verbose=verbose, do_parallelize=False).sort_values(ascending=False))
# print('')
# print('C-Values:')
# print(cvalues(string, verbose=verbose).sort_values(ascending=False))
