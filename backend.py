# !gcloud dataproc clusters list --region us-central1
# !pip install -q google-cloud-storage==1.43.0
# !pip install -q graphframes

from google.cloud import storage
import pickle
from google.cloud import *
import numpy as np
import math
from time import time
import pandas as pd
import sys
import builtins
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
from pathlib import Path
import pickle
from contextlib import closing
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
BLOCK_SIZE = 1999998
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self.client = storage.Client()
        self._bucket_name = self.client.bucket(bucket_name)
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()

                file_name = self._f.name
                blob = self._bucket_name.blob(f"postings_gcp/{file_name}")
                blob.upload_from_filename(file_name)

                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}
        self.client = storage.Client()
        self._bucket_name = self.client.bucket(bucket_name)

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                blob = self._bucket_name.get_blob(f'postings_gcp/{f_name}')
                self._open_files[f_name] = blob.open('rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = builtins.min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # DL represent a dict of {doc_id : len of the doc}
        self.DL = {}
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def posting_lists_iter_query(self, query_to_search):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
            for a specific query.
        """
        with closing(MultiFileReader()) as reader:
            for w in query_to_search:
                posting_list = []
                if w in self.posting_locs:
                    locs = self.posting_locs[w]
                    b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                    for i in range(self.df[w]):
                        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                        posting_list.append((doc_id, tf))
                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, prefix):  # prefix = indexB / indexT  / indexA
        ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        and writes it out to disk as files named {bucket_id}_XXX.bin under the
        current directory. Returns a posting locations dictionary that maps each
        word to the list of files and offsets that contain its posting list.
        Parameters:
        -----------
          b_w_pl: tuple
            Containing a bucket id and all (word, posting list) pairs in that bucket
            (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
        Return:
          posting_locs: dict
            Posting locations for each of the words written out in this bucket.
        '''
        posting_locs = defaultdict(list)
        bucket, list_w_pl = b_w_pl
        fullname = prefix + str(bucket)

        with closing(MultiFileWriter('.', fullname, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big') for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)

            file_name = writer._f.name
            blob = writer._bucket_name.blob(f"postings_gcp/{file_name}")
            blob.upload_from_filename(file_name)
        return posting_locs


bucket_name = 'adi3158'
client = storage.Client()
blobs = client.list_blobs(bucket_name)


class process:

    def __init__(self):
        x, y, z = self.load_dicts()
        self.page_rank_dict = x
        self.page_view_dict = y
        self.title_dict = z

        x, y, z = self.load_indecies()
        self.invertedT = x
        self.invertedA = y
        self.invertedB = z

    def load_dicts(self):
        '''
        load to the memory all the data we need to execute a search.
        all the outsource dicts, page rank and page view.
        '''
        # load pagerank
        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("page_rank_dict.pckl"):
                continue
            with blob.open("rb") as f:
                page_rank_dict = pickle.load(f)
                print("create new pagerank")
        # load page view
        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("pageviews-202108-user.pkl"):
                continue
            with blob.open("rb") as f:
                page_view_dict = pickle.load(f)
                print("create new page view")
        # load title dict
        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("title_dict.pkl"):
                continue
            with blob.open("rb") as f:
                title_dict = pickle.load(f)
                print("create title dict")
        return page_rank_dict, page_view_dict, title_dict

    def load_indecies(self):
        '''
        load all all the indices.
        '''
        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexT.pkl"):
                continue
            with blob.open("rb") as f:
                invertedT = pickle.load(f)
                print("create invertedT")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexA.pkl"):
                continue
            with blob.open("rb") as f:
                invertedA = pickle.load(f)
                print("create invertedA")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexB.pkl"):
                continue
            with blob.open("rb") as f:
                invertedB = pickle.load(f)
                print("create invertedB")
        return invertedT, invertedA, invertedB

    def tokenize(self, text):
        '''
        take string and returns list of tokens.
        '''
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in all_stopwords]
        return list_of_tokens

    def replace_socre_title(self, doc_id):
        '''
        take doc id and return the title of this doc id.
        '''
        return " ".join(self.title_dict[doc_id])

    def get_rank(self, lst):
        '''
        returns the page rank of the docs id in the list param.
        '''
        res = []
        for doc_id in lst:
            if doc_id in self.page_rank_dict.keys():
                res.append(self.page_rank_dict[doc_id])
            else:
                res.append(0)
        return res

    def get_pageview(self, lst):
        '''
        returns the page view of the docs id in the list param.
        '''
        res = []
        for doc_id in lst:
            if doc_id in self.page_view_dict.keys():
                res.append(self.page_view_dict[doc_id])
            else:
                res.append(0)
        return res

    def search_title(self, query):
        '''
        param: query to search
        return: list of tuples of (doc_id, title).
        '''
        list_query = self.tokenize(query)
        qDocIdApper = {}  # {309: 2, 2175: 1, 28239: 1, 28240: 1, 24001: 1, 6112: 1, 9232: 1}
        for token in list_query:
            words, pl = zip(*self.invertedT.posting_lists_iter_query([token]))
            for tup in pl[0]:  # [(309, 1), (2175, 1)]
                docId = tup[0]
                if docId in qDocIdApper.keys():
                    qDocIdApper[docId] += 1
                else:
                    qDocIdApper[docId] = 1
        # after we got the (doc_id,score), we replace the score with the title.
        lst1 = sorted(qDocIdApper.items(), key=lambda x: x[1], reverse=True)
        with_title_list = [(x[0], self.replace_socre_title(x[0])) for x in lst1]
        return with_title_list

    def search_anchor(self, query):
        '''
        param: query to search
        return: list of tuples of (doc_id, title).
        '''
        list_query = self.tokenize(query)
        qDocIdApper = {}  # {309: 2, 2175: 1, 28239: 1, 28240: 1, 24001: 1, 6112: 1, 9232: 1}
        for token in list_query:
            words, pls = zip(*self.invertedA.posting_lists_iter_query([token]))  # add bin directory !
            for tup in pls[0]:  # [(309, 1), (2175, 1)]
                docId = tup[0]
                if docId in qDocIdApper.keys():
                    qDocIdApper[docId] += 1
                else:
                    qDocIdApper[docId] = 1
        # after we got the (doc_id,score), we replace the score with the title.
        lst1 = sorted(qDocIdApper.items(), key=lambda x: x[1], reverse=True)
        with_title_list = [(x[0], self.replace_socre_title(x[0])) for x in lst1]
        return with_title_list

    def merge_results(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=100):
        '''
        this functions gets the results from 2 index search and merge it together according to the weights.
        At the end it cuts the results to N top results.
        '''
        lst2 = []
        for quer_id in title_scores.keys():
            lst_title = title_scores[quer_id]
            lst_body = body_scores[quer_id]
            lst = []
            temp_dic = {}
            for tup_title in lst_title:
                temp_dic[tup_title[0]] = tup_title[1] * title_weight

            for tup_body in lst_body:
                if tup_body[0] in temp_dic.keys():
                    temp_dic[tup_body[0]] += text_weight * tup_body[1]
                else:
                    temp_dic[tup_body[0]] = text_weight * tup_body[1]
            lst = [(k, v) for k, v in temp_dic.items()]
            lst2 = sorted(lst, reverse=True, key=lambda x: x[1])[:N]
        return lst2

    def wcalc(self, score, rank):
        '''
        func that give some weight to the page rank.
        '''
        return score + (2 * (score * rank)) / (score + rank)

    def w_rank_view(self, res):  # get list
        '''
        gets list of resluts
        returns: list of results after calc with the page rank
        '''
        new_res = []
        for tup in res:
            score = tup[1]
            rank = self.get_rank([tup[0]])[0]
            new_socre = self.wcalc(score, rank)
            new_res.append((tup[0], new_socre))
        return new_res

    def search_body(self, q):
        '''
        this function received a query to search and search only in the body of the text of all corpus.
        returns: list of tuples (doc_id , title )
        '''
        ourk = 100
        dic_query = {}
        query = self.tokenize(q)

        dic_query[1] = query
        words, pls = zip(*self.invertedB.posting_lists_iter_query(query))  # need to get list
        B = self.get_topN_score_for_queries(dic_query, self.invertedB, words, pls,
                                            ourk)  # {1: [(14044751, 0.57711), (49995633, 0.56526),
        new_res = self.w_rank_view(B[1])
        res_sort = sorted(new_res, reverse=True, key=lambda x: x[1])
        # after we got the (doc_id,score), we replace the score with the title.
        with_title = [(str(x[0]), self.replace_socre_title(x[0])) for x in res_sort]
        return with_title

    def search(self, q, a=0.78, b=0.22, c=0.3, d=0.7):  # list of string[make, pasta]
        '''
        The main search function.
        This function received weights, calc results for each index separately and than call for merge.
        '''

        ourK = 100
        dic_query = {}
        # tokenize the query before search.
        query = self.tokenize(q)
        dic_query[1] = query
        # search in each index alone.
        words, pls = zip(*self.invertedB.posting_lists_iter_query(query))
        B = self.get_topN_score_for_queries(dic_query, self.invertedB, words, pls, ourK)
        words, pls = zip(*self.invertedA.posting_lists_iter_query(query))
        A = self.get_topN_score_for_queries(dic_query, self.invertedA, words, pls, ourK)
        words, pls = zip(*self.invertedT.posting_lists_iter_query(query))
        T = self.get_topN_score_for_queries(dic_query, self.invertedT, words, pls, ourK)

        # merge the results by thw weights.
        TB = self.merge_results(A, T, a, b, ourK)
        mergeBTA = self.merge_results({1: TB}, B, c, d, ourK)
        new_res = self.w_rank_view(mergeBTA)
        # sort the new results and replace score with title.
        res_sort = sorted(new_res, reverse=True, key=lambda x: x[1])
        with_title = [(int(x[0]), self.replace_socre_title(x[0])) for x in res_sort]
        return with_title

    # search body help function:
    def generate_query_tfidf_vector(self, query_to_search, index):
        '''
        make TF-IDF vector out of the query.
        '''
        epsilon = .0000001
        Q = np.zeros(len(query_to_search))
        term_vector = query_to_search
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.df.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
                df = index.df[token]
                idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):
        '''
        returns only the docs id that relevant for the terms in the query.
        each doc id and term have their score.
        '''
        candidates = {}
        N = len(index.DL)
        for term in np.unique(query_to_search):
            if term in index.df.keys():
                list_of_doc = pls[words.index(term)]
                for doc_id, freq in list_of_doc:
                    score = (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + score
        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls):
        '''
        We do not need to utilize all document. Only the documents which have corresponding terms with the query.
        '''
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words, pls)
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        return candidates_scores, unique_candidates

    def cosine_similarity(self, candi, uni, query, Q, index):  # for query for candi
        '''
        calc cosine similarity for each doc id and all the relevant terms
        '''
        cosine_dic = {}
        b = np.linalg.norm(Q)
        for doc_id in uni:
            vec = []
            for term in query:
                if (doc_id, term) in candi.keys():
                    vec.append(candi[(doc_id, term)])
                else:
                    vec.append(0)
            y = np.array(vec)
            x = np.dot(y, Q)
            a = index.tfidf_dict[doc_id]
            tot = (x / (a * b))
            cosine_dic[doc_id] = tot
        return cosine_dic

    def get_top_n(self, sim_dict, N=100):
        '''
        returns sorted top n results.
        '''
        return sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    def get_topN_score_for_queries(self, queries_to_search, index, words, pls, N=100):
        '''
        returns dict with the final answers
        '''
        dic = {}
        for i in range(1, len(queries_to_search) + 1):
            candi, uni = self.generate_document_tfidf_matrix(queries_to_search[i], index, words, pls)
            Q = self.generate_query_tfidf_vector(queries_to_search[i], index)
            res = self.cosine_similarity(candi, uni, queries_to_search[i], Q, index)
            x = self.get_top_n(res, N)
            dic[i] = x
        return dic

    ''' check function '''

    def intersection(self, l1, l2):
        '''
        returns intersect between the lists.
        '''
        return list(set(l1) & set(l2))

    def precision_at_k(self, true_list, predicted_list, k=40):
        y = predicted_list[:k]
        x = self.intersection(y, true_list)
        res = (len(x) / k)
        return res

    def average_precision(self, true_list, predicted_list, k=40):
        preListK = predicted_list[:k]
        inter = self.intersection(true_list, preListK)
        lst = []
        for doc_id in inter:
            indx = preListK.index(doc_id) + 1
            precisionForDoc = self.precision_at_k(true_list, predicted_list, indx)
            lst.append(precisionForDoc)
        if len(lst) == 0:
            return 0
        avgPrec = builtins.sum(lst) / len(lst)
        return avgPrec

    def print_res(self, tfidf_queries_score_train, query, k1):
        '''
        gets our results and compute MAP@40, for every query we have in the training set.
        '''
        q = query
        our_res = tfidf_queries_score_train[1]
        our_docId = [x[0] for x in our_res]
        lst = self.average_precision(training_dic[q], our_docId)  # k=40
        lst2 = []
        c = 0
        for doc in our_docId:
            if doc in training_dic[q]:
                c += 1
                lst2.append((training_dic[q].index(doc), our_docId.index(doc)))
        print("map 40:", lst)
        print("our k: ", k1, " test k: ", len(training_dic[q]), "common: ", c)
