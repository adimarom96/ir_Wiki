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

    def __init__(self, base_dir, name, bucket_name):  # 7 #('.',fullname,bucket_name)
        self._base_dir = Path(base_dir)
        self._name = name
        self.client = storage.Client()  # new ohad todo
        self._bucket_name = self.client.bucket(bucket_name)
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):  # *******************************************
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()

                file_name = self._f.name  # ohad todo change
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
        # DL[(doc_id)] = DL.get(doc_id, 0) + (len(tokens))
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
    def write_a_posting_list(b_w_pl, prefix):  # prefix = indexB / indexT ...
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
                locs = writer.write(b)  # ****************************** add index name todo new ohad
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

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("page_rank_dict.pckl"):
                continue
            with blob.open("rb") as f:
                page_rank_dict = pickle.load(f)
                print("cretae new pagerank")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("pageviews-202108-user.pkl"):
                continue
            with blob.open("rb") as f:
                page_view_dict = pickle.load(f)
                print("cretae new pageview")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("title_dict.pkl"):
                continue
            with blob.open("rb") as f:
                title_dict = pickle.load(f)
                print("cretae title dict")
        return page_rank_dict, page_view_dict, title_dict

    def load_indecies(self):
        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexT.pkl"):
                continue
            with blob.open("rb") as f:
                invertedT = pickle.load(f)
                print("cretae invertedT")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexA.pkl"):
                continue
            with blob.open("rb") as f:
                invertedA = pickle.load(f)
                print("cretae invertedA")

        for blob in client.list_blobs(bucket_name):
            if not blob.name.endswith("indexB.pkl"):
                continue
            with blob.open("rb") as f:
                invertedB = pickle.load(f)
                print("cretae invertedB")
        return invertedT, invertedA, invertedB

    def tokenize(self, text):
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in all_stopwords]
        return list_of_tokens

    def replace_socre_title(self, doc_id):
        # print(title_dict[doc_id])
        return " ".join(self.title_dict[doc_id])

    # fornt end function
    def get_rank(self, lst):
        res = []
        for doc_id in lst:
            if doc_id in self.page_rank_dict.keys():
                res.append(self.page_rank_dict[doc_id])
            else:
                res.append(0)
        return res

    def get_pageview(self, lst):
        res = []
        for doc_id in lst:
            if doc_id in self.page_view_dict.keys():
                res.append(self.page_view_dict[doc_id])
            else:
                res.append(0)
        return res

    def search_title(self, query):
        list_query = self.tokenize(query)
        qDocIdApper = {}  # {309: 2, 2175: 1, 28239: 1, 28240: 1, 24001: 1, 6112: 1, 9232: 1}
        for token in list_query:

            # add bin directory !
            words, pl = zip(*self.invertedT.posting_lists_iter_query(
                [token]))  # todo: send pasta+make/make+pasta and check if the same
            for tup in pl[0]:  # [(309, 1), (2175, 1)]
                docId = tup[0]
                if docId in qDocIdApper.keys():
                    qDocIdApper[docId] += 1
                else:
                    qDocIdApper[docId] = 1

        lst1 = sorted(qDocIdApper.items(), key=lambda x: x[1], reverse=True)
        with_title_list = [(x[0], self.replace_socre_title(x[0])) for x in lst1]
        return with_title_list

    def search_anchor(self, query):
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

        lst1 = sorted(qDocIdApper.items(), key=lambda x: x[1], reverse=True)
        with_title_list = [(x[0], self.replace_socre_title(x[0])) for x in lst1]
        return with_title_list

    # check function and run
    def intersection(self, l1, l2):
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
        q = query
        #     q = " ".join(query)
        #     q1 =" ".split(query)
        #     print("join : ", q, " split : ", q1 , " query : ",query)
        our_res = tfidf_queries_score_train[1]
        our_docid = [x[0] for x in our_res]
        lst = self.average_precision(training_dic[q], our_docid)  # k=40
        lst2 = []
        c = 0
        for doc in our_docid:
            if doc in training_dic[q]:
                c += 1
                lst2.append((training_dic[q].index(doc), our_docid.index(doc)))
        print("map 40:", lst)
        print("our k: ", k1, " test k: ", len(training_dic[q]), "common: ", c)
        # print(lst2)

    def merge_results(self, title_scores, body_scores, title_weight=0.5, text_weight=0.5, N=3):

        dic = {}
        for quer_id in title_scores.keys():  ## if keys not equal might problem!
            lst_title = title_scores[quer_id]
            lst_body = body_scores[quer_id]
            lst = []
            temp_dic = {}  # key id value weighted score
            for tup_title in lst_title:
                temp_dic[tup_title[0]] = tup_title[1] * title_weight

            for tup_body in lst_body:
                if tup_body[0] in temp_dic.keys():
                    temp_dic[tup_body[0]] += text_weight * tup_body[1]
                else:
                    temp_dic[tup_body[0]] = text_weight * tup_body[1]
            lst = [(k, v) for k, v in temp_dic.items()]
            # lst.sort(key=lambda tup: tup[1],reverse=True)[:N]
            lst2 = sorted(lst, reverse=True, key=lambda x: x[1])[:N]
        return lst2

    def wcalc(self, score, rank):
        return score + (2 * (score * rank)) / (score + rank)

    def w_rank_view(self, res):  # get list
        new_res = []
        for tup in res:
            score = tup[1]
            rank = self.get_rank([tup[0]])[0]
            new_socre = self.wcalc(score, rank)
            new_res.append((tup[0], new_socre))
        return new_res

    def search_body(self,q):
        ourk = 87
        dic_query = {}
        query = self.tokenize(q)

        dic_query[1] = query
        words, pls = zip(*self.invertedB.posting_lists_iter_query(query))  # need to get list
        B = self.get_topN_score_for_queries(dic_query,self. invertedB, words, pls,
                                       ourk)  # {1: [(14044751, 0.57711), (49995633, 0.56526),
        new_res = self.w_rank_view(B[1])
        res_sort = sorted(new_res, reverse=True, key=lambda x: x[1])
        with_title = [(str(x[0]), self.replace_socre_title(x[0])) for x in res_sort]
        return with_title


    def search(self, q, a=0.78, b=0.22, c=0.3, d=0.7):  # list of string[make, pasta]
        t_start = time()
        ourk = 87
        dic_query = {}
        query = self.tokenize(q)

        dic_query[1] = query
        words, pls = zip(*self.invertedB.posting_lists_iter_query(query))  # need to get list
        B = self.get_topN_score_for_queries(dic_query, self.invertedB, words, pls,
                                            ourk)  # {1: [(14044751, 0.57711), (49995633, 0.56526),
        words, pls = zip(*self.invertedA.posting_lists_iter_query(query))
        A = self.get_topN_score_for_queries(dic_query, self.invertedA, words, pls, ourk)
        words, pls = zip(*self.invertedT.posting_lists_iter_query(query))
        T = self.get_topN_score_for_queries(dic_query, self.invertedT, words, pls, ourk)

        TB = self.merge_results(A, T, a, b, ourk)
        mergeBTA = self.merge_results({1: TB}, B, c, d, ourk)

        new_res = self.w_rank_view(mergeBTA)
        res_sort = sorted(new_res, reverse=True, key=lambda x: x[1])
        with_title = [(int(x[0]), self.replace_socre_title(x[0])) for x in res_sort]
        return with_title

    # search body help function:
    def generate_query_tfidf_vector(self, query_to_search, index):
        epsilon = .0000001
        Q = np.zeros(len(query_to_search))
        term_vector = query_to_search
        counter = Counter(query_to_search)
        for token in np.unique(query_to_search):
            if token in index.df.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = index.df[token]
                idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf

                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_to_search, index, words, pls):
        candidates = {}
        N = len(index.DL)
        for term in np.unique(query_to_search):
            if term in index.df.keys():
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = []
                for doc_id, freq in list_of_doc:
                    score = (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + score
        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, index, words, pls):
        t_start = time()
        qdict = {}
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, index, words,
                                                                    pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        return candidates_scores, unique_candidates

    def cosine_similarity(self, candi, uni, query, Q, index):  # for query for candi
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

    def get_top_n(self, sim_dict, N=3):  # todo N=100 ?
        return sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    def get_topN_score_for_queries(self, queries_to_search, index, words, pls, N=3):
        dic = {}
        for i in range(1, len(queries_to_search) + 1):
            candi, uni = self.generate_document_tfidf_matrix(queries_to_search[i], index, words, pls)
            Q = self.generate_query_tfidf_vector(queries_to_search[i], index)
            res = self.cosine_similarity(candi, uni, queries_to_search[i], Q, index)
            x = self.get_top_n(res, N)
            dic[i] = x
        return dic


python = [23862, 23329, 53672527, 21356332, 4920126, 5250192, 819149, 46448252, 83036, 88595, 18942, 696712, 2032271,
          1984246, 5204237, 645111, 18384111, 3673376, 25061839, 271890, 226402, 2380213, 1179348, 15586616, 50278739,
          19701, 3596573, 4225907, 19160, 1235986, 6908561, 3594951, 18805500, 5087621, 25049240, 2432299, 381782,
          9603954, 390263, 317752, 38007831, 2564605, 13370873, 2403126, 17402165, 23678545, 7837468, 23954341,
          11505904, 196698, 34292335, 52042, 2247376, 15858, 11322015, 13062829, 38833779, 7800160, 24193668, 440018,
          54351136, 28887886, 19620, 23045823, 43003632, 746577, 1211612, 8305253, 14985517, 30796675, 51800, 964717,
          6146589, 13024, 11583987, 57294217, 27471338, 5479462]
data_science = [35458904, 2720954, 54415758, 233488, 12487489, 48364486, 487132, 416589, 27051151, 66414222, 18985040,
                4141563, 376707, 67092078, 61624594, 59591015, 49681, 55052729, 31663650, 35757264, 63995662, 3461736,
                56499662, 43003632, 26685, 10147369, 24932989, 66982525, 5323, 46207323, 8495, 31915311, 51552534,
                64057049, 45443335, 37849704, 42149032, 63198823, 38833779, 26880450, 39171985, 5213, 44133735,
                28486111, 48972018, 23943140, 17740009, 49954680, 28326718, 63016369, 56023027, 58731, 55207134,
                62798611, 65732625, 58255666, 68181697, 47892671, 1309220, 65499427, 58255600, 47638279, 46626475,
                46374359, 53343108, 173332, 7671, 18745015, 50565707, 57143357, 45112545, 63148404, 50533783, 51112472,
                49281083, 51505979, 1181008, 56731447, 59116880, 51578025, 21462612, 1765779, 39675445, 42253]
migraine = [21035, 36984150, 2702930, 25045060, 24224679, 2555865, 36579642, 310429, 22352499, 11495285, 22294424,
            234876, 40748148, 69893, 61962436, 62871079, 843361, 7362700, 16982268, 15712244, 5690287, 7362738, 600236,
            12410589, 26584776, 3332410, 20038918, 739855, 1015919, 14201682, 24361010, 53035710, 22901459, 57672434,
            4206029, 738384, 36579839, 188521, 15325435, 3602651, 40428462, 322197, 19592340, 3868233, 2385806, 2933438,
            23174077, 14001660, 2425344, 288328, 21381229, 26585811, 12652799, 322210, 51078678, 621531, 685130,
            11193835, 21197980, 21078348, 3108484, 692988, 31556991, 18741438, 3053003, 50977642, 55115883, 17208913,
            64269900, 54077917, 36666029, 50083054, 28245491, 5692662, 18353587, 1994895, 21364162, 20208066, 38574433,
            910244, 6154091, 67754025, 2132969, 61386909, 18600765, 579516]
chocolate = [7089, 6693851, 6672660, 23159504, 49770662, 167891, 2399742, 100710, 76723, 5290678, 54229, 3881415,
             3720007, 32652613, 1471058, 5239060, 1421165, 1730071, 1277918, 7419133, 17720623, 1765026, 19079925,
             1979873, 497794, 57947, 15007729, 85655, 4250574, 2386481, 228541, 55225594, 318873, 22631033, 27767967,
             7061714, 8175846, 3881265, 3046256, 606737, 845137, 16161419, 3098266, 54573, 11323402, 936243, 39280615,
             13687674, 47155656, 7151675, 43627595, 26879832, 43098662, 2333593, 349448, 2052211, 4432842, 56412300,
             1411016, 2152015, 3502051, 33372192, 61422444, 2385217, 1217747, 24315397, 7082459, 856246, 6050655,
             27162455, 52140446, 37243595, 36961531, 245067, 1148978, 1770825, 976322, 10300434, 7249348, 14945749,
             62851606, 637004, 16224368, 18509922]
how_to_make_pasta = [25659792, 50404581, 29178, 3080697, 90267, 2568868, 3450096, 49647732, 462173, 43911472, 20229,
                     40478583, 56643734, 21257512, 2387344, 59405867, 1330188, 12638661, 501757, 446865, 4468939,
                     25215235, 456363, 95411, 30916, 53487581, 30876926, 301932, 47764842, 426522, 579040, 54155622,
                     60535326, 23852947, 4275382, 67279077, 16591942, 334546, 602599, 3735620, 10296674, 858120,
                     30876121, 443480, 1038309, 50653758, 2258995, 34121672, 5382150, 884056, 3141956, 349722, 6745,
                     3511512, 35211682, 611752, 66963891, 43977806, 36742560, 2899729, 5413930, 61742595, 6972293,
                     14926, 42674415, 193957, 1950442, 3396753, 2269888, 40055348, 63609800, 1343426, 26078050, 44808,
                     42444204, 35034191, 9303405, 4627535, 59407816, 28732, 1187122, 6984468, 3328852, 56313776,
                     43853813, 3533082, 1032674, 39797382, 8892877]
Does_pasta_have_preservatives = [50404581, 301932, 30876121, 12727781, 37018026, 382599, 56232, 1032674, 202437,
                                 9759063, 2047222, 11309920, 15015154, 46482, 406363, 39813131, 9785087, 47840259,
                                 230716, 2240648, 49380722, 62153, 64083, 5355, 1330224, 17345999, 36969, 50577743,
                                 31919750, 1197035, 2175, 15003673, 54145741, 42801, 458008, 47150650, 67922, 39390739,
                                 901091, 28771786, 6984468, 416752, 382619, 198153, 49065540, 496821, 3112548, 15434651,
                                 237489, 47862672, 915309, 708662, 21699434, 550448, 22893145, 17055183, 32863238,
                                 13824676, 1300923, 746225, 33065713, 47770304, 32593, 11002, 20156275, 22735258,
                                 31156754, 32587, 2077960, 344611, 5652480, 31425310, 991758, 40956516, 594987, 1093416,
                                 26473291, 877461, 340356, 13679, 1558639, 11832915, 926863, 14953848, 42155809, 54923,
                                 276975, 66554, 5775715, 3133549, 1104286, 26951370, 3240723, 6523448, 3902658]
how_google_works = [44674524, 1092923, 12431, 224584, 43419164, 9874319, 4338696, 3190097, 9651444, 33321947, 26334893,
                    1497849, 47489893, 32639051, 2030607, 60903, 19133401, 42411494, 47799755, 4028754, 42960975,
                    24386465, 5339378, 1494648, 14181749, 58582001, 10062721, 33039125, 286747, 48736239, 33367993,
                    50575063, 55633178, 29403992, 34113322, 3660182, 25173473, 62438513, 60904, 3235536, 40867519,
                    5376827, 44424763, 58708106, 64302888, 187946, 29156200, 46426771, 35847782, 5376796, 25295524,
                    33768164, 46551547, 773423, 5376868, 49931371, 59539691, 22411575, 35326347, 46493906, 11451897,
                    10619416, 36891093, 40116717, 23533163, 879962, 502593, 3371574, 43194901, 41815118, 35673556,
                    6575642, 43997189, 5913182, 2126501, 52840911, 1566175, 42951365, 42694174, 22992426, 466299,
                    736238, 7301470, 1431181, 48653985, 51328172, 12003767]
what_is_information_retrieval = [15271, 494530, 442684, 19988623, 731640, 24997830, 10179411, 16635934, 33407925,
                                 11486091, 50716473, 35804330, 18550455, 21106742, 4694434, 26591446, 296950, 24963841,
                                 346470, 509628, 261193, 28688670, 10218640, 1897206, 39000674, 17785794, 38156944,
                                 9586885, 743971, 1185840, 7872152, 10328235, 36794719, 509624, 5818361, 25935906,
                                 22254915, 4881262, 39585214, 30882491, 57312392, 3781784, 25959000, 14109784, 10818378,
                                 25957127, 9511414, 6422823, 20289869, 15101979, 48317971, 14343887, 762092, 4840292,
                                 25130414, 7602386, 37131566, 6118940, 56598843, 11184711, 1315248, 12101316, 22809006,
                                 29979321, 149354, 32817039, 25271852, 20948989, 36749242, 26143506, 19542049, 360030,
                                 20632884, 24963451, 30874683, 11647367, 383162, 13200719, 1981660, 53123104, 10715937,
                                 24019253, 25050663, 27511028, 1514191]
NBA = [22093, 16795291, 65166616, 65785063, 835946, 890793, 3921, 450389, 20455, 987153, 240940, 246185, 9000355,
       5608488, 3280233, 3505049, 5958023, 72852, 8806795, 1811390, 2423824, 516570, 15392541, 72893, 412214, 278018,
       12106552, 42846434, 12754503, 9807715, 4108839, 33328593, 64063961, 7215125, 1811320, 1111137, 5035602, 60483582,
       9397801, 255645, 16899, 43376, 72855, 65785040, 72866, 6215230, 4987149, 72878, 16160954, 243389, 64639133,
       38958735, 72858, 27196905, 38153033, 1385825, 9733533, 49926096, 4875689, 4750398, 28754077, 43569250, 22092,
       72889, 59798759, 49778089, 346029, 8588996, 1956255, 52454088, 25390847, 31667631, 878666, 48695845, 72857,
       459304, 27837030, 17107550, 72861, 54859596, 9195892, 6560301, 72875, 72883, 240989, 3196517, 24612090]
dim_sum = [100640, 269558, 1959777, 22670461, 67072363, 11971218, 34617580, 47775306, 47840375, 28827117, 11827767,
           17031486, 11689293, 9526854, 11980851, 11971397, 47806602, 11266183, 4992439, 47775348, 46357913, 52906,
           3768042, 3746367, 47827570, 41905327, 1844187, 4023588, 2474625, 4749106, 63602774, 4146044, 3828139,
           46186668, 47827574, 37967188, 10265984, 5321303, 46665704, 4344526, 11894026, 1907296, 60769053, 519667,
           18408298, 13958538, 13719853, 41546279, 67493391, 3577886, 2054954, 48241318, 4093674, 898916, 2012983,
           13902799, 2626421, 54284514, 10887219, 40759810, 20505468, 43607423, 6168739, 2134361, 47434601, 47769544,
           19433498, 47837049, 52554299, 678353]
training_dic = {}
training_dic['python'] = python
training_dic['data science'] = data_science
training_dic['migraine'] = migraine
training_dic['chocolate'] = chocolate
training_dic['how to make pasta'] = how_to_make_pasta
training_dic[
    'Does pasta have preservatives'] = Does_pasta_have_preservatives  # 46.703346729278564 / MAP 0.527439858396989
training_dic['how google works'] = how_google_works
training_dic['what is information retrieval'] = what_is_information_retrieval
training_dic['NBA'] = NBA
training_dic['dim sum'] = dim_sum
