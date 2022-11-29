import nltk,math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log10,sqrt
from collections import Counter
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
import os
from csv import DictReader
from collections import defaultdict
from nltk.corpus import stopwords

#corpusroot = 'test_text.csv'        #test data file
#vectors={}                          #tf-idf vectors for all documents (each line is a doc with a format such that Subject ID: Text)
#df=Counter()                        #storage for document frequency
#tfs={}                              #permanent storage for tfs of all tokens in all documents
#lengths=Counter()                   #used for calculating lengths of documents
#postings_list={}                    #posting list storage for each token in the corpus
#st_tokens=[]
#needed_info = defaultdict()

class Query():
    def __init__(self):
        self.corpusroot = 'test_text.csv'  # test data file
        self.vectors = {}  # tf-idf vectors for all documents (each line is a doc with a format such that Subject ID: Text)
        self.df = Counter()  # storage for document frequency
        self.tfs = {}  # permanent storage for tfs of all tokens in all documents
        self.lengths = Counter()  # used for calculating lengths of documents
        self.postings_list = {}  # posting list storage for each token in the corpus
        self.st_tokens = []

    def read_csv(self):
        with open(self.corpusroot, 'r', encoding = "ISO-8859-1") as read_obj:
            csv_dict_reader = DictReader(read_obj)
            needed_info = defaultdict()
            for row in csv_dict_reader:
                needed_info.update({row['SUBJECT_ID']:row['TEXT']})
                tokens = tokenizer.tokenize(needed_info[row['SUBJECT_ID']])  # tokenizing each document
                sw = stopwords.words('english')
                tokens = [stemmer.stem(token) for token in tokens if token not in sw]  # removing stopwords and performing stemming
                tf = Counter(tokens)
                self.df += Counter(list(set(tokens)))
                self.tfs[row['SUBJECT_ID']] = tf.copy()  # making a copy of tf into tfs for that filename
                tf.clear()  # clearing tf so that the next document will have an empty tf
            print("tfs", self.tfs)
            print("needed_info", needed_info)
            print("==============")

    def calWeight(self, text, token):                                      #returns the weight of a token in a document without normalizing
        self.idf = self.getidf(token)
        return (1+log10(self.tfs[text][token]))*self.idf                       #tfs has the logs of term frequencies of all docs in a multi-level dict

    def getidf(self, token):
        if self.df[token]==0:
            return -1
        return log10(len(self.tfs)/self.df[token])                #len(tfs) returns no. of docs; df[token] returns the token's document frequency

    def caltf_idf(self):
        for text in self.tfs:                                #loop for calculating tf-idf vectors and lengths of documents
            self.vectors[text]=Counter()                     #initializing the tf-idf vector for each doc
            length=0
            for token in self.tfs[text]:
                weight = self.calWeight(text, token)         #calWeight calculates the weight of a token in a doc without normalization
                self.vectors[text][token]=weight             #this is the weight of a token in a file
                length += weight**2                         #calculating length for normalizing later
            self.lengths[text]=math.sqrt(length)

    def norm_weight(self):
        for text in self.vectors:                                                    #loop for normalizing the weights
            for token in self.vectors[text]:
                self.vectors[text][token]= self.vectors[text][token] / self.lengths[text]      #dividing weights by the document's length
                if token not in self.postings_list:
                    self.postings_list[token]=Counter()
                self.postings_list[token][text] = self.vectors[text][token]                     #copying the normalized value into the posting list

    def query(self,qstring):                             #function that returns the best match for a query
        qstring=qstring.lower()                     #converting the words to lower case
        qtf={}
        qlength=0
        flag=0
        loc_docs={}
        tenth={}
        cos_sims=Counter()                          #initializing a counter for calculating cosine similarity b/w a token and a doc
        for token in qstring.split():
            token=stemmer.stem(token)               #stemming the token using PorterStemmer
            if token not in self.postings_list:          #if the token doesn't exist in vocabulary,ignore it (this includes stopwords removal)
                continue
            if self.getidf(token)==0:                    #if a token has idf = 0, all values in its postings list are zero. max 10 will be chosen randomly
                loc_docs[token], weights = zip(*self.postings_list[token].most_common())         #to avoid that, we store all docs
            else:
                loc_docs[token],weights = zip(*self.postings_list[token].most_common(2))         #taking top 2 in postings list, NEED to increase this number once have more data
            if len(weights) == 1:
                tenth[token] = weights[0]
            else:
                tenth[token]=weights[1]                                                         #storing the upper bound of each token
            if flag==1:
                commondocs=set(loc_docs[token]) & commondocs                                #commondocs keeps track of docs that have all tokens
            else:
                commondocs=set(loc_docs[token])
                flag=1
            qtf[token]=1+log10(qstring.count(token))    #updating term freq of token in query
            qlength+=qtf[token]**2                      #calculating length for normalizing the query tf later
        qlength=sqrt(qlength)
        for doc in self.vectors:
            cos_sim=0
            for token in qtf:
                if doc in loc_docs[token]:
                    cos_sim = cos_sim + (qtf[token] / qlength) * self.postings_list[token][doc]       #calculate actual score if document is in top 10
                else:
                    cos_sim = 0                                                                 #otherwise, calculate its upper bound score
            cos_sims[doc]=cos_sim
        max=cos_sims.most_common(1)                                                              #seeing which doc has the max value
        ans,wght=zip(*max)
        try:
            if ans[0] in commondocs:                                                             #if doc has actual score, return score
                return ans[0],wght[0]
            else:
                return "fetch more",0                                                            #if upperbound score is greater, return fetch more
        except UnboundLocalError:                                                                #if none of the tokens are in vocabulary, return none
            return "None",0


"""
References:
https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
https://stackoverflow.com/questions/15890503/valueerror-math-domain-error
https://thispointer.com/python-read-a-csv-file-line-by-line-with-or-without-header/
https://www.nltk.org/api/nltk.tokenize.html
"""
if __name__ == '__main__':
    query1 = Query()
    query1.read_csv()
    query1.caltf_idf()
    query1.norm_weight()

    print("(%s, %.12f)" % query1.query("Canadian Thanksgiving"))
    print("(%s, %.12f)" % query1.query("vector entropy"))
    print("(%s, %.12f)" % query1.query("Wolrd cup Japanese win"))
    print("(%s, %.12f)" % query1.query("text search engine"))