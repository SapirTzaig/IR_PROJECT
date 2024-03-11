import heapq
import math
import re
from collections import Counter
import pickle

import inverted_index_gcp

from flask import Flask, request, jsonify
from google.cloud import storage
from nltk import PorterStemmer
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

def load_pickle(file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    return pickle.loads(contents)


bucket_name = "18071996"

#upload the files from the bucket
################another path#########################################
inverted_text = load_pickle("files/inverted_text")
dict_Id_Size = load_pickle("files/dict_Id_Size")
dict_text_w2df = load_pickle("files/dict_text_w2df")
id_title_dict = load_pickle("files/id_title_dict")
not_stemmed_title_id_dict = load_pickle("files/not_stemmed_title_id_dict")
dict_pr = load_pickle("files/pagerank_dict")
text_posting_locs = load_pickle("files/text_posting_locs")
word_titles_doc_dict = load_pickle("files/word_titles_doc_dict")
inverted_text.dict_Id_Size = dict_Id_Size
inverted_text.df = dict_text_w2df
inverted_text.posting_locs = text_posting_locs

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    #this dict includes all documnets and their tfidf both text and title calculation including page rank
    dict_doc_word_tfidf = Counter()
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    #split the query into token
    query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    #using stemmer to stem the query words
    stemmer = PorterStemmer()
    #we will insert to this list the stemmed query tokens
    query_stemmed_tokens = []
    for w in query_tokens:
        if w not in all_stopwords:
            # if we didn't see this word
            word = stemmer.stem(w)
            query_stemmed_tokens.append(word)
    #convert the query list into set so we wont calculate more than once for each word its tfidf
    set_of_stemmed_query = set(query_stemmed_tokens)
    #lets run on all the words in the query
    for w in set_of_stemmed_query:
        #getting the word posting list using the inverted index method
        posting_w = inverted_text.read_a_posting_list("text_inverted_gcp", w, bucket_name)
        for id, tf in posting_w:
            #calculating doc's tfidf
            word_idf = math.log10(len(inverted_text.dict_Id_Size) / dict_text_w2df.get(w))
            tfidf = (tf / inverted_text.dict_Id_Size[id]) * word_idf
            #adding the doc's page rank to the dictionary
            dict_doc_word_tfidf[id] += math.log10(dict_pr[id])
            #adding the doc's tfidf to the dictionary
            dict_doc_word_tfidf[id] += tfidf*query_stemmed_tokens.count(w)/len(query_stemmed_tokens)
        #now lets calculate tfidf according to the stemmed word in the title
        for doc in word_titles_doc_dict[w]:
            #for each title we will create a list of its stemmed words
            title_token = [token.group() for token in RE_WORD.finditer((id_title_dict[doc]).lower())]
            title_stemmed_tokens = []
            for ww in title_token:
                if ww not in all_stopwords:
                    # if we didn't see this word
                    word1 = stemmer.stem(ww)
                    title_stemmed_tokens.append(word1)
            #calculate tfidf of the doc according to it's stemmed title
            title_idf = math.log10(len(id_title_dict) / len(word_titles_doc_dict[w]))
            title_tfidf = (title_stemmed_tokens.count(w) / len(id_title_dict[doc])) * title_idf
            dict_doc_word_tfidf[doc] += max(title_tfidf*query_stemmed_tokens.count(w)/len(query_stemmed_tokens), 0.15)
    #now we will focus on the unstemmed words
    set_of_not_stemmed_query = set(query_tokens)
    #for each word in the query we will calculte it's tfidf according to the unstemmed words in the corpus
    for word in set_of_not_stemmed_query:
        #check that the word apears in the dictionary of un stemmed words in the title
        if(word in not_stemmed_title_id_dict):
            for document in not_stemmed_title_id_dict[word]:
                #calculate tfidf for each doc that a word from the query appears in its list
                title_not_stemmed_token = [token.group() for token in RE_WORD.finditer((id_title_dict[document]).lower())]
                title_idf_not_stemmed = math.log10(len(id_title_dict) / len(not_stemmed_title_id_dict[word]))
                title_tfidf_not_stemmed = (title_not_stemmed_token.count(word) / len(id_title_dict[document])) * title_idf_not_stemmed
                dict_doc_word_tfidf[document] += max(title_tfidf_not_stemmed*query_tokens.count(word)/len(query_tokens), 0.15)
    # a list that will help us sort the tfidf dict
    top_documents_heap = []

    # Iterate over the dictionary
    for doc_id, tfidf in dict_doc_word_tfidf.items():
        # Push document to heap
        heapq.heappush(top_documents_heap, (tfidf, doc_id))

    # Retrieve titles for top documents from id_title_dict
    top_100_docs = heapq.nlargest(100, top_documents_heap)
    res = [(str(doc_id), id_title_dict[doc_id]) for _, doc_id in top_100_docs]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
