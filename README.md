

# WIKIPEDIA INFORMATION RETRIEVAL (IR) ENGINE

In this project, we Create an Efficiency information retrieval engine on the entire Wikipedia.

# PRE PROCESS
We collect all the data from Wikipedia and process it by using GCP cluster and SPARK RDD which allow us to operate functions over the whole data in parallel and Distributed on several machines. 
We tokenize each page and generate 3 indecis for the body, title, and the anchor text 
and save those into google buckets.
To optimize the performance in running time we calculate all the possible calculations before and save them also to the bucket (such as TF-IDF matrix, Document freq dictionary).

# RUN TIME
When running the server, we load the indecis and all the relevant data from the buckets.
We implement retrieve information by using Cosine Simalarty based on TF-IDF values of the given query and relevant docs (which includes one or more terms from the query) by using a posting - list for each word.
each posting - list was stored in a bin and in run time we access the neccery posting - list.


The code is organized so there is a frontend file that has the flask and the run command. 
Also in this file are all the search functions that were given to us, but in every one of them, we call a different search function from the process instance.
The second file is the backend that contains the process class. In this file, we also have the Inverted Index class.


We have some search functions.
  1. Anchor search -  binary search based on the anchor text of each document in Wikipedia.
  2. Title search  - binary search based on the title only of each document in Wikipedia.
  3. Body search   - using an index built by the text body of each page in Wikipedia to retrieve a query.
  4. Search   - combined search by using 3 indices based on the title, body and anchor text with different weight for each index.
  5. Page view - search by document id that retrieves the number of views that a specific page had in August 2021. 
  6. Page rank - search by document id and retrieve back the rank this page gets from the page Rank algorithm.


