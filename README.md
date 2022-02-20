# WIKIPEDIA INFORMATION RETRIEVAL (IR) ENGINE
The code is organized so there is a frontend file that has the flask and the run command. 
Also in this file are all the search functions that were given to us, but in every one of them we call a different search function from the process instance.
The second file is the backend that contains the process class. In this file we also have the Inverted Index class.
While the system goes up, we call for each index from the bucket and load it. 
Additionally we create the process instance that will hold them and will search with his search functions.

We have some search functions.
  1. Anchor search -  binary search based on the anchor text of each document in Wikipedia.
  2. Title search  - binary search based on the title only of each document in Wikipedia.
  3. Body search   - using an index built by the text body of each page in Wikipedia to retrieve a query.
  4. Search   - combined search by using 3 indices based on the title, body and anchor text with different weight for each index.
  5. Page view - search by document id that retrieves the number of views that a specific page had in August 2021. 
  6. Page rank - search by document id and retrieve back the rank this page gets from the page Rank algorithm.
