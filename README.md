# Information Retrieval Project- Wikipedia Search Engine


The app searches relevent documents from wikipedia dumps and retrives them


## Creators:
* **Liron Miriam Shemen** - (https://github.com/LironShemen)
* **Sapir Tzaig** - (https://github.com/SapirTzaig)

## Design goals:

- Time efficiency for retriving information
- Max out the mean average precision at 10
- Retrieve relevent docs from the wikipedia dumps

## Text operations and methods to accomplish our goals

- Using tfidf on both title and text
- Using page rank algorithm in order to improve retrivel preformance
- Using stemming on both words in text and title


## Engine Files:
All relevent information are written in files:

- `search_frontend.py`
- `Inverted_Index_gcp.py`

## Inverted Index:
- `jupiter_notebook_inverted_gcp.ipynb`

## Run APP:

- run command: `python3 search_frontend.py`
