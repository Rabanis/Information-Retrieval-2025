import pandas as pd
import nltk, os, time
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem.snowball import SnowballStemmer

try:
    # Check if stopwords are already downloaded
    find('corpora/stopwords.zip')
except LookupError:
    # If not found, download the stopwords corpus
    print(LookupError, " - Missing stopword list. Downloading...")
    nltk.download("stopwords")
    print("Download Successful!")

collection_dir = os.getcwd() + os.sep + "collection" + os.sep

def main():
    start = time.perf_counter()
    createVocab()
    print(time.perf_counter() - start)

# return a pandas Series with all the queries. stopW: A boolean that shows 
# if the resulting queries will be free of stop words.
def getQueries(stopW = True):
    try:
        # Try to open and read the Queries.txt file that contains all the queries for the project.
        # Store each line as a seperate string in the global list for later use. If an error 
        # occures, print message.
        with open(collection_dir + "Queries.txt", "r") as queries_file:
            queries = []
            while True:
                content = queries_file.readline()
                content = content[:-1] #ignore \n character
                if not content:
                    break
                content = " ".join(content.split()) # remove extra spaces
                queries.append(content)
        
            if stopW:
                queries = removeStopWords(queries)
                
            print("Successful query retrieval!")
            return pd.Series(queries)
        
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return

# return a pandas Series with all the relevant docs to the query
def getRelevant():
    try:
        # Try and open the file containing the relevant docs of each query. Then read each line 
        # until no more content exists and format the strings into lists of integers for later
        # use. Lastly append the global list of relevant docs' ids. If exception, print message.
        with open(collection_dir + "Relevant.txt", "r") as relevant_file:
            relevant = []
            while True:
                content = relevant_file.readline() 
                content = content[:-1] #ignore the \n character
                if not content:
                    break
                
                content = list(map(int, content.split()))
                relevant.append(content) 

        print("Relevant lists retrieved!")
        return pd.Series(relevant)
    
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return

# function that gets a list of strings and removes stop words in each string
def removeStopWords(toRem: list):
    # We get a list of strings as an argument and want to remove all unnessecary 
    # words from the strings. We use the nltk library for this and the stopwords 
    # it provides. We load the stopwords and iterate through each item in the list.
    # Then the string is splitted into words and filtered through the stop word list.
    # Lastly, the toGet list gets appended and once complete, returned to the caller.
    try:
        toGet = []
        stop_words = set(stopwords.words('english'))

        for i in toRem:
            splitted = i.split()
            filtered_sentence = [w.lower() for w in splitted if not w.lower() in stop_words]
            toGet.append(" ".join(filtered_sentence))

        return toGet

    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return []

#### PROBABLY WON'T USE ####
# function that get a string and returns the words of its contents stemmed
def stemWords(text: str):
    try:
        stemmer = SnowballStemmer("english")
        toReturn = []
        if type(text) == list:
            text = " ".join(text)
            print(text)
        toStem = text.split()
        for word in toStem:
            toReturn.append(stemmer.stem(word))
            
        return " ".join(toReturn)
    except Exception as error:
        print("error! Please pass a list of strings!", type(error).__name__, "–", error)
        return []

# return a pandas Series with the ids of the documents and as data their (filtered) text
def parseDocs():
    # Open the folder of the documents and read their contents. Then store each document in a 
    # dictionary as a key: value pair where the key is the id of the document and the value
    # is its contets. 
    try:
        # create a dictionary to store text and ids to return
        docSer = {}
        curr_path = collection_dir + "docs" + os.sep

        # parse all files in the docs folder
        for fle in os.listdir(curr_path):
            with open(curr_path + fle, "r") as doc:
                temp = []
                # parse the doc and store lines in temp
                while True:
                    document = doc.readline()
                    document = document[:-1] #ignore \n character
                    if not document:
                        break
                    temp.append(document)
                # remove stopwords from text
                temp = removeStopWords(temp)
                temp = " ".join(temp)               # make the list into a string
                temp = " ".join(temp.split())       # remove extra spaces

                # Insert into the dictionary the fle as int: index and value the text of the document
                docSer[int(fle)] = temp
        
        # return as a Series the dictionary where the id of the doc is the index and the data is the text as a string
        return pd.Series(data= docSer).sort_index() # return the DF sorted on the documents ids

    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return

# function to remove duplicates from string
def remDuplicates(text: str):
    words = text.split()
    return list(set(words)) # Removes duplicates whithout concern for the order of words

# function that reads a csv into a Pandas Dataframe where the first column are the ids of the documents
# and the 2nd their content. Stores all the unique words found in the texts in the Vocabulary.csv file.
def createVocab():
    try:
        # read from the csv and store 
        texts = pd.read_csv(collection_dir + "Temporal.csv", names=["ID", "Text"])

        # iterate through all the texts and remove duplicate words 
        masterString = texts["Text"].apply(lambda entry: " ".join(remDuplicates(entry)))
        masterString = pd.concat([masterString, getQueries()], ignore_index= True) # include the queries in the vocabulary process

        # now we have in the masterString all the strings without duplicates. So we take them all,
        # we join them all into a masterString and pass it through the function again to remove all 
        # reoccuring instances of words in different texts. masterString now is a list of words
        masterString = remDuplicates(" ".join(masterString.to_list()))

        # create a pandas Series from the masterString and save it into a csv. This is the Vocabulary
        vocabulary = pd.Series(masterString)#.apply(lambda x: x.lower())
        vocabulary.to_csv(collection_dir + "Vocabulary.csv", header= False)


    except FileNotFoundError:
        # if the exception rises due to the file not existing, create the csv by calling the parser 
        print("Creating the csv. Please wait...\n")
        parseDocs().to_csv(collection_dir + "Temporal.csv", header= False)
        createVocab()
        return
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return

# Call the main function to run the script
if __name__ == "__main__":
    main()
