import Vocabulary as voc
import pandas as pd
import math
import time
import ast


def main():
    start = time.perf_counter()
    getResults()
    print("time to execute: ", time.perf_counter() - start)

# function that creates the inverted indexing file
def inverseIndexFile():
    try:
        vocab_Dataframe = pd.read_csv("./collection/Vocabulary.csv", index_col= 1, names= ["Doc-TF"], header= None)
        # the resulting file will need a list for each word, so we set all elements as empty lists
        vocab_Dataframe["Doc-TF"] = [[] for _ in range(len(vocab_Dataframe))]
        docs_Dataframe = pd.read_csv("./collection/Temporal.csv", names=["ID", "Text"])
        IDF_Series = pd.Series(index= vocab_Dataframe.index, data= 0.0, name= "IDF")    # series to hold idf values. Stores floats.
        number_of_docs = docs_Dataframe.shape[0]    # number of documents in the collection


        # iterate through all documents
        for docs_index in docs_Dataframe.index:

            id = int(docs_Dataframe.at[docs_index, "ID"])
            text = docs_Dataframe.at[docs_index, "Text"].split()
            # count all word occurances in text
            doc_word_freq = pd.Series(data= text).value_counts()

# If we need the position of each word in the inverse index file, modify with doc_word_freq

            # calculate simple idf for each word
            for word in doc_word_freq.keys():
                IDF_Series[word] = IDF_Series[word] + 1

                # for readability. TF = log2(1 + tf). Convert to float and keep first 3 digits
                tf = round(math.log2(1 + doc_word_freq.loc[word]), 3)
                vocab_Dataframe.at[word, "Doc-TF"].append([id, tf])


        vocab_Dataframe["IDF"] = IDF_Series
        vocab_Dataframe.index.name = "word"
        
        #calculate tf*idf for each word and store it in the place of tf
        for word, row in vocab_Dataframe.iterrows():
            log_idf = calculate_idf(number_of_docs, row["IDF"])
            vocab_Dataframe.at[word, "IDF"] = log_idf

            # Update the term frequencies with their TF-IDF values
            for doc_info in row["Doc-TF"]:
                doc_id, tf = doc_info
                tf_idf = round(log_idf * tf, 3)
                doc_info[1] = tf_idf

        vocab_Dataframe.sort_index(axis= 0, inplace= True)      # sort on words
        vocab_Dataframe.to_csv(voc.collection_dir + "Inverse_Index.csv", header= True)

    except FileNotFoundError:
        # if the vocab_Dataframe.csv is missing, create it and call again self.
        print("Creating the Vocab_Dataframe, please wait...")
        vocab_time = time.perf_counter()
        voc.createVocab()
        print("\tTime to create vocabulary: ", time.perf_counter() - vocab_time, "\n")
        inverseIndexFile()
        return
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return
    
# function that reads the inverted index file and stores in a new csv the document vectors for later use.
def create_Vectors():
    try:
        iif_df = pd.read_csv("./collection/Inverse_Index.csv", index_col= 0, names= ["Doc-TF", "IDF"], header= 0)
        iif_df["Doc-TF"] = iif_df["Doc-TF"].apply(ast.literal_eval) # convert string object to list of lists.

        ##### DOCUMENTS #####
        documents_ids = list(pd.read_csv("./collection/Temporal.csv", index_col= 0, header= None).index)    # load the ids for indexing
        doc_vectors_df = pd.DataFrame(index= documents_ids, columns= ["vector", "norm"])      # create dataframe for document vectors
        doc_vectors_df["vector"] = [{} for _ in range(len(documents_ids))]            # set values as empty dictionaries.
        doc_vectors_df["norm"] = 0.0      # set the norm of each vector to 0

        # for each word, get each document's corresponding tf value and update it with the resulting weight.
        for word, row in iif_df.iterrows():

            # Update the term frequencies with their TF-IDF values
            for word_info in row["Doc-TF"]:
                doc_id = word_info[0]
                word_weight = word_info[1]      # tf-idf weight
                doc_vectors_df.loc[doc_id, "vector"][word] = word_weight
                doc_vectors_df.loc[doc_id, "norm"] = doc_vectors_df.loc[doc_id, "norm"] + pow(word_weight, 2)
            

        #### QUERIES #####
        query_Series = voc.getQueries()
        query_vector_df = pd.DataFrame(index= list(range(1,21)), columns= ["vector", "norm"])
        query_vector_df["vector"] = [{} for _ in range(1,21)]            # set values as empty dictionaries.
        query_vector_df["norm"] = 0.0      # set the norm of each vector to 0

        i = 0       # counter for indexing
        for query in query_Series:
            i = i + 1               
            q = set(query.split())  # list of words, set() = no duplicates
            for word in q:
                idf = float(iif_df.loc[word, "IDF"])
                tf = query.split().count(word)     
                weight = idf * tf                 # calculate the tf in the query for better representation
                query_vector_df.loc[i, "vector"][word] = weight
                query_vector_df.loc[i, "norm"] = query_vector_df.loc[i, "norm"] + pow(weight, 2)

        # store the norms of all vectors for later use and perfomance increase
        doc_vectors_df["norm"] = doc_vectors_df["norm"].apply(math.sqrt)
        query_vector_df["norm"] = query_vector_df["norm"].apply(math.sqrt)

        #### Store CSVs ####
        query_vector_df.to_csv(voc.collection_dir + "Query_Vectors.csv", header= None)
        doc_vectors_df.to_csv(voc.collection_dir + "Document_Vectors.csv", header= None)
        return
                
    except FileNotFoundError:
        # if the Inverse_Index.csv is missing, create it and call again self.
        print("Creating the Inverted Index File, please wait...")
        iif_time = time.perf_counter()
        inverseIndexFile()
        print("\tTime to create the Inverted Index File: ", time.perf_counter() - iif_time, "\n")
        create_Vectors()
        return
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return

# method that creates the master matrix of all the vectors and then computes similarities between queries and documents.
def master_Matrix():
    try:
        # load vectors
        iif_df = pd.read_csv("./collection/Inverse_Index.csv", index_col= 0, names= ["Doc-TF", "IDF"], header= 0)
        iif_df["Doc-TF"] = iif_df["Doc-TF"].apply(ast.literal_eval) # convert string object to list of lists.
        docs_vector_df = pd.read_csv("./collection/Document_Vectors.csv", index_col= 0, names= ["Vector", "norm"], header= None)
        docs_vector_df["Vector"] = docs_vector_df["Vector"].apply(ast.literal_eval) # convert string object to dictionary.
        queries_df = pd.read_csv("./collection/Query_Vectors.csv", index_col= 0, names= ["Vector", "norm"], header= None)
        queries_df["Vector"] = queries_df["Vector"].apply(ast.literal_eval) # convert string object to dictionary.
        related_Series = voc.getRelevant()  # series to store all the queries's relevant documents
        result = {}                         # dictionary to store the resulting similarities
        print("Finished loading files!\n")
        
    ######## create the master matrix which stores the query and document vectors #########
        start_time = time.perf_counter()

        master_DF = pd.DataFrame(index= iif_df.index, columns= docs_vector_df.index, data= 0.0)
        for i in range(20, 0, -1):
            master_DF.insert(loc= 0,column= "q" + str(i), value= 0.0)
        

        for query_id, vec in queries_df.iterrows():
            query_vector = vec.iloc[0]

            for weight in query_vector:
                master_DF.at[weight, "q" + str(query_id)] = float(query_vector[weight])
        
        for doc_id, vec in docs_vector_df.iterrows():
            document_vector = vec.iloc[0]

            for weight in document_vector:
                master_DF.at[weight, doc_id] = float(document_vector[weight])

        print("Master Matrix created!\nTime to create: ", time.perf_counter() - start_time, "\n")

    ######## Compute similarities! ###########

        # For each query
        for q_index, quer_row in queries_df.iterrows():
        # get each document related to the query and store their similarity.
            
            start_time = time.perf_counter()
            query_words = []
            related_Documents = []
            similarity_dict = {}    # dictionary to hold for each query the related document ids

            for word in quer_row.loc["Vector"].items():
                # word looks like this: ["word", tf-idf_weight: float]
                query_words.append(word[0])

                word_related_doc_list = iif_df.loc[word[0], "Doc-TF"]

                # retrieve all related to the query docs.
                for doc in word_related_doc_list:
                    related_Documents.append(doc[0])
                    
            related_Documents = set(related_Documents) # keep each doc once


            for doc in related_Documents:
                if not doc in master_DF.columns:
                    continue
                doc_vec = master_DF[doc]
                q_vec = master_DF["q" + str(q_index)]

                similarity_dict[doc] = round(compute_Similarity_simple(doc_vec, q_vec), 5)

                result[q_index] = similarity_dict
            
            print("time to calculate similarities: ", time.perf_counter() - start_time)
            similarity_dict = pd.Series(similarity_dict).sort_values(ascending= False)  # make the dictionary a Series to sort values.
            # set the threshold for similarity between vectors
            most_relevant = similarity_dict[similarity_dict > 0.1]
            #most_relevant = similarity_dict.head(50)

            result[q_index] = sorted(list(most_relevant.keys()))
            print("Query", q_index, "finished!\nTime to calculate: ", time.perf_counter() - start_time)
            
            print(len(result[q_index]), "relevant documents!")

            query_relevant = set(related_Series.loc[q_index - 1])
            print(calculate_metrics(set(result[q_index]) ,query_relevant), "\n")

        pd.Series(result).to_csv(voc.collection_dir + "Results_Master.csv" ,header= None)


    except FileNotFoundError:
        # if the Document_Vectors.csv is missing, create it and call again self.
        print("Creating the document and query vectors, please wait...")
        qTime = time.perf_counter()
        create_Vectors()
        print("\tTime to create the files: ", time.perf_counter() - qTime, "\n")
        master_Matrix()
        return
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return


# Version 2. Ask the queries and retrieve relevant documents as a list.
def getResults(threshold = 0.0):
    try:
        # load vectors
        iif_df = pd.read_csv("./collection/Inverse_Index.csv", index_col= 0, names= ["Doc-TF", "IDF"], header= 0)
        iif_df["Doc-TF"] = iif_df["Doc-TF"].apply(ast.literal_eval) # convert string object to list of lists.
        docs_vector_df = pd.read_csv("./collection/Document_Vectors.csv", index_col= 0, names= ["Vector", "norm"], header= None)
        docs_vector_df["Vector"] = docs_vector_df["Vector"].apply(ast.literal_eval) # convert string object to dictionary.
        queries_df = pd.read_csv("./collection/Query_Vectors.csv", index_col= 0, names= ["Vector", "norm"], header= None)
        queries_df["Vector"] = queries_df["Vector"].apply(ast.literal_eval) # convert string object to dictionary.

        # store retrieved documents
        result = {}
        metrics = pd.DataFrame(index= ["precision", "recall"] ,columns= list(range(1,21)))
        # get relevant documents
        expert_Relevant = voc.getRelevant() # pandas Series


        # For each query
        for q_index, quer_row in queries_df.iterrows():
        # get each document related to the query and store their similarity.

            start_time = time.perf_counter()
            quer_words = []
            related_Documents = []
            similarity_dict = {}    # dictionary to hold for each query the related document ids
            query_norm = quer_row["norm"]   # used for computing similarity

            for word in quer_row.loc["Vector"].items():
                # word looks like this: ["word", tf-idf_weight: float]
                quer_words.append(word[0])

                word_related_doc_list = iif_df.loc[word[0], "Doc-TF"]

                # retrieve all related to the query docs.
                for doc in word_related_doc_list:
                    related_Documents.append(doc[0])
                    
            related_Documents = set(related_Documents) # keep each doc once

            df1 = pd.DataFrame(list(quer_row.loc["Vector"].items()), columns= ["word", "weight"])

            for doc in related_Documents:       # for each document related:

                document_vector = docs_vector_df.loc[doc, "Vector"] # dictionary
                document_vector_norm = docs_vector_df.loc[doc, "norm"]

                # dataframe of document_vector dictionary. Merge with the query with an outer join.
                df2 = pd.DataFrame(list(document_vector.items()), columns= ["word", "weight"])
                
                # new df to hold the temporal vectors to be compared.
                # merge on only the common words. All others will give a dot product of 0
                compare_df = pd.merge(df2, df1, on= "word", suffixes= ["_D", "_Q"]).fillna(0)
                compare_df.set_index("word", inplace= True)

                # compute the similarity of the two vectors. 
                similarity_dict[doc] = compute_Similarity(compare_df["weight_D"], document_vector_norm, compare_df["weight_Q"], query_norm)
                
            similarity_dict = pd.Series(similarity_dict).sort_values(ascending= False)  # make the dictionary a Series to sort values.

            # set the threshold for similarity between vectors
            most_relevant = similarity_dict[similarity_dict > threshold]
            #most_relevant = similarity_dict.head(50)

            result[q_index] = list(most_relevant.keys())
            print("Query", q_index, "finished!\nTime to calculate: ", time.perf_counter() - start_time)
            
            print(len(result[q_index]), "relevant documents!")

            query_relevant = set(expert_Relevant.loc[q_index - 1])
            metrics[q_index] = calculate_metrics(set(result[q_index]), query_relevant)
            #metrics.at["no. docs retrieved", q_index] = len(result[q_index])
            print(metrics[q_index], "\n")



        metrics.T.to_csv(voc.collection_dir + "metrics.csv")
        pd.Series(result).to_csv(voc.collection_dir + "Results.csv" ,header= None)

            
    except FileNotFoundError:
        # if the Document_Vectors.csv is missing, create it and call again self.
        print("Creating the document and query vectors, please wait...")
        qTime = time.perf_counter()
        create_Vectors()
        print("\tTime to create the files: ", time.perf_counter() - qTime, "\n")
        getResults()
        return
    except Exception as error:
        errLine = error.__traceback__.tb_lineno
        print("Line", errLine, "Error: ", type(error).__name__, "–", error)
        return


def calculate_idf(number_of_docs, idf):
    # if idf = 0, the word originated from a query
    if idf == 0:
        idf = 1
    return round(math.log2(number_of_docs / idf), 3)

# Compute the cosine similarity of 2 vectors of the same dimension. Returns a float between [0, 1].
def compute_Similarity(vec_1: pd.Series, vec_1_norm, vec_2: pd.Series, vec_2_norm):
    #### Denominator ####
    denom = vec_1_norm * vec_2_norm

    #### Numerator ####
    numer = vec_2.dot(vec_1)

    return float(numer / denom)

def compute_Similarity_simple(vec_1: pd.Series, vec_2: pd.Series):
    #### Denominator ####
    vec_1.apply(lambda x: pow(x, 2))
    size_1 = math.sqrt(vec_1.sum())

    vec_2.apply(lambda x: pow(x, 2))
    size_2 = math.sqrt(vec_2.sum())

    denom = size_1 * size_2

    #### Numerator ####
    numer = vec_2.dot(vec_1)

    return float(numer / denom)

def calculate_metrics(retrieved_docs, relevant_docs):

    if not retrieved_docs:
        precision = 0.0
    else:
        precision = len(retrieved_docs & relevant_docs) / len(retrieved_docs)

    recall = len(retrieved_docs & relevant_docs) / len(relevant_docs) if relevant_docs else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3)}


# Call the main function to run the script
if __name__ == "__main__":
    main()