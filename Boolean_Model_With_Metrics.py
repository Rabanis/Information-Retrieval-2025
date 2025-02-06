import pandas as pd
import nltk, os, time
import ast
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem.snowball import SnowballStemmer
import Vocabulary as voc

try:
    find('corpora/stopwords.zip')
except LookupError:
    print(LookupError, " - Missing stopword list. Downloading...")
    nltk.download("stopwords")
    print("Download Successful!")

collection_dir = os.getcwd() + os.sep + "collection" + os.sep

def main():
    start = time.perf_counter()
    voc.createVocab()
    createInvIndex()

    # Paths to necessary files
    relevant_file_path = os.path.join(collection_dir, "Relevant.txt")
    inverted_index_path = os.path.join(collection_dir, "InvertedIndex.txt")

    # Load queries, relevant docs, and inverted index
    queries = voc.getQueries(stopW=True)  # Load preprocessed queries
    print(queries)
    relevant_docs = load_relevant_docs(relevant_file_path)
    inverted_index = load_inverted_index(inverted_index_path)

    # Modify queries with boolean operators
    operator = "AND"  
    boolean_queries = add_boolean_operators(queries, operator)

    # Run the boolean model and calculate metrics
    run_boolean_model(boolean_queries, relevant_docs, inverted_index)

    print(time.perf_counter() - start)



def createInvIndex():
    try:
        print("Creating the Boolean Model Inverted Index File. Please wait...")
        voc = pd.read_csv(collection_dir + "Vocabulary.csv", header=None, names=["Term"])
        terms = set(voc["Term"])
        inverted_index_file = {}

        for term in terms:
            inverted_index_file[term] = []

        docs_path = os.path.join(collection_dir, "docs")
        for doc_filename in os.listdir(docs_path):
            doc_id = int(doc_filename)
            doc_path = os.path.join(docs_path, doc_filename)

            with open(doc_path, "r") as doc:
                line_num = 0
                for word in doc:
                    word = word.strip()
                    if word in terms:
                        if doc_id not in inverted_index_file[word]:
                            inverted_index_file[word].append(doc_id)
                    line_num += 1

        with open(collection_dir + "InvertedIndex.txt", "w") as file:
            for term, postings in inverted_index_file.items():
                file.write(f"{term} : {postings}\n")

        print("Inverted Index created successfully!")
    except Exception as error:
        print(f"Error: {type(error).__name__} – {error}")

def postfix(infix_tokens):
    precedence = {
        "NOT": 3,
        "AND": 2,
        "OR": 1,
        "(": 0,
        ")": 0
    }
    output = []
    operator_stack = []

    for token in infix_tokens:
        if token == "(":
            operator_stack.append(token)
        elif token == ")":
            operator = operator_stack.pop()
            while operator != "(":
                output.append(operator)
                operator = operator_stack.pop()
        elif token in precedence:
            while operator_stack and precedence[operator_stack[-1]] > precedence[token]:
                output.append(operator_stack.pop())
            operator_stack.append(token)
        else:
            output.append(token.lower())

    while operator_stack:
        output.append(operator_stack.pop())
    return output

def AND_op(word1, word2):
    return set(word1).intersection(word2) if word1 and word2 else set()

def OR_op(word1, word2):
    return set(word1).union(word2)

def NOT_op(a, total_docs):
    return set(total_docs).difference(a)

def load_inverted_index(file_path):
    inverted_index = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                term, postings = line.strip().split(' : ')
                postings = set(ast.literal_eval(postings))
                inverted_index[term] = postings
        print("Inverted index loaded successfully!")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        createInvIndex()
        return {}
    except Exception as error:
        print(f"Error while loading inverted index: {type(error).__name__} – {error}")
        return {}
    return inverted_index

def process_query(q, dictionary_inverted):
    q = q.replace('(', '( ').replace(')', ' )')
    q = q.split()
    query = [term for term in q]

    for i in range(len(query)):
        if query[i] in ['and', 'or', 'not']: # Ftiaxnei to AND OR NOT nane kefalaia
            query[i] = query[i].upper()

    postfix_queue = postfix(query)
    results_stack = []

    for token in postfix_queue:
        if token not in ['AND', 'OR', 'NOT']:
            token = token.replace('(', '').replace(')', '').lower()
            result = dictionary_inverted.get(token, set())
            results_stack.append(result)
        elif token == 'AND':
            a = results_stack.pop()
            b = results_stack.pop()
            results_stack.append(a & b)
        elif token == 'OR':
            a = results_stack.pop()
            b = results_stack.pop()
            results_stack.append(a | b)
        elif token == 'NOT':
            a = results_stack.pop()
            universe = set(dictionary_inverted.keys())
            results_stack.append(universe - a)

    return results_stack.pop()

# Load relevant documents from Relevant.txt
def load_relevant_docs(file_path):
    relevant_docs = []
    with open(file_path, "r") as file:
        for line in file:
            relevant_docs.append(set(map(int, line.strip().split())))
    return relevant_docs


# Add boolean operators (AND/OR) between query terms
def add_boolean_operators(queries, operator="AND"):
    boolean_queries = []
    for query in queries:
        words = query.split()
        boolean_query = f" {operator} ".join(words)
        boolean_queries.append(boolean_query)
    return boolean_queries


# Run the boolean model and evaluate metrics
def run_boolean_model(queries, relevant_docs, inverted_index):
    total_docs = set(range(len(inverted_index))) 
    results = []

    for i, query in enumerate(queries):
        start_time = time.perf_counter()
        retrieved_docs = process_query(query, inverted_index)
        exec_time = time.perf_counter() - start_time

        metrics = calculate_metrics(retrieved_docs, relevant_docs[i])

        # Print the query results
        print(f"Query{i + 1}: Files Found: {sorted(retrieved_docs)}")
        print(f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, Execution Time: {exec_time:.4f} seconds\n")

        # Append results for overall analysis if needed
        results.append({
            "query": query,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs[i],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "execution_time": exec_time
        })

def calculate_metrics(retrieved_docs, relevant_docs):
    if not retrieved_docs:
        precision = 0.0
    else:
        precision = len(retrieved_docs & relevant_docs) / len(retrieved_docs)

    recall = len(retrieved_docs & relevant_docs) / len(relevant_docs) if relevant_docs else 0.0

    return {"precision": precision, "recall": recall}

if __name__ == "__main__":
    main()
    
