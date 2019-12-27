This project is to design and implement a model that determines how similar two chunks of text are. The similarity score takes an integer value between 1 and 5 (included). The higher the score, the more similar the two chunks are.

In general, semantic textual similarity (STS) is a challenging problem, as it requires both an understanding of lexical-level similarity, and the semantic composition of the two chunks of text being analyzed. As a reference, here are some STS examples:

Sentence 1: Birdie is washing itself in the water basin.
Sentence 2: The bird is bathing in the sink.
Score: 4
Comment: Both sentences convey the message that a bird is taking a bath.

Sentence 1: The young lady enjoys listening to the guitar.
Sentence 2: The young lady enjoys playing the guitar.
Score: 2
Comment: Both sentences involve a lady and a guitar, but convey different actions i.e. listening to the guitar and playing the guitar respectively.

---------------------------------------------------------------------------------------------------------
The data folder containing train, dev and test files.

The train and dev files are of the form:
<Input_Id><TAB><Sentence 1><TAB><Sentence 2><TAB><Score>

The test file is of the form:
<Input_Id><TAB><Sentence 1><TAB><Sentence 2>

----------------------------------------------------------------------------------------------------------
The project consists of four tasks: 
Task 1: Create a class CorpusReader that is able to read the data files and represent the information in a way such that our model can process it. 
Task 2: Implement a deep NLP pipeline to extract the following NLP based features from the natural language statements:
o	Tokenize the two sentences into words.
o	Lemmatize the words to extract lemmas as features
o	Part-of-speech (POS) tag the words to extract POS tag features
o	Perform dependency parsing or full-syntactic parsing to get parse-tree based patterns as features
o	Using WordNet, extract hypernymns, hyponyms, meronyms, AND holonyms as features
o	Additional features  
Task 3: Implement a machine-learning, statistical, or heuristic (or a combination) based approach to determine the semantic textual similarity (STS) between two pieces of text and produce at similarity score (integer value between 1 (lowest) and 5(highest)):
o	Run the above described deep NLP on the input corpus (train or dev set).
o	Using the train set, implement/apply a machine-learning, statistical, or heuristic (or a combination) based approach to learn a rules/model that can determine the STS between any two pieces of input text and produce at similarity score (integer value between 1 (lowest) and 5(highest)). 
o	On the dev set, evaluate your STS system using the evaluation script. The script takes two files as arguments: the gold file containing the gold labels, and the prediction file containing the predicted labels. Note that the gold file is same as the train/dev sets provided, and the prediction file is the one output by our program, which must have the same format as the attached sample file.
Task 4: The performance of our model will be evaluated on the test set. 
o	Run the above described deep NLP on the input test set.
o	Run the test set through your STS system using the given evaluation script.  

----------------------------------------------------------------------------------------------------------
How to run:

For Task 1, please refer to the CorpusReader class defined in 'task1_data_process.py';

For Task 2, please enter 'python task2_feature_extraction.py' at command line

For Task 3, please enter 'python task3_model.py' at command line

For Task 4, please enter 'python task4_predict.py' at command line. 
    Task 4 generates two output file: test_prediction_rf.txt (using just random forest regressor model) 
          and test_prediction_embedded.txt (embedding random forest regressor and xgboost models)

Evaluation:
For evaluation of the dev-set:
      please enter 'python evaluation.py data/new-dev-set.txt dev_prediction.txt' at command line
      
for evaluation of the test-set:
      please enter 'python evaluation.py data/test-set.txt test_prediction_rf.txt' at command line
                or 'python evaluation.py data/test-set.txt test_prediction_embedded.txt' at command line
      


