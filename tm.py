""" CSF469 Information Retrieval Assignment 3 -
    IBM Alignment Models and phrase-based translation

    Dataset used : provided aligned corpora in folder -> Dataset
                   created corpus - mycorpus.json

    Task Descriptions:
    Task 1) IBM Model 1 implementation.
    Task 2) IBM Model 1 and 2 Analysis (using nltk).
    Task 3) Phrase based extraction and scoring (using nltk)
"""

import json
import numpy as np
import pickle
from copy import deepcopy
from itertools import product
from nltk.translate import AlignedSent
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.phrase_based import  phrase_extraction

class task1_model:
    """implementation of IBM model 1 and EM algorithm
    """
    def __init__(self,data):
        self.data = data
        self.t_prob = None
        self.a_prob = None
        self.em_algorithm(data)

    def get_words(self,data):
        """Get all words from the corpus in each language"""
        f_words = []
        e_words = []
        for d in data:
            f_sent = d["fr"] ## foreign sentence
            e_sent = d["en"] ## English sentence
            f_words.extend(f_sent.split())
            d["fr"] = f_sent.split()
            e_words.extend(e_sent.split())
            d["en"] = e_sent.split()
        return list(set(f_words)),list(set(e_words))

    def inital_probabilities(self,fc,ec):
        """create the inital matrix of probabilities for translation
           I/P : fc - # of Foreign words (columns of the output matrix )
                 ec - # of English words (rows of the output matrix)
           O/P : 2-D matrix with initial probabilities as 1/(# of foreign words)
        """
        matrix = np.empty(ec*fc)
        matrix = matrix.reshape(ec,fc)
        matrix.fill(1/fc)
        return matrix    # repreents probability - p(e|f)

    def get_corpus_alignments(self,bitext,f_vocab,e_vocab):
        """returns all alignments of from the corpus for each sentence pair
         All alignments defined in terms of their words' index in the vocabulary lists
         I/P : The sentence aligned corpora
         O/P : The list of all corpus alignments
               Dictionary of sentence pair and alignment if mappings

        """
        alignments = []  # all alignments in the corpus
        sentence_alignments = {}  ## associated alignments for each snetence pair
        sent_count = 0
        for pair in bitext:
            sentence_alignments[sent_count] = []
            f_sent = pair["fr"]
            e_sent = pair["en"]
            e_count = len(e_sent) # number of wrods in each sentence
            f_count = len(f_sent)
            ## generate all combinations of alignments
            tuple_sets = []
            # all possible e->f mappings for each english word in separate list
            for i in range(e_count):  # getting english words count of sets of ali tuples
                list = []
                iv_idx = e_vocab.index(e_sent[i])  ## getting corresponding index of word in the the vocabulary list
                for j in range(f_count):
                    jv_idx = f_vocab.index(f_sent[j])
                    list.append((iv_idx,jv_idx))       #of form (e,f)
                tuple_sets.append(list)
            for combination in product(*tuple_sets):    ## change thos for more than 3 words
                alignments.append(combination)
                sentence_alignments[sent_count].append(len(alignments)-1)
            sent_count += 1
        #print(alignments)
        return alignments,sentence_alignments

    def get_alignment_prob(self,alignments,sentence_alignments,t_prob):
        """Calculating the alignment probabilities using the translation probability
            I/P : all the alignments and translation probabilities of the words
            o/P : Dictionary of alignment probability of each alignment,
                  key of the Dictionary is the index in the alignment list
        """
        a_prob = {}
        #Calculate alignment probability
        for i in range(len(alignments)):
            a_list = alignments[i]
            total_prob_sum = 0
            for tuple in a_list:
                r = tuple[0]    # english words across rows
                c = tuple[1]    # foreign words across columns
                total_prob_sum += t_prob[r][c]
            a_prob[i] = total_prob_sum
        ## Nomalise alignment probability
        for s_id in sentence_alignments.keys():
            alignment_sum = 0
            align_ids = sentence_alignments[s_id]
            for a_id in align_ids:
                alignment_sum += a_prob[a_id]  # Calculate nomalzatio nfactor
            for a_id in align_ids:
                a_prob[a_id] /= alignment_sum # normalize

        return a_prob

    def update_fractional_counts(self,alignments,a_prob,t_prob,sentence_alignments):
        """Implements Maximization step of EM algorithm
           Uses the calculated aligment probabilities to get
           new translation probabilities.
           I/P : alignments - all alignment in the corpus
                 a_prob - alignment probabilities
                 t_prob - translation probabilities
                 sentence_alignments - sentence aligment mappings
        """
        r = t_prob.shape[0]
        c = t_prob.shape[1]
        for i in range(r):
            for j in range(c):
                a_sum = 0
                n_sum = 0
                # Check presence of tuple in each aligment
                for a in range(len(alignments)):
                    alignment = alignments[a]
                    #Step 1 -fractional count of (e,f) counts in all alignments
                    if (i,j) in alignment:
                        a_sum += a_prob[a]
                    #Step 2 -count of  f in (e,f) in all alignments and normalize the fractional counts
                    if j in [x[1] for x in alignment]:
                        n_sum += a_prob[a]
                t_prob[i][j] = a_sum/n_sum

    def is_converged(self,a,b):
        """Test for convergence of EM algorithm, returns true if a and b  are equal"""
        return np.array_equal(a,b)

    def get_final_alignments(self,a_prob,sentence_alignments):
        """Choose the most probable alignment for every sentence pair
          returns : dictionary of chosen alignment index in alignments list for each sentence pair
        """
        final ={}
        for key in sentence_alignments.keys():
            alignments = sentence_alignments[key]
            max = None
            max_align = None
            for a in alignments:
                if max_align == None:
                    max = a_prob[a]
                    max_align = a
                elif max < a_prob[a]:
                    max = a_prob[a]
                    max_align = a
            final[key] = max_align
        return final

    def formatted_alignments(self,chosen_a_idxs,bitext,alignments,e_words,f_words):
        """formats chosen alignments to print
        """
        output =[]
        output_idxs = []
        for key in chosen_a_idxs.keys():
            temp = []
            temp_idx = []
            idx = chosen_a_idxs[key]
            alignment = alignments[idx]
            for t in alignment:
                 temp.append((e_words[t[0]],f_words[t[1]]))
                 temp_idx.append((bitext[key]["en"].index(e_words[t[0]]),bitext[key]["fr"].index(f_words[t[1]])))
            output.append(temp)
            output_idxs.append(temp_idx)
        return output,output_idxs

    def em_algorithm(self,bitext,max_iter = 5):
        """returns alignments by iterations of EM algorithm
        """
        max_iterations = max_iter
        f_words,e_words = self.get_words(bitext)   # get vocabulary in each language from the corpus
        fw_count = len(f_words)
        ew_count = len(e_words)
        t_prob = self.inital_probabilities(fw_count,ew_count) # inital translation probabilities
        iteration_count = 0
        converged = False
        alignments,sentence_alignments = self.get_corpus_alignments(bitext,f_words,e_words)
        while not converged and iteration_count < max_iterations:
            #find alignment probabilities
            t_prob_prev = deepcopy(t_prob)   ## copying the previous iteeration probability
            ## Expectation Step
            a_prob = self.get_alignment_prob(alignments,sentence_alignments,t_prob)
            ## Maximization Step -Finds new translation Probabilities
            self.update_fractional_counts(alignments,a_prob,t_prob,sentence_alignments)
            iteration_count += 1
            converged = self.is_converged(t_prob,t_prob_prev)
            self.alignment_prob = a_prob
        print("Algorithm converged after ",iteration_count," iterations")
        self.translation_table  = t_prob
        chosen_a_idxs = self.get_final_alignments(a_prob,sentence_alignments)
        self.alignment_words,self.alignment_idx = self.formatted_alignments(chosen_a_idxs,bitext,alignments,e_words,f_words)

def get_data_json(filename):
    """Method to retrive data from json files"""
    with open(filename) as f:
        data = json.load(f)
    return data

def aligned_set(bitext):
    """Create the sentence pair of object type AlignedSent"""
    aligned = []
    for d in bitext:
        f_sent = d["fr"] ## foreign sentence
        e_sent = d["en"] ## English sentence
        fr = f_sent.split()
        en = e_sent.split()
        aligned.append(AlignedSent(fr,en))
    return aligned

def print_output(data,alignments,file):
    """prints output of task 1"""
    print("######################################################################")
    print("Task 1 : IBM model 1 and EM algorithm implementation ,with corpus @",file)
    print("######################################################################")

    for i in range(len(data)):
        print("English Sentence : ",data[i]["en"])
        print("Foreign Sentence : ",data[i]["fr"])
        print("Alignment :  ",alignments[i])
        print("----------------------------------------------------------------------")

def print_output_task2(model1,model2):
    """prints output of task 2
        Comparision with
    """
    print("######################################################################")
    print("Task 2 : IBM model 1 and 2 Analysis(using NLTK)")
    print("######################################################################")
    for (a,b) in zip(model1,model2):
        print("English Sentence : ",a.mots)
        print("Foreign Sentence : ",a.words)
        print("Alignment(Model 1):  ",a.alignment)
        print("Alignment(Model 2):  ",b.alignment)
        print("----------------------------------------------------------------------")

def extract_phrases(data,model):
    """Extract phrases frm the given dataset and alignment
    """
    phrases = []
    alignment = model.alignment_idx
    for i in range(len(data)):
        sent_phrases = phrase_extraction(data[i]["fr"],data[i]["en"],alignment[i])
        phrases.append(sent_phrases)
    return phrases

def task1(dataset,printoutput=True,writepickle = False,pfile=None,usepickle=False):
    """Executes steps of task one and return model object"""
    bitext = get_data_json(dataset)
    if usepickle == True:
        with open(pfile, 'rb') as f:
            model=pickle.load(f)
    else:
        bitext_list = deepcopy(bitext)
        model = task1_model(bitext_list)
        if writepickle == True:
            with open(pfile, 'wb') as f:
                pickle.dump(model, f)

    if printoutput == True:
        print_output(bitext,model.alignment_words,dataset)
    return model,bitext

def count_ef_pair(e,f,e_f_pair_count):
    """returns count of phrase pair"""
    if e in e_f_pair_count:
        if f in e_f_pair_count[e]:
            return e_f_pair_count[e][f]
        else:
            return 0
    else:
        return 0

def count_fe_pair(e,f,f_e_pair_count):
    """returns count of phrase pair"""
    if f in f_e_pair_count:
        if e in f_e_pair_count[f]:
            return f_e_pair_count[f][e]
        else:
            return 0
    else:
        return 0

def display_phrasewise_list(prob_dict):
    """Display the dictionary values"""
    print("***********Phrase pairs and their ranks*****************")
    for f_phrase in prob_dict:
        e_phrases = prob_dict[f_phrase]
        s = [(phrase, e_phrases[phrase]) for phrase in sorted(e_phrases, key=e_phrases.get, reverse=True)]
        print(f_phrase ,"->",s)
        print("----------------------------------------------------------------------")

def phrase_scoring_ranking(phrases,model,dataset,bitext):
    """Calcultes probability values of phrases and ranks them"""
    e_phrases = []
    f_phrases = []
    count = 0
    f_phrase_count = {}
    e_phrase_count = {}  #not needed
    #e_f_pair_count = {} #e words as rows and f words as columns
    f_e_pair_count = {} #e words as rows and f words as columns
    for phrase_set in phrases:
        for phrase in phrase_set:
            e_phrases.append(phrase[3])
            f_phrases.append(phrase[2])
            if phrase[2] in f_phrase_count:
                f_phrase_count[phrase[2]] += 1
            else:
                f_phrase_count[phrase[2]] = 1
            if phrase[2] in f_e_pair_count:
                if phrase[3] in f_e_pair_count[phrase[2]]:
                    f_e_pair_count[phrase[2]][phrase[3]] += 1
                else:
                    f_e_pair_count[phrase[2]][phrase[3]] = 1
            else:
                f_e_pair_count[phrase[2]]={}
                f_e_pair_count[phrase[2]][phrase[3]] = 1

    e_phrases = list(set(e_phrases))
    f_phrases = list(set(f_phrases))
    ep_count = len(e_phrases)
    fp_count = len(f_phrases)
    #pmatrix = np.empty(ep_count*fp_count) # ######Not needed if dictionary is used
    #pmatrix = pmatrix.reshape(ep_count,fp_count)
    #pmatrix.fill(0)
    ef_prob_dict = {}
    for e in e_phrases:
        for f in f_phrases:
            ef_count =count_fe_pair(e,f,f_e_pair_count)# f_e_pair_count[e][f]
            f_count = f_phrase_count[f]
            e_idx = e_phrases.index(e)                 ###Check the count logic again
            f_idx = f_phrases.index(f)
            pair_prob = ef_count/f_count
            #pmatrix[e_idx][f_idx] = pair_prob
            if f in f_e_pair_count:
                if e in f_e_pair_count[f]:
                    if f in ef_prob_dict:
                        ef_prob_dict[f][e]=pair_prob
                    else:
                        ef_prob_dict[f] = {}
                        ef_prob_dict[f][e] = pair_prob

            #if pmatrix[e_idx][f_idx] != 0:
            #    print(e,f,ef_count,f_count,pair_prob)
    return ef_prob_dict

def task3(dataset,writepickle=False,pfilename=None,usepickle=True):
    """# 3 steps of phrase extraction -
        get word alignments,
        extract phrases
        ranking and scoring
    """
    model,bitext = task1(dataset,printoutput = False,writepickle=writepickle,pfile = pfilename,usepickle=usepickle)
    phrases = extract_phrases(bitext,model)
    scored_phrases = phrase_scoring_ranking(phrases,model,dataset,bitext)
    print_output_task3(scored_phrases,dataset)


def print_output_task3(ef_prob_dict,dataset):
    """Output of task 3"""
    print("######################################################################")
    print("Task 3 : Phrase Based translation(using NLTK) on dataset",dataset)
    print("######################################################################")
    display_phrasewise_list(ef_prob_dict)

###############################################################################
## Filenames
dataset1 = 'Dataset/data1.json'
dataset2 = 'Dataset/data2.json'
my_dataset = 'Dataset/mycorpus.json'

#Task 1) IBM Model 1 and EM algorithm implementation.
##Running the model on my dataset
d1,bitext1 = task1(dataset1)
## running the model on new created dataset
d2,bitext2 = task1(my_dataset)

#Task 2) IBM Model 1 and 2 Analysis (using nltk).
aligned_corpus_m1 = aligned_set(bitext1)
aligned_corpus_m2 = aligned_set(bitext1)
ibm1 = IBMModel1(aligned_corpus_m1,5)
ibm2 = IBMModel2(aligned_corpus_m2,5)
#translation_ptable1 = ibm1.translation_table
#translation_ptable2 = ibm2.translation_table
print_output_task2(aligned_corpus_m1,aligned_corpus_m2)

#Task 3) Phrase based extraction and scoring (using nltk)
## comment this line and uncomment next if preprocessed file is present
task3(dataset2,writepickle=True,pfilename = "t3m1",usepickle= False)
#task3(dataset2,writepickle=True,pfilename = "t3m1",usepickle= True)
task3(my_dataset,usepickle=False)
