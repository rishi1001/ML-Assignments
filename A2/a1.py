import sys
import json
import numpy as np
import math
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def read(location):
    file = open(location)
    lines = file.readlines()
    review_text = []
    labels = []
    for line in lines:
        data = json.loads(line)
        review_text.append(data['reviewText'])
        labels.append(int(data['overall']))
    return review_text,labels

def get_summary(location):
    file = open(location)
    lines = file.readlines()
    summary_text = []
    for line in lines:
        data = json.loads(line)
        summary_text.append(data['summary'])
    words = []
    for summary in summary_text:
        words_summary = summary.split()    
        words.append(words_summary)
    return words

def get_words(review_text):
    words = []
    for review in review_text:
        words_review = review.split()  
        words.append(words_review)
    return words

def get_items(words):
    word_map={}
    unique_words=[]
    i=0
    for word_list in words:
        for word in word_list:
            if word not in word_map:
                word_map[word]=i
                unique_words.append(word)
                i+=1
    return word_map,unique_words

def get_param(words,word_map,labels,tot_unique_words,m):                  
    phi_x = [[1 for i in range(6)] for j in range(tot_unique_words)]
    phi_y = [0 for i in range(6)]
    tot_y = [tot_unique_words for i in range(6)]
    for i in range(m):
        phi_y[labels[i]]+=1
        for word in words[i]:
            phi_x[word_map[word]][labels[i]]+=1
            tot_y[labels[i]]+=1
    for i in range(1,6): 
        phi_y[i]=phi_y[i]/m
    for i in range(len(phi_x)):
        for j in range(len(phi_x[0])):
            phi_x[i][j]=phi_x[i][j]/tot_y[j]
    return phi_x,phi_y

def get_param_summary(words,word_map,labels,tot_unique_words,m,summary_words):                  
    phi_x = [[1 for i in range(6)] for j in range(tot_unique_words)]
    phi_y = [0 for i in range(6)]
    tot_y = [tot_unique_words for i in range(6)]
    weight = 5
    for i in range(m):
        phi_y[labels[i]]+=1
        for word in words[i]:
            if word in summary_words[i]:
                phi_x[word_map[word]][labels[i]]+=weight
            else:
                phi_x[word_map[word]][labels[i]]+=1
            tot_y[labels[i]]+=1
    for i in range(1,6): 
        phi_y[i]=phi_y[i]/m
    for i in range(len(phi_x)):
        for j in range(len(phi_x[0])):
            phi_x[i][j]=phi_x[i][j]/tot_y[j]
    return phi_x,phi_y


def get_prediction(phi_x,phi_y,m,word_map,words):
    log_p = [[0 for i in range(6)] for j in range(m)]
    for i in range(m):
        for j in range(1,6):
            log_p[i][j]+=math.log(phi_y[j])
    words_not_present = 0
    for i in range(m):
        for word in words[i]:
            for j in range(1,6):
                if word not in word_map:
                    words_not_present +=1
                    continue
                log_p[i][j]+=math.log(phi_x[word_map[word]][j])
    words_not_present/=5
    print("Total words not present in vocabulary :",words_not_present)
    pred_y = []
    for i in range(m):
        pred_y.append(log_p[i].index(max(log_p[i][1:6])))
    return pred_y

def accuracy(pred_y,labels,m):
    correct = 0
    for i in range(m):
        if pred_y[i]==labels[i]:
            correct+=1
    return correct/m

def random_guess(m):
    Y_pred = np.zeros(m)
    for i in range(m):
        Y_pred[i]= random.randint(1,5)        
    return Y_pred

def solve(words,labels):
    word_map,unique_words = get_items(words)
    m = len(words)
    tot_unique_words = len(unique_words)
    phi_x,phi_y = get_param(words,word_map,labels,tot_unique_words,m)
    pred_y = get_prediction(phi_x,phi_y,m,word_map,words)
    print("Train Accuracy :",accuracy(pred_y,labels,m))
    return phi_x,phi_y,word_map,tot_unique_words

def solve_with_summary(words,labels,summary_words):
    word_map,unique_words = get_items(words)
    m = len(words)
    tot_unique_words = len(unique_words)
    phi_x,phi_y = get_param_summary(words,word_map,labels,tot_unique_words,m,summary_words)
    pred_y = get_prediction(phi_x,phi_y,m,word_map,words)
    print("Train Accuracy :",accuracy(pred_y,labels,m))
    return phi_x,phi_y,word_map,tot_unique_words

def stop_word(words,labels):
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word_list in words:
        a = []
        for word in word_list:
            if not word in stop_words:
                a.append(word)
        filtered_words.append(a)
    return solve(filtered_words,labels)

def stem_word(words,labels):
    ps = PorterStemmer()
    stem_words = []
    i=0
    for word_list in words:
        a = []
        for word in word_list:
            a.append(ps.stem(word))
            i+=1
            if i%100000==0:
                print(i)
        stem_words.append(a)
    return solve(stem_words,labels)

def lemmatize_word(words,labels):
    lm = WordNetLemmatizer()
    lemmatize_words = []
    i=0
    for word_list in words:
        a = []
        for word in word_list:
            a.append(lm.lemmatize(word))
            i+=1
            if i%100000==0:
                print(i)
        lemmatize_words.append(a)
    return solve(lemmatize_words,labels)

def stop_stem_words(words,labels):
    ps = PorterStemmer()
    stem_words = []
    i=0
    for word_list in words:
        a = []
        for word in word_list:
            a.append(ps.stem(word))
            i+=1
            if i%100000==0:
                print(i)
        stem_words.append(a)
    words = stem_words
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word_list in words:
        a = []
        for word in word_list:
            if not word in stop_words:
                a.append(word)
        filtered_words.append(a)
    
    return solve(filtered_words,labels)

def get_confusion_matrix(Y_pred,Y):                
    tot_labels = 5
    matrix = np.zeros((tot_labels,tot_labels),dtype=np.int)
    for i in range(len(Y)):
        matrix[int(Y[i]-1)][int(Y_pred[i]-1)]+=1
    return matrix

def get_f1(matrix):
    tot_labels = 5
    f1_score = np.empty(tot_labels)
    for i in range(tot_labels):
        tp = 0 
        fp = 0
        fn = 0
        for j in range(tot_labels):
            if j==i:
                tp+=matrix[i][j]
            else:
                fn+=matrix[i][j]
                fp+=matrix[j][i]
        f1 = tp/(tp+0.5*(fp+fn))
        f1_score[i]=f1
    return f1_score

def feature_words(review_text):
    words = []
    j=1
    for review in review_text:
        words_review = review.split()     
        a = []
        if len(words_review)==0:
            words.append(a)
            continue
        start_word = words_review[0]
        for i in range(1,len(words_review)):
            new_word = start_word+" "+words_review[i]
            start_word = words_review[i]
            a.append(new_word)
        words.append(a)
    return words


def part_A(review_text,labels,review_text_test,labels_test):
    words = get_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=solve(words,labels)
    words_test = get_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Test Accuracy :",accuracy(pred_y_test,labels_test,m_test))

    confusion_matrx = get_confusion_matrix(pred_y_test,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    return

def part_B(labels_test):
    m_test = len(labels_test)
    Y_pred = random_guess(m_test)
    print("Random Guess Accuray :",accuracy(Y_pred,labels_test,m_test))
    confusion_matrx = get_confusion_matrix(Y_pred,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    counts = np.bincount(labels_test)
    a = np.argmax(counts)
    print("Maximum Label is : ",a)
    Y_pred = np.zeros(m_test)
    for i in range(m_test):
        Y_pred[i]=a
    print("Maximum Predict Accuray :",accuracy(Y_pred,labels_test,m_test))
    confusion_matrx = get_confusion_matrix(Y_pred,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    return

def part_C(review_text,labels,review_text_test,labels_test):
    words = get_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=solve(words,labels)
    words_test = get_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Confusion Matrix :")
    print(get_confusion_matrix(pred_y_test,labels_test))    
    return

def part_D(review_text,labels,review_text_test,labels_test):
    words = get_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=stop_stem_words(words,labels)
    words_test = get_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Test Accuracy :",accuracy(pred_y_test,labels_test,m_test))

    confusion_matrx = get_confusion_matrix(pred_y_test,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    return

def part_E(review_text,labels,review_text_test,labels_test):
    # feature 1 -> bi-gram
    print("Feature 1 : Bi-Grams")
    words = feature_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=solve(words,labels)
    words_test = feature_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Test Accuracy :",accuracy(pred_y_test,labels_test,m_test))

    confusion_matrx = get_confusion_matrix(pred_y_test,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)

    # feature 2 -> lemmatize
    print("Feature 2 : Lemmatize the Words")
    words = get_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=lemmatize_word(words,labels)
    words_test = get_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Test Accuracy :",accuracy(pred_y_test,labels_test,m_test))

    confusion_matrx = get_confusion_matrix(pred_y_test,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    return

def part_G(review_text,labels,review_text_test,labels_test):
    summary_words = get_summary('data/train_data.json')
    words = get_words(review_text)
    phi_x,phi_y,word_map,tot_unique_words=solve_with_summary(words,labels,summary_words)
    words_test = get_words(review_text_test)
    m_test = len(words_test)
    pred_y_test = get_prediction(phi_x,phi_y,m_test,word_map,words_test)
    print("Test Accuracy :",accuracy(pred_y_test,labels_test,m_test))

    confusion_matrx = get_confusion_matrix(pred_y_test,labels_test)
    f1 = get_f1(confusion_matrx)
    macro_f1 = np.average(f1)
    print("F1: ",f1)
    print("Macro F1: ",macro_f1)
    return

if __name__ == "__main__":
    print("START")
    if len(sys.argv) != 4:
        sys.stderr("Wrong format for command line arguments. Follow <file.py><train.json><test.json><part_number> ")
    train_file=str(sys.argv[1])
    review_text,labels=read(train_file)
    test_file=str(sys.argv[2])
    review_text_test, labels_test = read(test_file)
    part_num=str(sys.argv[3])
    if part_num=='a':
        part_A(review_text,labels,review_text_test,labels_test)
    elif part_num=='b':
        part_B(labels_test)
    elif part_num=='c':
        part_C(review_text,labels,review_text_test,labels_test)
    elif part_num=='d':
        part_D(review_text,labels,review_text_test,labels_test)
    elif part_num=='e':
        part_E(review_text,labels,review_text_test,labels_test)
    elif part_num=='g':
        part_G(review_text,labels,review_text_test,labels_test)






