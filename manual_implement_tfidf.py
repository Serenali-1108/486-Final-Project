import csv
import ast  
import math

unique_words_list = []  
word_index_dict = {}  

i=0
with open('twitter.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        words_seen_this_line = set()
        word_list_str = row['tokens']
        word_list = ast.literal_eval(word_list_str)
        for word in word_list:
            if word not in words_seen_this_line:
                words_seen_this_line.add(word)
                if word not in unique_words_list:
                    unique_words_list.append(word)
                    word_index_dict[word] = (unique_words_list.index(word), 1)
                else:
                    # If the word is already in the dictionary, increment its line count（for getting its idf）
                    index, line_count = word_index_dict[word]
                    word_index_dict[word] = (index, line_count + 1)
        i+=1
        if i == 50000: #taking 1/10 of whole size
            break
uniqueWordCount=len(unique_words_list)

j = 0
with open('twitter.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    with open('added_TFIDF_Vector.csv', 'w', newline='') as modified_csv:
        fieldnames = reader.fieldnames + ['tfidfVector']    #add the new column
        writer = csv.DictWriter(modified_csv, fieldnames=fieldnames)    
        writer.writeheader()

        for row in reader:
            word_list_str = row['tokens']
            word_list = ast.literal_eval(word_list_str)
            tfidfVector = [0] * uniqueWordCount #init all position to 0
            for word in word_list:
                tf = word_list.count(word)
                indexInVector = word_index_dict[word][0]
                docFreq = word_index_dict[word][1]
                number = 50000/docFreq       #TODO: now set up 1/10 of entire size 
                idf = math.log(number, 10)
                tfidfVector[indexInVector] = round(tf * idf,3)
                row['tfidfVector'] = tfidfVector
                # print(tfidfVector)

            # Normalization
            sqrt_sum_of_squares = math.sqrt(sum(x ** 2 for x in tfidfVector if x != 0))
            new_value = [x / sqrt_sum_of_squares if x != 0 else 0 for x in tfidfVector]
            row['tfidfVector'] = new_value
            writer.writerow(row)

            j += 1
            if j == 50000:
                break
