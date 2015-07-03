__author__ = 'zhangye'
#this program converts a file into a list of sentences with labels
#input directory is "abstracts2"
#output directoy is "abstracts2_sen"
import os
dir = "abstracts2/"
output_dir = "abstracts2_sen/"
def process(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace("- - !!python/unicode","")
    return sentence.strip()
def file_sen(dir=dir):
    for file in os.listdir(dir):
        has_iparse = False
        with open (dir+file,"rb") as cur_file:
            #print "haha"
            sentences = []
            labels = []
            current_sen = ""
            sen_no = 0
            for c in cur_file:
                if("sents:" in c) : continue
                if("tags: []" in c and "- " not in c): break
                if("- - !!python/unicode" in c or '- - "' in c):
                    if(sen_no==0):
                        current_sen += c.strip()
                    else:
                        sentences.append(current_sen)
                        current_sen = c.strip()
                    sen_no += 1
                elif("- tags: " in c):
                    if("iparse" in c):
                        labels.append(1)
                        has_iparse = True
                    else:
                        labels.append(0)
                else:
                    current_sen += c.strip()
            sentences.append(current_sen)

        if has_iparse==True:
            write_file = open(output_dir+file,'wb')
            for i,s in enumerate(sentences):
                write_file.write(process(s)+"\t"+str(labels[i])+"\n")
            write_file.close()

if __name__ == '__main__':
    file_sen()