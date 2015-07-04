__author__ = 'zhangye'
import re
import drugbank
def normalize(input,drug):
    #replace drug name by "DRUG"
    input = input.decode('unicode-escape')
    input = input.encode('ascii','ignore')

    input = drug.sub(input)

    p1 = re.compile("\d+\.*\d+\s*-\s*\d+\.*\d+")
    input = p1.sub(' MEASURE_RANGE ',input)

#match measure   eg. mug/d
    before_slash = ["g","mg","kg","ng","ug",'mug','microg','mcg']
    after_slash = ["d","dl","h","day","ml",'kg']
    before = '|'.join(before_slash)
    after = '|'.join(after_slash)
    p2 = re.compile('('+before+')'+'/'+'('+after+')',re.IGNORECASE)
    input = p2.sub(" MEASURE_UNIT ",input)

    #match equal relationship
    p3 = re.compile("n\s*=\s*\d+")
    input = p3.sub(" EQUAL_REL ",input)

    #match measure  250 mg
    p4 = re.compile("\d+\.*\d+\s*"+"("+before+")")
    input = p4.sub(" INTEGER MEASURE_UNIT ",input)

    #match ratio
    p8 = re.compile("\d+\s*:\s*\d+")
    input = p8.sub(" RATIO_POP ",input)

    #replace percentage
    p7 = re.compile("\d+\.\d+%")
    input = p7.sub(" PERCENTAGE ",input)

    #replace all float numbers
    p6 = re.compile(r"\b\d+\.\d+\b")
    input = p6.sub("FLOAT",input)

    #replace all integers
    p5 = re.compile(r"\b\d+\b")
    input = p5.sub("INTEGER", input)



    return  input
def main():
    drug = drugbank.Drugbank()
    str = "Healthy adults, stratified by age (18-64 years and \u226565 years), were\
        \ randomized (1:1 allocation), in a double-blind, parallel-group design, to\
        \ receive two intramuscular doses (21 days apart) of vaccine containing approximately\
        \ 15 \u03BCg or 30 \u03BCg of hemagglutinin (HA)."

    print normalize(str,drug)

if __name__ == "__main__":
    main()