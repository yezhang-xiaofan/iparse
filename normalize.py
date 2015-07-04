__author__ = 'zhangye'
import re

def normalize(input):
    p1 = re.compile("\d+\.*\d+\s*-\s*\d+\.*\d+")
    input = p1.sub(' MEASURE_RANGE ',input)

#match measure   eg. mug/d
    before_slash = ["g","mg","kg","ng","ug",'mug']
    after_slash = ["d","dl","h","day","ml"]
    before = '|'.join(before_slash)
    after = '|'.join(after_slash)
    p2 = re.compile('('+before+')'+'/'+'('+after+')',re.IGNORECASE)
    input = p2.sub(" MEASURE_UNIT ",input)

#match equal relationship
    p3 = re.compile("n\s*=\s*\d+")
    input = p3.sub(" EQUAL_REL ",input)

    #match measure  250 mg
    p4 = re.compile("(\d+\s*)"+before)
    input = p4.sub(" INTEGER MEASURE_UNIT ",input)


    #replace all integers
    p5 = re.compile("\d+")
    input = p5.sub("INTEGER", input)
    return  input
def main():
    str = 'A total of 127 patients with asymptomatic metastatic hormone\
        refractory prostate cancer (HRPC) were randomly assigned in a 2:1 ratio to\
        receive three infusions of sipuleucel-T (n = 82) or placebo (n = 45) every\
        2 weeks.'
    print normalize(str)

if __name__ == "__main__":
    main()