#
import argparse
import string
import math
import json

argParser = argparse.ArgumentParser()
argParser.add_argument('--infile', action='store', required=True, help="Source file of sentences.")
argParser.add_argument('--prefix', action='store', required=True, help="Output file prefix.")
argParser.add_argument('--field', action='store', required=True, help="Field to output")
argParser.add_argument('--blanks', dest='blanks', action='store_true', default=False, help="If present lines separated by blank line.")
argParser.add_argument('--count', action='store', type=int, required=True, help="Total number of lines in input file.")  #  yeah, I could read it twice or read into a pandas. . .

args = argParser.parse_args()

field = args.field
count = args.count

#  Assume no blank line between rows (needed for pretraining only)
suffix="\n"
if( args.blanks ):  #  so will be used for pretraining
    suffix="\n\n"

trainFile = open( args.prefix+".train.txt", "w" )  #  output files, train, test, validate
testFile = open( args.prefix+".test.txt", "w" )    #  truncate what's there
validFile = open( args.prefix+".valid.txt", "w" )

curNdx = 0
validStart = math.ceil( count * 0.9 )  #  so consistently get the next integer
wroteTest = False  #  write the single line test row first, then set to true

with open( args.infile ) as infile:
    for curLine in infile:
        curJSON = json.loads( curLine )
        curVal = curJSON[ field ]
#  this was a boondoggle        filteredVal = ''.join( filter(lambda x:x in string.printable, curVal ) )  #  remove non-printable characters, which I hope is causing a problem in fairseq-train
        if( curNdx < validStart ):  #  Write to train
            trainFile.write( curVal + suffix )
        elif( not wroteTest ):
            testFile.write( curVal + suffix )
            wroteTest = True
        else:
            validFile.write( curVal + suffix )
        curNdx += 1

infile.close()  #  close the files so they are flushed properly
trainFile.close()
testFile.close()
validFile.close()


