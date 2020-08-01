#Roberta.large.mnli inference on GLUE switched dev matched text dataset

import torch
# Download RoBERTa already finetuned for MNLI

# This code sample uses roberta.large.mnli but for custom models, please update the following line

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

# This is the output file where initial column data as well as new predicted column data is stored

outFile = open( "rob-large-mnli-infered-switched-dev-matched-all.txt", "w" )  #  output has all the columns

ncorrect, nsamples= 0,0

predicted_neutral, predicted_contradiction, predicted_entailment= 0,0,0

total_neutral,total_contradiction, total_entailment = 0,0,0

oneutral, oentailment, ocontradiction = 0,0,0

neutral_neutral, neutral_contra, neutral_entail = 0,0,0
contra_neutral, contra_contra, contra_entail = 0,0,0
entail_neutral, entail_contra, entail_entail = 0,0,0


with open('/opt/models/data-bin/switched-dev-matched-mnli/switched-dev-matched.txt') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target, genre = tokens[0], tokens[1], tokens[2], tokens[3]

        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
        outFile.write( sent1 + "\t" + sent2 + "\t" + target + "\t" + genre + "\t" + prediction_label + "\n")

        if target == "neutral":
            total_neutral += 1
            if prediction_label == "neutral":
                predicted_neutral += 1
                neutral_neutral +=1
            if prediction_label == "contradiction":
                predicted_contradiction += 1
                neutral_contra +=1
            if prediction_label == "entailment":
                predicted_entailment += 1
                neutral_entail +=1
                
                
        if target == "contradiction":
            total_contradiction += 1
            if prediction_label == "neutral":
                predicted_neutral += 1
                contra_neutral +=1
            if prediction_label == "contradiction":
                predicted_contradiction += 1
                contra_contra +=1
            if prediction_label == "entailment":
                predicted_entailment += 1
                contra_entail +=1

        if target == "entailment":
            total_entailment += 1
            if prediction_label == "neutral":
                predicted_neutral += 1
                entail_neutral +=1
            if prediction_label == "contradiction":
                predicted_contradiction += 1
                entail_contra +=1
            if prediction_label == "entailment":
                predicted_entailment += 1
                entail_entail +=1
                        
print('| Accuracy: ', float(ncorrect)/float(nsamples))
#print('| Neutral performance:', float(nneutral)/float(tneutral))
#print('| Contradiction performance:', float(ncontradiction)/float(tcontradiction))
#print('| Entailment performance:', float(nentailment)/float(tentailment))

print('| Entailment:Total Entailment:{}, Entail Entailments: {} {}, Entail Neutral: {}, Entail Contras: {} '.format(total_entailment, float(entail_entail)/float(total_entailment), entail_entail, entail_neutral, entail_contra))
print('| Contradiction:Total Contradiction:{}, Contradiction Entailments: {}, Contradiction Neutral: {}, Contradiction Contras: {} '.format(total_contradiction, contra_entail, contra_neutral, contra_contra))
print('| Neutral:Total Neutral:{}, Neutral Entailments: {}, Neutral Neutral: {}, Neutral Contras: {} '.format(total_neutral, neutral_entail, neutral_neutral, neutral_contra))

