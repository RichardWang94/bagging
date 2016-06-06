from __future__ import division
import sys
import numpy as np

def calculate_metrics(tp, fp, tn, fn, num1, sum_of_other):

    tpr = float(tp) / num1
    fpr = float(fp) / (sum_of_other)
    errorRate = (fp + fn) / float(num1 + sum_of_other)
    accuracy = (tp + tn) / float(num1 + sum_of_other)
    precision = float(tp) / (tp + fp)

    return tpr, fpr, errorRate,accuracy, precision


def readData(filename):

    dataFile = open(filename, 'r')

    line = dataFile.readline()
    specs = map(int, line.split())

    data = np.genfromtxt(dataFile)
    dataFile.close()

    return (data, specs)

def calculate_centroids(data, bootstrap, nA, nB, d, numPoints):
    classAData = np.empty([0, d])
    classBData = np.empty([0, d])

    for index in range(numPoints):
        if (bootstrap[index] < nA):
            classAData = np.vstack([classAData, data[index]])
        else:
            classBData = np.vstack([classBData,data[index]])

    #print "classAData"
    #print classAData
    #print "classBData"
    #print classBData

    centroidA = np.mean(classAData, axis=0)
    centroidB = np.mean(classBData, axis=0)
    if (classBData.shape[0] == 0):
        return (centroidA, centroidB, 1)
    if (classAData.shape[0] == 0):
        return (centroidA, centroidB, 0)
    return (centroidA, centroidB, -1)

def findDecisionFunction(centroidA, centroidB):

    line = np.subtract(centroidB, centroidA)
    midpoint = np.divide(np.add(centroidB, centroidA), 2)

    decision_value = decision_function(line, centroidA, midpoint)
    sign = decision_value / abs(decision_value)

    #Just a function with values stored in it
    #Use +1 or -1 to find classes

    def classAorB(point_nd):
        value = decision_function(line, point_nd, midpoint)
        if value == 0:
            decision = 1
        else:
            decision = value * sign
        return decision / abs(decision)

    return classAorB

def decision_function(line, point_nd, midpoint):
    return np.dot(line, np.subtract(point_nd, midpoint))

def classify(training_data, bootstrap, nA, nB, d, numPoints, test_file, numA, numB):

    (centroidA, centroidB, flag) = calculate_centroids(training_data, bootstrap, nA,nA+nB, d, numPoints)
    myClass = list()
    if (flag == -1):
        classAOrB = findDecisionFunction(centroidA, centroidB)
        for i in range(numA + numB):
            line = test_file.readline()
            point = map(float, line.split())
            if classAOrB(point) == 1:
                myClass.append(1)
            else:
                myClass.append(0)
        return myClass
    elif(flag == 1):
        for i in range(numA + numB):
            myClass.append(1)
    elif(flag == 0):
        for i in range(numA + numB):
            myClass.append(0)
    """
    keys =['A','B']

    truePositives = {key: 0 for key in keys}
    trueNegatives = {key: 0 for key in keys}
    falsePositives = {key: 0 for key in keys}
    falseNegatives = {key: 0 for key in keys}

    # Test A
    for i in range(numA):
        line = test_file.readline()
        point = map(float, line.split())
        if classAOrB(point) == 1:
            trueNegatives['B'] += 1
            if classAOrC(point) == 1:
                truePositives['A'] += 1
            else:  # C
                falseNegatives['A'] += 1

    for i in range(numB):
        line = test_file.readline()
        point = map(float, line.split())
        if classAOrB(point) == 1:
            falseNegatives['B'] += 1
            if classAOrC(point) == 1:
                falsePositives['A'] += 1
            else:  # C
                trueNegatives['A'] += 1
    test_file.close()

    #Calculate Metrics
    truePositiveRateA, falsePositiveRateA, errorRateA,accuracyA, precisionA = calculate_metrics(truePositives['A'], falsePositives['A'],trueNegatives['A'],falseNegatives['A'],numA,numB)
    truePositiveRateB, falsePositiveRateB, errorRateB,accuracyB, precisionB = calculate_metrics(truePositives['B'], falsePositives['B'],trueNegatives['B'],falseNegatives['B'],numB,numA)

    truePositiveRate = (truePositiveRateA + truePositiveRateB) / 2.0
    falsePositiveRate = (falsePositiveRateA + falsePositiveRateB) / 2.0
    errorRate = (errorRateA + errorRateB) / 2.0
    accuracy = (accuracyA + accuracyB) / 2.0
    precision = (precisionA + precisionB) / 2.0

    print "True positive rate = %f" % truePositiveRate
    print "False positive rate = %f" % falsePositiveRate
    print "Error rate = %f" % errorRate
    print "Accuracy = %f" % accuracy
    print "Precision = %f" % precision
    """
if __name__ == "__main__":

    if (len(sys.argv) != 5):
        if (len(sys.argv) != 6):
            print "Received wrong number of arguments.\nUsage: \'python bagit [-v] T size <train> <test>\'"
            sys.exit()

    verboseflag = False;
    if (sys.argv[1] == '-v'):
        verboseflag = True;
        numClassifiers = int(sys.argv[2]);
        numPoints = int(sys.argv[3]);
        training_file = sys.argv[4];
        testing_file = sys.argv[5];
    else:
        numClassifiers = int(sys.argv[1]);
        numPoints = int(sys.argv[2]);
        training_file = sys.argv[3];
        testing_file = sys.argv[4];

    data, (d, a,b) = readData(training_file)
  #  print data
  #  print ""
    bootstrapSamples = np.empty([numClassifiers, numPoints], dtype=int)
    for currClassifier in range(numClassifiers):
        bootstrapSamples[currClassifier] = np.random.choice(a + b, numPoints, replace=True)
        #print "bootstrapSamples"
        #print bootstrapSamples[currClassifier]
        #  print ""
    test_file = open(testing_file, 'r')
    line = test_file.readline()
    (dim, numA, numB) = map(int, line.split())
    sumClass = list()
    for classifiers in range(numClassifiers):
        bsSamples = np.empty([numPoints, d])
        for dataPoint in range(numPoints):
            bsSamples[dataPoint] = data[bootstrapSamples[classifiers][dataPoint]]
        if classifiers != 0:
            test_file = open(testing_file, 'r')
            test_file.readline()
        myClass = classify(bsSamples, bootstrapSamples[classifiers], a, b, d, numPoints, test_file, numA, numB)
        if classifiers == 0:
            sumClass = myClass
        else:
            for index in range(len(myClass)):
                sumClass[index] += myClass[index]
    sumClassFloats = [float(i) for i in sumClass]
    for index in range(len(sumClassFloats)):
        sumClassFloats[index] = sumClassFloats[index]/float(numClassifiers)
    #print sumClassFloats
    falsepositives = 0
    falsenegatives = 0
    for index in range(len(sumClassFloats)):
        if ((sumClassFloats[index] < 0.5) & (index < numA)):
            falsenegatives += 1
        if ((sumClassFloats[index] >= 0.5) & (index >= numA)):
            falsepositives += 1
    positiveExamples = numA
    negativeExamples = numB
    print "Positive examples: %d" % positiveExamples
    print "Negative examples: %d" % negativeExamples
    print "False positives: %d" % falsepositives
    print "False negatives: %d" % falsenegatives
    if verboseflag:
        for setNum in range(numClassifiers):
            print ""
            print "Bootstrap sample set %d:" % (setNum + 1)
            for index in range(numPoints):
                for currd in range(d):
                    sys.stdout.write("%f "%data[bootstrapSamples[setNum][index]][currd])
                sys.stdout.write("- ")
                if bootstrapSamples[setNum][index] < a:
                    print "True"
                else:
                    print "False"
        print ""
        print "Classification: "
        test_file = open(testing_file, 'r')
        line = test_file.readline()
        (dim, numA, numB) = map(int, line.split())
        for i in range(numA + numB):
            line = test_file.readline()
            point = map(float, line.split())
            for j in range(d):
                sys.stdout.write("%f "%point[j])
            sys.stdout.write("- ")
            if ((sumClassFloats[i] < 0.5) & (i < numA)):
                print "False (false negative)"
            elif ((sumClassFloats[i] >= 0.5) & (i >= numA)):
                print "True (false positive)"
            elif (sumClassFloats[i] < 0.5):
                print "False (correct)"
            else:
                print "True (correct)"
