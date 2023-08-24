def spNOT(inputsp):
    return 1-inputsp

def spAND(inputsp_list):
    output = 1
    for inputsp in inputsp_list:
        output *= inputsp
    return output

def spOR(inputsp_list):
    output = 1
    for inputsp in inputsp_list:
        output *= (1-inputsp)
    return 1 - output

def sp2XOR(inputsp1, inputsp2):
    output = (1 - inputsp1)*inputsp2 + inputsp1 * (1 - inputsp2)
    return output

def spXOR(inputsp_list):
    output = sp2XOR(inputsp_list[0], inputsp_list[1])
    for i in range(2, len(inputsp_list)):
        output = sp2XOR(output, inputsp_list[i])
    return output

def spNAND(inputsp_list):
    output = (1 - spAND(inputsp_list))
    return output

def spNOR(inputsp_list):
    output = (1 - spOR(inputsp_list))
    return output

def spXNOR(inputsp_list):
    output = (1 - spXOR(inputsp_list))
    return output

def switchingActivity(output):
    return 2 * output * (1-output)
