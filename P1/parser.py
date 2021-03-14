
def parse(args):
    params = {}

    for i in range(0, len(args)-1, 2):
        params[args[i]] = args[i+1]
    
    return params

