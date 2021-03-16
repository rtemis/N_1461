def parse(args):
    params = {}

    nargs = len(args)-1
    i = 0
    while i < nargs:
        if '-' != args[i+1][0]:
            params[args[i]] = args[i+1]
            i += 2
        else:
            i += 1 

    return params
