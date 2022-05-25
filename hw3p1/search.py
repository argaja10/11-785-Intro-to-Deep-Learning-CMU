import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_path = []
    forward_prob = []
    for i in range(y_probs.shape[-1]):
        p = 1
        path = ['*']*y_probs.shape[-2]
        for j in range(y_probs.shape[-2]):
            temp = ['*']
            max_prob = 0
            for k in range(y_probs.shape[0]):
                if max_prob<y_probs[k][j][i]:
                    max_prob = y_probs[k][j][i]
                    if k!=0:
                        temp = SymbolSets[k-1]
                    else:
                        temp = ['*']
            path[j] = temp
            p = p*max_prob
    forward_path.append(path)
    forward_prob.append(p)
    
    for i in range(y_probs.shape[-1]):
        compressed_path = ''
        temp = 0
        for j in range(y_probs.shape[-2]):           
            if temp != 0 and forward_path[i][j] == temp:
                continue
            if forward_path[i][j] == ['*']:
                temp = 0
                continue
            compressed_path = compressed_path + forward_path[i][j]
            temp = forward_path[i][j]
        forward_path.append(compressed_path)

    return forward_path[-1], forward_prob[0]
    #raise NotImplementedError


##############################################################################
def InitializePaths(SymbolSets, y):
    InitialBlankPathScore, InitialPathScore = {}, {}
    path = ''
    InitialBlankPathScore[path] = y[0] 
    InitialPathsWithFinalBlank = {path}

    InitialPathsWithFinalSymbol = set()
    for i in range(len(SymbolSets)):
        path = SymbolSets[i]
        InitialPathScore[path] = y[i + 1]
        InitialPathsWithFinalSymbol.add(path)  
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}

   
    for path in PathsWithTerminalBlank:
        
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]
    
    for path in PathsWithTerminalSymbol:
       
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] =  UpdatedBlankPathScore[path] + PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}

    
    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)): 
            newpath = path + SymbolSet[i]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

    
    for path in PathsWithTerminalSymbol:
        for i in range(len(SymbolSet)):
            newpath = path if (SymbolSet[i] == path[-1]) else path + SymbolSet[i] 
            if newpath in UpdatedPathsWithTerminalSymbol: 
                UpdatedPathScore[newpath] = UpdatedPathScore[newpath] + PathScore[path] * y[i+1]
            else: 
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore = {}, {}
    PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = set(), set()
    scorelist = []
    
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    
    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth] if (BeamWidth < len(scorelist)) else scorelist[-1]

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
    
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

  
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] = FinalPathScore[p] + BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore




def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    PathScore = {}
    BlankPathScore = {} 
    num_symbols, seq_len, batch_size = y_probs.shape

   
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:, 0, :])

    
    for t in range(1, seq_len):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                           NewPathsWithTerminalSymbol,
                                                                                           NewBlankPathScore, NewPathScore,
                                                                                           BeamWidth)

        NewPathsWithTerminalBlank, NewBlankPathScore =  ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :], BlankPathScore, PathScore)

       
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:, t, :], BlankPathScore, PathScore)

    
    MergedPaths, mergedPathScores = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)


   
    bestPath = max(mergedPathScores, key=mergedPathScores.get) 
    return (bestPath, mergedPathScores)
    #raise NotImplementedError
