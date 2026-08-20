


def get_stats(ids, countss):
    counts = {} if countss is None else countss
    for pair in zip(ids,ids[1:]):

        counts[pair]= counts.get(pair, 0)+1

    return counts

def merge(ids:list, max_pair, new_id):
    new_ids = []
    i = 0;
    while i<len(ids):
        if (i< len(ids)-1) and (ids[i],ids[i+1]) == max_pair:
            new_ids.append(new_id)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids
