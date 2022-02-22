import itertools as it


set_objects_2 = 'AB'
set_objects_3 = 'ABC'
set_objects_4 = 'ABCD'

x = set(
    list(it.chain(*[list(it.permutations(obj)) for obj in list(it.chain(*[list(it.combinations_with_replacement(set_objects_3, i)) for i in range(0,
                                                                                                                                                  len(set_objects_3) + 1)]))])))


def check_if_c_after_d(s):
    if 'D' not in s:
        return True
    for i in reversed(range(len(s))):
        if s[i] == 'D':
            if s[i - 1] != 'C':
                return False
    return True


with_restrict = [
    s for s in set(
        list(
            it.chain(*[list(it.permutations(obj)) for obj in list(it.chain(*[list(it.combinations_with_replacement(set_objects_4, i)) for i in range(0,
                                                                                                                                                     len(set_objects_4) + 1)]))])))
    if check_if_c_after_d(s)
]
