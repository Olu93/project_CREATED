import itertools as it

set_objects_2 = 'AB'
set_objects_3 = 'ABC'
set_objects_4 = 'ABCD'


def generate_production_set(set_objects):
    return set(
    list(it.chain(*[list(it.permutations(obj)) for obj in list(it.chain(*[list(it.combinations_with_replacement(set_objects, i)) for i in range(0,
                                                                                                                                                  len(set_objects) + 1)]))])))


x = generate_production_set(set_objects_3)

def check_if_c_before_d(s):
    if 'D' not in s:
        return True
    for i in reversed(range(len(s))):
        if s[i] == 'D':
            if s[i - 1] != 'C':
                return False
    return True

def check_if_d_after_c(s):
    if 'C' not in s:
        return True
    for i in range(len(s)):
        if s[i] == 'C':
            if i == len(s)-1:
                return False
            if s[i + 1] != 'D':
                return False
    return True

without_restrict = generate_production_set(set_objects_4)
with_restrict = [
    s for s in generate_production_set(set_objects_4)
    if check_if_d_after_c(s)
]
with_restrict