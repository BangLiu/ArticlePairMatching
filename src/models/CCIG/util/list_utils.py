# coding=utf-8


def common(list1, list2):
    return list(set(list1).intersection(list2))


def substract(list1, list2):
    return list(set(list1) - set(list2))


def remove_values_from_list(l, val):
    return [value for value in l if value != val]
