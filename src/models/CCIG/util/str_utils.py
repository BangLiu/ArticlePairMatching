# coding:utf-8


def longestCommonPrefix(strs):
    """
    Get the longest prefix of a list of strings.
    """
    if len(strs) == 0:
        return ""
    str = strs[0]
    Min = len(str)
    for i in range(1, len(strs)):
        j = 0
        p = strs[i]
        while j < Min and j < len(p) and p[j] == str[j]:
            j += 1
        Min = Min if Min < j else j
    return str[:Min]
