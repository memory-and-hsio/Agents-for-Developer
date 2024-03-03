import random

# create a function to generate randomized name list
def generate_name_list():
    # create a list of names
    names = ["John", "James", "Jack", "Jill", "Jane", "Jenny", "Jill", "Jude", "Jade", "Jasper", "Jasmine", "Jared", "Jen", "Jenifer", "Jeniffer"]
    # shuffle the list
    random.shuffle(names)
    # return the list
    return names


# create a function to generate randomized name list
def generate_name_list():
    # create a list of names
    names = ["John", "James", "Jack", "Jill", "Jane", "Jenny", "Jill", "Jude", "Jade", "Jasper", "Jasmine", "Jared", "Jen", "Jenifer", "Jeniffer"]
    # shuffle the list
    random.shuffle(names)
    # return the list
    return names

# design a function to generate a list of random names
def generate_name_list():
    # create a list of names
    names = ["John", "James", "Jack", "Jill", "Jane", "Jenny", "Jill", "Jude", "Jade", "Jasper", "Jasmine", "Jared", "Jen", "Jenifer", "Jeniffer"]
    # shuffle the list
    random.shuffle(names)
    # return the list
    return names

# design a function to sort name list in ascending order
def sort_name_list(names):
    # sort the list
    names.sort()
    # return the list
    return names

# create a function to do binary search on the list
def binary_search_name_list(names, name):
    # sort the list
    names.sort()
    # initialize the low and high
    low = 0
    high = len(names) - 1
    # loop through the list
    while low <= high:
        # calculate the mid
        mid = (low + high) // 2
        # check if the name is found
        if names[mid] == name:
            # return the index
            return mid
        # check if the name is in the left half
        elif names[mid] > name:
            high = mid - 1
        # check if the name is in the right half
        else:
            low = mid + 1
    # return -1 if the name is not found
    return -1

# test function for binary_search_name_list
def test_binary_search_name_list():
    # create a list of names
    names = ["John", "James", "Jack", "Jill", "Jane", "Jenny", "Jill", "Jude", "Jade", "Jasper", "Jasmine", "Jared", "Jen", "Jenifer", "Jeniffer"]
    # sort the list
    names.sort()
    # test the function
    assert binary_search_name_list(names, "John") == 7
    assert binary_search_name_list(names, "James") == 6
    assert binary_search_name_list(names, "Jack") == 5
    assert binary_search_name_list(names, "Jill") == 4
    assert binary_search_name_list(names, "Jane") == 3
    assert binary_search_name_list(names, "Jenny") == 2
    assert binary_search_name_list(names, "Jill") == 1
    assert binary_search_name_list(names, "Jude") == 0
    assert binary_search_name_list(names, "Jade") == 14
    assert binary_search_name_list(names, "Jasper") == 13
    assert binary_search_name_list(names, "Jasmine") == 12
    assert binary_search_name_list(names, "Jared") == 11
    assert binary_search_name_list(names, "Jen") == 10
    assert binary_search_name_list(names, "Jenifer") == 9
    assert binary_search_name_list(names, "Jeniffer") == 8
    assert binary_search_name_list(names, "Jeniffer") == -1

    