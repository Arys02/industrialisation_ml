def data_clean(data : str) -> (str, bool):
    error_table = {
        "Tomate": "Tomato",
        "Tomatto": "Tomato",
        "tomaot": "Tomato",
        "tomato": "Tomato",
        "Poire": "Peer",
        "Pera": "Peer",
        "Brussell Sprout" : "Brussel Sprout",
        "Brusselsprout" : "Brussel Sprout",
        "Brusel Sprout" : "Brussel Sprout",
    }

    if data in error_table.keys():
        return error_table[data], True
    return data, False