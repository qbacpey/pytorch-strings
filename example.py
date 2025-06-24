from string_tensor import *

if __name__ == "__main__":
    StringTensorClasses: list[type[StringColumnTensor]] = [PlainEncodingStringColumnTensor, CPlainEncodingStringColumnTensor, DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor, UnsortedDictionaryEncodingStringColumnTensor, UnsortedCDictionaryEncodingStringColumnTensor]

    plain: PlainEncodingStringColumnTensor = PlainEncodingStringColumnTensor.Encoder.encode(
        [
            "apwho",
            "Initially",
            "apple",
            "applppp",
            "bpple",
            "each",
            "encoding",
            "method",
            "will",
            "will",
            "be",
            "applppp",
        ]
    )
    for cls in StringTensorClasses:
        col = cls.from_string_tensor(plain)

        # Query for equality
        row_ids = col.query_equals("will")
        print("Row IDs for 'will':", row_ids)

        # Query for less than
        row_ids_lt = col.query_less_than("bpple")
        print("Row IDs for strings less than 'bpple':", row_ids_lt)

        # Query for prefix
        row_ids_prefix = col.query_prefix("ap")
        print("Row IDs for strings starting with 'ap':", row_ids_prefix)

        print(col.to_strings())
