class A:
    def __init__(self, i: int):
        self.i: int = i


class B:
    # No W0201 if B has the same field `i` as A
    pass


class Test:
    a: A | B

    def __init__(self) -> None:

        # no branching -> no w0201
        # self.a = A(i=1)

        # branching -> w0201
        if True:  # same result if an actual boolean is used
            self.a: A = A(i=1)  # same result without type hint
        else:  # No W0201 if the `else` branch is removed
            self.a: B = B()

    def c(self) -> None:
        if not isinstance(self.a, A):
            raise TypeError("self.a is not an instance of A")
        d: A = self.a  # same result without type hint
        d.i = 2  # W0201 warning here!


# [{
# 	"resource": <file path redacted>,
# 	"owner": "_generated_diagnostic_collection_name_#6",
# 	"code": {
# 		"value": "W0201:attribute-defined-outside-init",
# 		"target": {
# 			"$mid": 1,
# 			"path": "/en/latest/user_guide/messages/warning/attribute-defined-outside-init.html",
# 			"scheme": "https",
# 			"authority": "pylint.readthedocs.io"
# 		}
# 	},
# 	"severity": 4,
# 	"message": "Attribute 'i' defined outside __init__",
# 	"source": "Pylint",
# 	"startLineNumber": 29,
# 	"startColumn": 9,
# 	"endLineNumber": 29,
# 	"endColumn": 12,
# 	"modelVersionId": 893,
# 	"origin": "extHost2"
# }]
